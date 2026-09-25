// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const env_util = @import("../env.zig");

const worker_slot_cache_line_bytes: usize = 64;
const worker_slot_padding_bytes: usize = if (@sizeOf(std.atomic.Value(u32)) < worker_slot_cache_line_bytes)
    (worker_slot_cache_line_bytes - @sizeOf(std.atomic.Value(u32)))
else
    0;

threadlocal var tls_current_pool_token: usize = 0;
threadlocal var tls_current_tid: usize = 0;

// TODO: Maybe accept an IO impl?
fn futexIo() std.Io {
    return std.Io.Threaded.global_single_threaded.io();
}

/// A persistent, deterministic thread pool designed for numerical kernels.
///
/// Design goals (v0):
/// - No allocations during `parallelForAny`.
/// - Deterministic work partitioning (static contiguous ranges per tid).
/// - Low overhead: Uses Atomic + Futex for synchronization.
/// - Same-pool overlapping host submissions are serialized internally.
/// - Same-pool nested submissions run inline on the calling tid.
///
/// Terminology:
/// - `thread_count_total` includes the calling thread (tid=0) plus worker threads (tid=1..).
/// How long a worker spins for the next job, and the submitter for the workers,
/// before falling back to a futex sleep.
///
/// The two are not the same trade. A spinning WORKER competes for a core with a
/// worker still running the current job, so a long budget is actively harmful on
/// a graph of long steps: at ~1 ms VGG-19 on ten threads went from 37.6 ms to
/// 54.8 ms an image. A spinning SUBMITTER has nothing else to do — it has
/// finished its own share and is waiting on the slowest worker — so it only risks
/// the core it already holds, and sleeping there costs a syscall plus a wake on
/// every one of a decode step's ~150 dispatches. Hence short for workers, long
/// for the submitter; the latter is worth ~1.5% at eight threads.
/// `publishTo` keeps the wake side syscall-free in the common case.
const WORKER_SPIN_NS: u64 = 50 * std.time.ns_per_us;
const SUBMIT_SPIN_NS: u64 = 400 * std.time.ns_per_us;

/// A spin budget in time, not iterations: one `spinLoopHint` is ~11 ns on an
/// Apple core and several times that on recent x86, so a count would mean a
/// different wait on every CPU. The clock is read only once the wait outlasts a
/// few spins, so a job that is already there costs nothing extra.
const Spin = struct {
    budget_ns: u64,
    deadline: u64 = 0,
    spins: u32 = 0,

    const CHECK_EVERY: u32 = 16;

    /// Spin once; false once the budget is spent and the caller should sleep.
    fn again(self: *Spin) bool {
        std.atomic.spinLoopHint();
        self.spins +%= 1;
        if (self.spins % CHECK_EVERY != 0) return true;
        const t = nowNs();
        if (self.deadline == 0) self.deadline = t + self.budget_ns;
        return t < self.deadline;
    }
};

fn nowNs() u64 {
    return @intCast(std.Io.Clock.awake.now(futexIo()).toNanoseconds());
}

pub const ThreadPool = struct {
    allocator: std.mem.Allocator,

    worker_threads: []std.Thread,
    thread_count_total: usize,

    shared: ?*Shared = null,
    submit_mutex: std.Io.Mutex = .init,

    pub const Options = struct {
        /// Total threads to use including the calling thread.
        /// Set to 1 to disable parallelism.
        thread_count: usize = 1,
    };

    const WorkerSlot = struct {
        job_seq: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
        /// Set by the worker just before it sleeps on `job_seq`, so a publisher
        /// can skip the wake syscall for a worker that is still spinning. A decode
        /// step publishes hundreds of times per token; waking unconditionally cost
        /// a syscall per worker per publish whether or not anyone was asleep.
        sleeping: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
        /// The last sequence this worker finished. The submitter waits on these
        /// rather than on one shared counter: seven workers decrementing a single
        /// line hand its ownership around seven times per join, where seven lines
        /// each stay with their own worker.
        done_seq: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
        /// Mirror of `sleeping` for the other direction: set while the submitter
        /// is parked on `done_seq`, so a finishing worker skips the wake syscall
        /// in the overwhelmingly common case where the submitter is still spinning.
        submit_sleeping: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),
        _padding: [worker_slot_padding_bytes - 2 * @sizeOf(std.atomic.Value(bool)) - @sizeOf(std.atomic.Value(u32))]u8 = @splat(0),
    };

    comptime {
        std.debug.assert(@sizeOf(WorkerSlot) == worker_slot_cache_line_bytes);
    }

    const Shared = struct {
        // Synchronization primitives
        // Per-worker job sequence (futex address). Only workers that will execute
        // work for the current job are woken.
        worker_slots: []align(worker_slot_cache_line_bytes) WorkerSlot = &[_]WorkerSlot{},
        /// Next unclaimed index, for jobs that hand work out on demand.
        next_index: std.atomic.Value(usize) = std.atomic.Value(usize).init(0),
        dynamic: bool = false,
        ready_workers: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
        stop: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),

        // Job payload
        ctx: *anyopaque = undefined,
        n: usize = 0,
        grain: usize = 0,
        func: *const fn (ctx: *anyopaque, start: usize, end: usize, tid: usize) void = undefined,

        // Threads used for the current job (includes main thread tid=0).
        job_thread_count_total: usize = 1,
    };

    const WorkerArgs = struct {
        shared: *Shared,
        tid: usize,
    };

    pub fn init(allocator: std.mem.Allocator, opts: Options) !ThreadPool {
        if (opts.thread_count == 0) return error.InvalidArgument;

        var pool: ThreadPool = .{
            .allocator = allocator,
            .worker_threads = &[_]std.Thread{},
            .thread_count_total = opts.thread_count,
            .shared = null,
        };

        if (opts.thread_count <= 1) {
            pool.thread_count_total = 1;
            return pool;
        }

        const shared: *Shared = try allocator.create(Shared);
        errdefer allocator.destroy(shared);
        shared.* = .{};
        pool.shared = shared;

        const worker_count: usize = opts.thread_count - 1;
        pool.worker_threads = try allocator.alloc(std.Thread, worker_count);
        errdefer allocator.free(pool.worker_threads);

        // Allocate cache-line-separated worker wakeup slots to avoid false sharing
        // between futex words for adjacent workers.
        shared.worker_slots = try allocator.alignedAlloc(
            WorkerSlot,
            std.mem.Alignment.fromByteUnits(worker_slot_cache_line_bytes),
            worker_count,
        );
        errdefer allocator.free(shared.worker_slots);
        for (shared.worker_slots) |*slot| slot.* = .{};

        var i: usize = 0;
        errdefer {
            shared.stop.store(true, .release);
            for (shared.worker_slots[0..i]) |*slot| {
                _ = slot.job_seq.fetchAdd(1, .release);
                futexIo().futexWake(u32, &slot.job_seq.raw, 1);
            }
            for (pool.worker_threads[0..i]) |thread| thread.join();
        }
        while (i < worker_count) : (i += 1) {
            const args: WorkerArgs = .{ .shared = shared, .tid = i + 1 };
            pool.worker_threads[i] = try std.Thread.spawn(.{}, workerMain, .{args});
        }

        // Barrier: ensure all workers have entered workerMain and are ready to observe the
        // first job. Without this, a job launched immediately after init could be missed by
        // workers that haven't started waiting yet, causing a hang.
        var ready: u32 = shared.ready_workers.load(.acquire);
        while (ready < @as(u32, @intCast(worker_count))) {
            futexIo().futexWaitUncancelable(u32, &shared.ready_workers.raw, ready);
            ready = shared.ready_workers.load(.acquire);
        }

        return pool;
    }

    pub fn deinit(self: *ThreadPool) void {
        std.Io.Threaded.mutexLock(&self.submit_mutex);
        defer std.Io.Threaded.mutexUnlock(&self.submit_mutex);

        const shared = self.shared orelse {
            if (self.worker_threads.len > 0) self.allocator.free(self.worker_threads);
            self.thread_count_total = 1;
            return;
        };

        // Signal stop
        shared.stop.store(true, .release);
        // Wake all workers (each waits on its own futex word).
        var idx: usize = 0;
        while (idx < shared.worker_slots.len) : (idx += 1) {
            // Unconditional here: shutdown must not race a worker into sleeping.
            const seq = workerJobSeq(shared, idx);
            const v = seq.load(.acquire);
            seq.store(v + 1, .release);
            futexIo().futexWake(u32, &seq.raw, 1);
        }

        for (self.worker_threads) |t| {
            t.join();
        }

        if (self.worker_threads.len > 0) {
            self.allocator.free(self.worker_threads);
        }
        if (shared.worker_slots.len > 0) {
            self.allocator.free(shared.worker_slots);
            shared.worker_slots = &[_]WorkerSlot{};
        }
        self.worker_threads = &[_]std.Thread{};
        self.thread_count_total = 1;
        self.shared = null;
        self.allocator.destroy(shared);
    }

    pub fn threadCount(self: *const ThreadPool) usize {
        return self.thread_count_total;
    }

    /// Deterministic parallel-for over `[0..n)`.
    ///
    /// `grain` is a hint for splitting each per-thread contiguous range into smaller
    /// sequential chunks (still deterministic, no stealing). Overlapping host
    /// submissions to the same pool are serialized, and nested same-pool calls run
    /// inline on the calling tid to avoid deadlock on the shared job slot.
    /// Fallible jobs use the same scheduler and join semantics. Each invocation
    /// owns its failure slot, including nested and concurrent submissions.
    pub fn parallelForFallible(
        self: *ThreadPool,
        comptime E: type,
        ctx: *anyopaque,
        n: usize,
        grain: usize,
        func: *const fn (*anyopaque, usize, usize, usize) E!void,
    ) E!void {
        const Job = struct {
            ctx: *anyopaque,
            func: *const fn (*anyopaque, usize, usize, usize) E!void,
            failure: std.atomic.Value(u32) = .init(0),

            fn run(raw: *anyopaque, start: usize, end: usize, tid: usize) void {
                const job: *@This() = @ptrCast(@alignCast(raw));
                job.func(job.ctx, start, end, tid) catch |err| {
                    _ = job.failure.cmpxchgStrong(0, @intFromError(err), .release, .monotonic);
                };
            }
        };
        var job: Job = .{ .ctx = ctx, .func = func };
        self.parallelForAny(&job, n, grain, Job.run);
        const failure = job.failure.load(.acquire);
        if (failure != 0) return @errorCast(@errorFromInt(@as(u16, @intCast(failure))));
    }

    /// Like `parallelForAny`, but threads claim ranges on demand rather than each
    /// taking a fixed share — for jobs whose cost per index is uniform but whose
    /// cores are not. The split is therefore not reproducible; every index still
    /// runs exactly once, on exactly one thread.
    pub fn parallelForDynamic(
        self: *ThreadPool,
        ctx: *anyopaque,
        n: usize,
        grain: usize,
        func: *const fn (ctx: *anyopaque, start: usize, end: usize, tid: usize) void,
    ) void {
        self.runParallel(ctx, n, grain, func, true);
    }

    pub fn parallelForAny(
        self: *ThreadPool,
        ctx: *anyopaque,
        n: usize,
        grain: usize,
        func: *const fn (ctx: *anyopaque, start: usize, end: usize, tid: usize) void,
    ) void {
        self.runParallel(ctx, n, grain, func, false);
    }

    fn runParallel(
        self: *ThreadPool,
        ctx: *anyopaque,
        n: usize,
        grain: usize,
        func: *const fn (ctx: *anyopaque, start: usize, end: usize, tid: usize) void,
        dynamic: bool,
    ) void {
        if (n == 0) return;

        if (self.thread_count_total <= 1) {
            func(ctx, 0, n, 0);
            return;
        }

        const shared_pre = self.shared orelse {
            func(ctx, 0, n, 0);
            return;
        };
        if (tls_current_pool_token == poolToken(shared_pre)) {
            runInlineSerial(ctx, n, grain, func, tls_current_tid);
            return;
        }

        std.Io.Threaded.mutexLock(&self.submit_mutex);
        defer std.Io.Threaded.mutexUnlock(&self.submit_mutex);

        const shared = self.shared orelse {
            func(ctx, 0, n, 0);
            return;
        };

        // Only use as many threads as there are work items.
        // This avoids waking many workers that would get empty ranges.
        const threads_job_total: usize = @min(self.thread_count_total, n);
        const workers_job: usize = if (threads_job_total > 0) (threads_job_total - 1) else 0;

        // Prepare job
        shared.ctx = ctx;
        shared.n = n;
        shared.grain = if (grain == 0) n else grain;
        shared.func = func;

        shared.job_thread_count_total = threads_job_total;
        shared.dynamic = dynamic;
        if (dynamic) shared.next_index.store(0, .release);

        // Publish job to selected workers by incrementing their per-worker sequence.
        var w: usize = 0;
        while (w < workers_job) : (w += 1) publishTo(shared, w);

        // Run main thread work (tid=0)
        {
            const prev_tls = pushTlsExecution(shared, 0);
            defer popTlsExecution(prev_tls);
            if (shared.dynamic) {
                runClaiming(ctx, n, shared.grain, func, 0, &shared.next_index);
            } else {
                runForTid(ctx, n, shared.grain, func, 0, threads_job_total);
            }
        }

        // Wait for workers to finish: each publishes the sequence it completed.
        var w2: usize = 0;
        while (w2 < workers_job) : (w2 += 1) {
            const slot = &shared.worker_slots[w2];
            const want = slot.job_seq.load(.acquire);
            var spin: Spin = .{ .budget_ns = SUBMIT_SPIN_NS };
            while (slot.done_seq.load(.acquire) != want) {
                if (spin.again()) continue;
                slot.submit_sleeping.store(true, .release);
                const seen = slot.done_seq.load(.acquire);
                if (seen != want) futexIo().futexWaitUncancelable(u32, &slot.done_seq.raw, seen);
                slot.submit_sleeping.store(false, .release);
            }
        }
    }

    fn workerMain(args: WorkerArgs) void {
        const shared = args.shared;
        const tid = args.tid;

        const idx: usize = tid - 1;
        const seq_ptr = workerJobSeq(shared, idx);
        // The initial sequence is established before spawning workers. Reading
        // a live sequence after publishing readiness can miss the first job.
        // Using that initial value also handles shutdown during partial startup.
        var last_seq: u32 = 0;

        // Signal that this worker is ready.
        _ = shared.ready_workers.fetchAdd(1, .release);
        futexIo().futexWake(u32, &shared.ready_workers.raw, 1);

        while (true) {
            // Wait for new job (per-worker futex word).
            var spin: Spin = .{ .budget_ns = WORKER_SPIN_NS };
            var seq = seq_ptr.load(.acquire);

            while (seq == last_seq) {
                if (spin.again()) {
                    seq = seq_ptr.load(.acquire);
                    continue;
                }
                shared.worker_slots[idx].sleeping.store(true, .seq_cst);
                futexIo().futexWaitUncancelable(u32, &seq_ptr.raw, last_seq);
                shared.worker_slots[idx].sleeping.store(false, .seq_cst);
                seq = seq_ptr.load(.acquire);
            }

            // Check stop signal
            if (shared.stop.load(.acquire)) return;

            last_seq = seq;

            // Execute
            const tc: usize = shared.job_thread_count_total;
            {
                const prev_tls = pushTlsExecution(shared, tid);
                defer popTlsExecution(prev_tls);
                if (shared.dynamic) {
                    runClaiming(shared.ctx, shared.n, shared.grain, shared.func, tid, &shared.next_index);
                } else {
                    runForTid(shared.ctx, shared.n, shared.grain, shared.func, tid, tc);
                }
            }

            // Notify completion on this worker's own line.
            const slot = &shared.worker_slots[idx];
            slot.done_seq.store(last_seq, .release);
            if (slot.submit_sleeping.load(.acquire)) {
                futexIo().futexWake(u32, &slot.done_seq.raw, 1);
            }
        }
    }

    const TlsExecutionState = struct {
        pool_token: usize,
        tid: usize,
    };

    fn pushTlsExecution(shared: *Shared, tid: usize) TlsExecutionState {
        const prev: TlsExecutionState = .{
            .pool_token = tls_current_pool_token,
            .tid = tls_current_tid,
        };
        tls_current_pool_token = poolToken(shared);
        tls_current_tid = tid;
        return prev;
    }

    fn popTlsExecution(prev: TlsExecutionState) void {
        tls_current_pool_token = prev.pool_token;
        tls_current_tid = prev.tid;
    }

    fn poolToken(shared: *Shared) usize {
        return @intFromPtr(shared);
    }

    fn workerJobSeq(shared: *Shared, idx: usize) *std.atomic.Value(u32) {
        return &shared.worker_slots[idx].job_seq;
    }

    /// Publish a new sequence to one worker, waking it only if it went to sleep.
    ///
    /// The store and the `sleeping` read are `seq_cst` because they are different
    /// atomics and a missed wake is a hang, not a slowdown: release/acquire alone
    /// would let the read float above the store and observe a worker as awake that
    /// is already parked. A worker that parks without seeing the store is still
    /// covered — `futexWait` compares the sequence itself and returns at once.
    fn publishTo(shared: *Shared, idx: usize) void {
        const slot = &shared.worker_slots[idx];
        const seq = slot.job_seq.load(.acquire);
        slot.job_seq.store(seq + 1, .seq_cst);
        if (slot.sleeping.load(.seq_cst)) {
            futexIo().futexWake(u32, &slot.job_seq.raw, 1);
        }
    }

    fn runInlineSerial(
        ctx: *anyopaque,
        n: usize,
        grain: usize,
        func: *const fn (ctx: *anyopaque, start: usize, end: usize, tid: usize) void,
        tid: usize,
    ) void {
        if (n == 0) return;

        var start: usize = 0;
        const g: usize = if (grain == 0) n else grain;
        while (start < n) {
            const end: usize = @min(n, start + g);
            func(ctx, start, end, tid);
            start = end;
        }
    }

    /// Claim `grain`-sized ranges from a shared counter until the job is done.
    ///
    /// An equal share per thread gates every step on the slowest core, which on a
    /// machine with both performance and efficiency cores is the one that was given
    /// as much work as a core twice its speed. Claiming on demand lets the fast
    /// cores take more. Each index is still executed exactly once, so results do
    /// not depend on who ran what — only `parallelForAny` promises a fixed split.
    fn runClaiming(
        ctx: *anyopaque,
        n: usize,
        grain: usize,
        func: *const fn (ctx: *anyopaque, start: usize, end: usize, tid: usize) void,
        tid: usize,
        next: *std.atomic.Value(usize),
    ) void {
        const g: usize = @max(@as(usize, 1), grain);
        while (true) {
            const start = next.fetchAdd(g, .acq_rel);
            if (start >= n) return;
            func(ctx, start, @min(n, start + g), tid);
        }
    }

    fn runForTid(
        ctx: *anyopaque,
        n: usize,
        grain: usize,
        func: *const fn (ctx: *anyopaque, start: usize, end: usize, tid: usize) void,
        tid: usize,
        thread_count_total: usize,
    ) void {
        const start0: usize = (n * tid) / thread_count_total;
        const end0: usize = (n * (tid + 1)) / thread_count_total;
        if (start0 >= end0) return;

        var start: usize = start0;
        const g: usize = if (grain == 0) (end0 - start0) else grain;
        while (start < end0) {
            const end: usize = @min(end0, start + g);
            func(ctx, start, end, tid);
            start = end;
        }
    }
};

fn expectedTidForIndex(index: usize, n: usize, thread_count_total: usize) usize {
    var tid: usize = 0;
    while (tid < thread_count_total) : (tid += 1) {
        const start: usize = (n * tid) / thread_count_total;
        const end: usize = (n * (tid + 1)) / thread_count_total;
        if (index >= start and index < end) return tid;
    }
    unreachable;
}

fn waitForBool(flag: *const std.atomic.Value(bool), expected: bool) !void {
    var tries: usize = 0;
    while (flag.load(.acquire) != expected) : (tries += 1) {
        if (tries >= 50_000) return error.Timeout;
        if ((tries & 255) == 0) {
            std.Thread.yield() catch {};
        } else {
            std.atomic.spinLoopHint();
        }
    }
}

fn yieldMany(count: usize) void {
    var i: usize = 0;
    while (i < count) : (i += 1) {
        std.Thread.yield() catch {};
    }
}

fn shouldSkipThreadPoolTests() bool {
    // This test suite can occasionally stall on some Windows environments.
    // Allow users/CI to skip it deterministically.
    const builtin = @import("builtin");
    if (builtin.os.tag != .windows) return false;

    return env_util.flagEnabled("AION_SKIP_THREAD_POOL_TESTS");
}

fn skipIfRequested() !void {
    if (shouldSkipThreadPoolTests()) return error.SkipZigTest;
}

test "thread pool: deterministic partitioning across shapes" {
    try skipIfRequested();
    const allocator: std.mem.Allocator = std.testing.allocator;

    var pool = try ThreadPool.init(allocator, .{ .thread_count = 4 });
    defer pool.deinit();

    var values: [2048]u32 = @splat(0);
    var tids: [2048]usize = @splat(std.math.maxInt(usize));

    const Ctx = struct {
        values: []u32,
        tids: []usize,
        seed: u32,
    };

    const fn_ptr = struct {
        fn run(ctx_any: *anyopaque, start: usize, end: usize, tid: usize) void {
            const c: *Ctx = @ptrCast(@alignCast(ctx_any));
            var i: usize = start;
            while (i < end) : (i += 1) {
                c.values[i] = c.seed + @as(u32, @intCast(i * 3 + 7));
                c.tids[i] = tid;
            }
        }
    }.run;

    const Case = struct {
        n: usize,
        grain: usize,
    };

    const cases = [_]Case{
        .{ .n = 0, .grain = 0 },
        .{ .n = 1, .grain = 0 },
        .{ .n = 3, .grain = 1 },
        .{ .n = 4, .grain = 0 },
        .{ .n = 7, .grain = 3 },
        .{ .n = 64, .grain = 16 },
        .{ .n = 257, .grain = 1 },
        .{ .n = 2048, .grain = 0 },
    };

    var iter: usize = 0;
    while (iter < 128) : (iter += 1) {
        for (cases) |case| {
            @memset(values[0..case.n], 0);
            @memset(tids[0..case.n], std.math.maxInt(usize));

            var ctx: Ctx = .{
                .values = values[0..case.n],
                .tids = tids[0..case.n],
                .seed = @intCast(iter * 17),
            };

            pool.parallelForAny(@ptrCast(&ctx), case.n, case.grain, fn_ptr);

            const expected_threads: usize = if (case.n == 0) 0 else @min(pool.threadCount(), case.n);
            var i: usize = 0;
            while (i < case.n) : (i += 1) {
                try std.testing.expectEqual(ctx.seed + @as(u32, @intCast(i * 3 + 7)), ctx.values[i]);
                try std.testing.expectEqual(expectedTidForIndex(i, case.n, expected_threads), ctx.tids[i]);
            }
        }
    }
}

test "thread pool: dynamic claiming runs every index exactly once" {
    try skipIfRequested();
    const allocator: std.mem.Allocator = std.testing.allocator;

    var pool = try ThreadPool.init(allocator, .{ .thread_count = 4 });
    defer pool.deinit();

    var hits: [1031]std.atomic.Value(u32) = undefined;

    const Ctx = struct {
        hits: []std.atomic.Value(u32),
        /// One thread dawdles so the others must claim what a fixed split gave it.
        slow_tid: usize,
    };
    const run = struct {
        fn f(ctx_any: *anyopaque, start: usize, end: usize, tid: usize) void {
            const c: *Ctx = @ptrCast(@alignCast(ctx_any));
            if (tid == c.slow_tid) yieldMany(8);
            for (start..end) |i| _ = c.hits[i].fetchAdd(1, .monotonic);
        }
    }.f;

    const shapes = [_][2]usize{ .{ 1, 1 }, .{ 3, 1 }, .{ 4, 0 }, .{ 17, 5 }, .{ 256, 1 }, .{ 1031, 7 }, .{ 1031, 2000 } };
    for (0..64) |iter| {
        for (shapes) |shape| {
            const n = shape[0];
            for (hits[0..n]) |*h| h.* = .init(0);
            var ctx: Ctx = .{ .hits = hits[0..n], .slow_tid = iter % 4 };
            pool.parallelForDynamic(@ptrCast(&ctx), n, shape[1], run);
            for (hits[0..n]) |*h| try std.testing.expectEqual(@as(u32, 1), h.load(.monotonic));

            // A static job in between must not inherit the claim counter.
            for (hits[0..n]) |*h| h.* = .init(0);
            pool.parallelForAny(@ptrCast(&ctx), n, shape[1], run);
            for (hits[0..n]) |*h| try std.testing.expectEqual(@as(u32, 1), h.load(.monotonic));
        }
        // Past the spin budgets now and then, so jobs also land on parked workers.
        if (iter % 16 == 0) std.Io.sleep(futexIo(), .fromMilliseconds(2), .awake) catch {};
    }
}

test "thread pool: repeated init and deinit remains stable" {
    try skipIfRequested();
    const allocator: std.mem.Allocator = std.testing.allocator;

    const Ctx = struct {
        buf: []u32,
    };

    const fn_ptr = struct {
        fn run(ctx_any: *anyopaque, start: usize, end: usize, tid: usize) void {
            const c: *Ctx = @ptrCast(@alignCast(ctx_any));
            var i: usize = start;
            while (i < end) : (i += 1) {
                c.buf[i] = @as(u32, @intCast(tid + i));
            }
        }
    }.run;

    var round: usize = 0;
    while (round < 64) : (round += 1) {
        {
            var pool = try ThreadPool.init(allocator, .{ .thread_count = 4 });
            defer pool.deinit();

            var buf: [64]u32 = @splat(0);
            var ctx: Ctx = .{ .buf = buf[0..] };
            pool.parallelForAny(@ptrCast(&ctx), ctx.buf.len, 4, fn_ptr);

            var i: usize = 0;
            while (i < ctx.buf.len) : (i += 1) {
                try std.testing.expectEqual(@as(u32, @intCast(expectedTidForIndex(i, ctx.buf.len, pool.threadCount()) + i)), ctx.buf[i]);
            }
        }
    }
}

test "thread pool: concurrent submissions serialize" {
    try skipIfRequested();
    const allocator: std.mem.Allocator = std.testing.allocator;

    var pool = try ThreadPool.init(allocator, .{ .thread_count = 4 });
    defer pool.deinit();

    const JobCtx = struct {
        buf: []u32,
        value: u32,
        entered: *std.atomic.Value(bool),
        gate: *const std.atomic.Value(bool),
        callback_count: *std.atomic.Value(u32),
    };

    const job_fn = struct {
        fn run(ctx_any: *anyopaque, start: usize, end: usize, tid: usize) void {
            _ = tid;
            const c: *JobCtx = @ptrCast(@alignCast(ctx_any));
            c.entered.store(true, .release);
            _ = c.callback_count.fetchAdd(1, .acq_rel);

            while (!c.gate.load(.acquire)) {
                std.atomic.spinLoopHint();
            }

            var i: usize = start;
            while (i < end) : (i += 1) {
                c.buf[i] = c.value;
            }
        }
    }.run;

    const SubmitArgs = struct {
        pool: *ThreadPool,
        ctx: *JobCtx,
        submit_entered: ?*std.atomic.Value(bool) = null,
    };

    const submit_job = struct {
        fn run(args: SubmitArgs) void {
            if (args.submit_entered) |flag| flag.store(true, .release);
            args.pool.parallelForAny(@ptrCast(args.ctx), args.ctx.buf.len, 4, job_fn);
        }
    }.run;

    var gate_a = std.atomic.Value(bool).init(false);
    const gate_b = std.atomic.Value(bool).init(true);
    var entered_a = std.atomic.Value(bool).init(false);
    var entered_b = std.atomic.Value(bool).init(false);
    var submit_entered_b = std.atomic.Value(bool).init(false);
    var callback_count_a = std.atomic.Value(u32).init(0);
    var callback_count_b = std.atomic.Value(u32).init(0);
    var buf_a: [64]u32 = @splat(0);
    var buf_b: [64]u32 = @splat(0);

    var ctx_a: JobCtx = .{
        .buf = buf_a[0..],
        .value = 11,
        .entered = &entered_a,
        .gate = &gate_a,
        .callback_count = &callback_count_a,
    };
    var ctx_b: JobCtx = .{
        .buf = buf_b[0..],
        .value = 29,
        .entered = &entered_b,
        .gate = &gate_b,
        .callback_count = &callback_count_b,
    };

    const submitter_a = try std.Thread.spawn(.{}, submit_job, .{SubmitArgs{
        .pool = &pool,
        .ctx = &ctx_a,
    }});
    try waitForBool(&entered_a, true);

    const submitter_b = try std.Thread.spawn(.{}, submit_job, .{SubmitArgs{
        .pool = &pool,
        .ctx = &ctx_b,
        .submit_entered = &submit_entered_b,
    }});
    try waitForBool(&submit_entered_b, true);

    yieldMany(2048);
    try std.testing.expectEqual(@as(u32, 0), callback_count_b.load(.acquire));
    try std.testing.expect(!entered_b.load(.acquire));

    gate_a.store(true, .release);
    submitter_a.join();
    submitter_b.join();

    for (buf_a) |value| try std.testing.expectEqual(@as(u32, 11), value);
    for (buf_b) |value| try std.testing.expectEqual(@as(u32, 29), value);
}

test "thread pool: nested same-pool submission runs inline on caller tid" {
    try skipIfRequested();
    const allocator: std.mem.Allocator = std.testing.allocator;

    var pool = try ThreadPool.init(allocator, .{ .thread_count = 4 });
    defer pool.deinit();

    const outer_n = 8;
    const inner_n = 7;

    const InnerCtx = struct {
        row: []usize,
    };

    const OuterCtx = struct {
        pool: *ThreadPool,
        outer_tids: []usize,
        inner_tids: []usize,
    };

    const inner_fn = struct {
        fn run(ctx_any: *anyopaque, start: usize, end: usize, tid: usize) void {
            const c: *InnerCtx = @ptrCast(@alignCast(ctx_any));
            var i: usize = start;
            while (i < end) : (i += 1) {
                c.row[i] = tid;
            }
        }
    }.run;

    const outer_fn = struct {
        fn run(ctx_any: *anyopaque, start: usize, end: usize, tid: usize) void {
            const c: *OuterCtx = @ptrCast(@alignCast(ctx_any));
            var outer_idx: usize = start;
            while (outer_idx < end) : (outer_idx += 1) {
                c.outer_tids[outer_idx] = tid;
                var inner_ctx: InnerCtx = .{
                    .row = c.inner_tids[(outer_idx * inner_n)..][0..inner_n],
                };
                c.pool.parallelForAny(@ptrCast(&inner_ctx), inner_n, 1, inner_fn);
            }
        }
    }.run;

    var outer_tids: [outer_n]usize = @splat(std.math.maxInt(usize));
    var inner_tids: [outer_n * inner_n]usize = @splat(std.math.maxInt(usize));
    var outer_ctx: OuterCtx = .{
        .pool = &pool,
        .outer_tids = outer_tids[0..],
        .inner_tids = inner_tids[0..],
    };

    pool.parallelForAny(@ptrCast(&outer_ctx), outer_n, 1, outer_fn);

    var outer_idx: usize = 0;
    while (outer_idx < outer_n) : (outer_idx += 1) {
        const expected_outer_tid = expectedTidForIndex(outer_idx, outer_n, @min(pool.threadCount(), outer_n));
        try std.testing.expectEqual(expected_outer_tid, outer_tids[outer_idx]);

        var inner_idx: usize = 0;
        while (inner_idx < inner_n) : (inner_idx += 1) {
            try std.testing.expectEqual(
                outer_tids[outer_idx],
                inner_tids[(outer_idx * inner_n) + inner_idx],
            );
        }
    }
}

test "thread pool: deinit waits for an active submission" {
    try skipIfRequested();
    const allocator: std.mem.Allocator = std.testing.allocator;

    var pool = try ThreadPool.init(allocator, .{ .thread_count = 4 });

    const JobCtx = struct {
        started: *std.atomic.Value(bool),
        gate: *const std.atomic.Value(bool),
    };

    const job_fn = struct {
        fn run(ctx_any: *anyopaque, start: usize, end: usize, tid: usize) void {
            _ = start;
            _ = end;
            _ = tid;
            const c: *JobCtx = @ptrCast(@alignCast(ctx_any));
            c.started.store(true, .release);
            while (!c.gate.load(.acquire)) {
                std.atomic.spinLoopHint();
            }
        }
    }.run;

    const SubmitArgs = struct {
        pool: *ThreadPool,
        ctx: *JobCtx,
    };

    const submit_job = struct {
        fn run(args: SubmitArgs) void {
            args.pool.parallelForAny(@ptrCast(args.ctx), 32, 1, job_fn);
        }
    }.run;

    const DeinitArgs = struct {
        pool: *ThreadPool,
        started: *std.atomic.Value(bool),
        done: *std.atomic.Value(bool),
    };

    const deinit_pool = struct {
        fn run(args: DeinitArgs) void {
            args.started.store(true, .release);
            args.pool.deinit();
            args.done.store(true, .release);
        }
    }.run;

    var gate = std.atomic.Value(bool).init(false);
    var started = std.atomic.Value(bool).init(false);
    var deinit_started = std.atomic.Value(bool).init(false);
    var deinit_done = std.atomic.Value(bool).init(false);
    var ctx: JobCtx = .{
        .started = &started,
        .gate = &gate,
    };

    const submitter = try std.Thread.spawn(.{}, submit_job, .{SubmitArgs{
        .pool = &pool,
        .ctx = &ctx,
    }});
    try waitForBool(&started, true);

    const deinit_thread = try std.Thread.spawn(.{}, deinit_pool, .{DeinitArgs{
        .pool = &pool,
        .started = &deinit_started,
        .done = &deinit_done,
    }});
    try waitForBool(&deinit_started, true);

    yieldMany(2048);
    try std.testing.expect(!deinit_done.load(.acquire));

    gate.store(true, .release);

    submitter.join();
    deinit_thread.join();
    try std.testing.expect(deinit_done.load(.acquire));
}

test "fallible parallel jobs join and return worker failures" {
    var pool = try ThreadPool.init(std.testing.allocator, .{ .thread_count = 3 });
    defer pool.deinit();
    const E = error{InjectedFailure};
    const Task = struct {
        completed: std.atomic.Value(usize) = .init(0),
        fn run(raw: *anyopaque, start: usize, end: usize, _: usize) E!void {
            const self: *@This() = @ptrCast(@alignCast(raw));
            _ = self.completed.fetchAdd(end - start, .monotonic);
            if (start <= 5 and end > 5) return error.InjectedFailure;
        }
    };
    var task: Task = .{};
    try std.testing.expectError(error.InjectedFailure, pool.parallelForFallible(E, &task, 12, 1, Task.run));
    try std.testing.expectEqual(@as(usize, 12), task.completed.load(.acquire));
    // The previous failure must not leak into the next invocation.
    try pool.parallelForFallible(E, &task, 2, 1, Task.run);
}
