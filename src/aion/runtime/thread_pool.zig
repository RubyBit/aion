const std = @import("std");

/// A persistent, deterministic thread pool designed for numerical kernels.
///
/// Design goals:
/// - No allocations during `parallelForAny`.
/// - Deterministic work partitioning (static contiguous ranges per tid).
/// - Low overhead: Uses Atomic + Futex for synchronization.
///
/// Terminology:
/// - `thread_count_total` includes the calling thread (tid=0) plus worker threads (tid=1..).
pub const ThreadPool = struct {
    allocator: std.mem.Allocator,

    worker_threads: []std.Thread,
    thread_count_total: usize,

    shared: ?*Shared = null,

    pub const Options = struct {
        /// Total threads to use including the calling thread.
        /// Set to 1 to disable parallelism.
        thread_count: usize = 1,
    };

    const Shared = struct {
        // Synchronization primitives
        job_seq: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
        pending_workers: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
        ready_workers: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
        stop: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),

        // Job payload
        ctx: *anyopaque = undefined,
        n: usize = 0,
        grain: usize = 0,
        func: *const fn (ctx: *anyopaque, start: usize, end: usize, tid: usize) void = undefined,

        thread_count_total: usize = 1,
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
        shared.thread_count_total = pool.thread_count_total;
        pool.shared = shared;

        const worker_count: usize = opts.thread_count - 1;
        pool.worker_threads = try allocator.alloc(std.Thread, worker_count);
        errdefer allocator.free(pool.worker_threads);

        var i: usize = 0;
        while (i < worker_count) : (i += 1) {
            const args: WorkerArgs = .{ .shared = shared, .tid = i + 1 };
            pool.worker_threads[i] = try std.Thread.spawn(.{}, workerMain, .{args});
        }

        // Barrier: ensure all workers have entered workerMain and are ready to observe the
        // first job. Without this, a job launched immediately after init could be missed by
        // workers that haven't started waiting yet, causing a hang.
        var ready: u32 = shared.ready_workers.load(.acquire);
        while (ready < @as(u32, @intCast(worker_count))) {
            std.Thread.Futex.wait(&shared.ready_workers, ready);
            ready = shared.ready_workers.load(.acquire);
        }

        return pool;
    }

    pub fn deinit(self: *ThreadPool) void {
        const shared = self.shared orelse {
            if (self.worker_threads.len > 0) self.allocator.free(self.worker_threads);
            self.thread_count_total = 1;
            return;
        };

        // Signal stop
        shared.stop.store(true, .seq_cst);
        const seq = shared.job_seq.load(.monotonic);
        shared.job_seq.store(seq + 1, .release);
        std.Thread.Futex.wake(&shared.job_seq, @intCast(self.worker_threads.len));

        for (self.worker_threads) |t| {
            t.join();
        }

        if (self.worker_threads.len > 0) {
            self.allocator.free(self.worker_threads);
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
    /// sequential chunks (still deterministic, no stealing).
    pub fn parallelForAny(
        self: *ThreadPool,
        ctx: *anyopaque,
        n: usize,
        grain: usize,
        func: *const fn (ctx: *anyopaque, start: usize, end: usize, tid: usize) void,
    ) void {
        if (n == 0) return;

        if (self.thread_count_total <= 1) {
            func(ctx, 0, n, 0);
            return;
        }

        const shared = self.shared orelse {
            func(ctx, 0, n, 0);
            return;
        };

        const workers_total = self.thread_count_total - 1;

        // Prepare job
        shared.ctx = ctx;
        shared.n = n;
        shared.grain = if (grain == 0) n else grain;
        shared.func = func;

        // Reset pending count to all workers
        shared.pending_workers.store(@intCast(workers_total), .monotonic);

        // Publish job by incrementing sequence
        const seq = shared.job_seq.load(.monotonic);
        shared.job_seq.store(seq + 1, .release);

        // Wake workers
        std.Thread.Futex.wake(&shared.job_seq, @intCast(workers_total));

        // Run main thread work (tid=0)
        runForTid(ctx, n, shared.grain, func, 0, self.thread_count_total);

        // Wait for workers to finish
        var spin_wait: usize = 0;
        while (true) {
            const pending = shared.pending_workers.load(.acquire);
            if (pending == 0) break;

            if (spin_wait < 1000) {
                std.atomic.spinLoopHint();
                spin_wait += 1;
                continue;
            }

            std.Thread.Futex.wait(&shared.pending_workers, pending);
        }
    }

    fn workerMain(args: WorkerArgs) void {
        const shared = args.shared;
        const tid = args.tid;

        // Signal that this worker is ready.
        _ = shared.ready_workers.fetchAdd(1, .release);
        std.Thread.Futex.wake(&shared.ready_workers, 1);

        var last_seq = shared.job_seq.load(.monotonic);

        while (true) {
            // Wait for new job
            var spin_wait: usize = 0;
            var seq = shared.job_seq.load(.acquire);

            while (seq == last_seq) {
                if (spin_wait < 5000) {
                    std.atomic.spinLoopHint();
                    spin_wait += 1;
                    seq = shared.job_seq.load(.monotonic);
                    continue;
                }
                std.Thread.Futex.wait(&shared.job_seq, last_seq);
                seq = shared.job_seq.load(.monotonic);
            }

            // Check stop signal
            if (shared.stop.load(.monotonic)) return;

            // Ensure acquire fence for payload if needed, job_seq load(.acquire) above covers it
            _ = shared.job_seq.load(.acquire);

            last_seq = seq;

            // Execute
            runForTid(shared.ctx, shared.n, shared.grain, shared.func, tid, shared.thread_count_total);

            // Notify completion
            const prev = shared.pending_workers.fetchSub(1, .release);
            _ = prev;
            // Wake waiters unconditionally.
            // Futex waiters are not guaranteed to wake on value change unless a wake is issued.
            std.Thread.Futex.wake(&shared.pending_workers, 1);
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

test "thread pool: parallelForAny matches single-thread" {
    const allocator: std.mem.Allocator = std.testing.allocator;

    var pool = try ThreadPool.init(allocator, .{ .thread_count = 4 });
    defer pool.deinit();

    var data: [10000]u32 = .{0} ** 10000;

    const Ctx = struct {
        buf: []u32,
    };

    var ctx: Ctx = .{ .buf = data[0..] };

    const fn_ptr = struct {
        fn run(ctx_any: *anyopaque, start: usize, end: usize, tid: usize) void {
            _ = tid;
            const c: *Ctx = @ptrCast(@alignCast(ctx_any));
            var i: usize = start;
            while (i < end) : (i += 1) {
                c.buf[i] = @intCast(i * 3 + 7);
            }
        }
    }.run;

    pool.parallelForAny(@ptrCast(&ctx), ctx.buf.len, 1024, fn_ptr);

    var i: usize = 0;
    while (i < ctx.buf.len) : (i += 1) {
        try std.testing.expectEqual(@as(u32, @intCast(i * 3 + 7)), ctx.buf[i]);
    }
}
