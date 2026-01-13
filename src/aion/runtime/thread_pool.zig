const std = @import("std");

/// A persistent, deterministic thread pool designed for numerical kernels.
///
/// Design goals:
/// - No allocations during `parallelForAny`.
/// - Deterministic work partitioning (static contiguous ranges per tid).
/// - Low overhead on Windows: threads are created once and reused.
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
        mutex: std.Thread.Mutex = .{},
        start_cond: std.Thread.Condition = .{},
        done_cond: std.Thread.Condition = .{},

        stop: bool = false,

        // Job state
        gen: u64 = 0,
        // No separate `active` flag: `gen` change indicates a new job.
        pending_workers: usize = 0,

        n: usize = 0,
        grain: usize = 0,
        ctx: *anyopaque = undefined,
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
            // tid 1..thread_count_total-1 for workers
            const args: WorkerArgs = .{ .shared = shared, .tid = i + 1 };
            pool.worker_threads[i] = try std.Thread.spawn(.{}, workerMain, .{args});
        }

        return pool;
    }

    pub fn deinit(self: *ThreadPool) void {
        if (self.thread_count_total <= 1) return;

        const shared: *Shared = self.shared orelse {
            // Should not happen, but keep deinit robust.
            self.allocator.free(self.worker_threads);
            self.worker_threads = &[_]std.Thread{};
            self.thread_count_total = 1;
            return;
        };

        shared.mutex.lock();
        shared.stop = true;
        shared.start_cond.broadcast();
        shared.mutex.unlock();

        for (self.worker_threads) |t| {
            t.join();
        }

        self.allocator.free(self.worker_threads);
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

        const shared: *Shared = self.shared orelse {
            // Should not happen if thread_count_total > 1.
            func(ctx, 0, n, 0);
            return;
        };

        // Count how many worker threads will actually have non-empty ranges.
        // This avoids waiting for workers whose static partition is empty.
        // Note: we still use a single broadcast wake; workers without work will
        // go back to sleep quickly, but completion accounting won't stall.
        var workers_with_work: usize = 0;
        {
            var tid: usize = 1;
            while (tid < self.thread_count_total) : (tid += 1) {
                const start0: usize = (n * tid) / self.thread_count_total;
                const end0: usize = (n * (tid + 1)) / self.thread_count_total;
                if (start0 < end0) workers_with_work += 1;
            }
        }

        // If no worker can contribute, avoid the synchronization entirely.
        if (workers_with_work == 0) {
            func(ctx, 0, n, 0);
            return;
        }

        shared.mutex.lock();
        // Publish job
        shared.ctx = ctx;
        shared.n = n;
        shared.grain = if (grain == 0) n else grain;
        shared.func = func;
        shared.pending_workers = workers_with_work;
        shared.gen +%= 1;
        const my_gen: u64 = shared.gen;

        // Wake workers
        shared.start_cond.broadcast();
        shared.mutex.unlock();

        // Run tid=0 on the calling thread
        _ = runForTid(ctx, n, shared.grain, func, 0, self.thread_count_total);

        // Wait for workers
        shared.mutex.lock();
        while (shared.gen == my_gen and shared.pending_workers != 0) {
            shared.done_cond.wait(&shared.mutex);
        }
        shared.mutex.unlock();
    }

    fn workerMain(args: WorkerArgs) void {
        const shared: *Shared = args.shared;
        const tid: usize = args.tid;
        var seen_gen: u64 = 0;

        while (true) {
            shared.mutex.lock();
            while (!shared.stop and shared.gen == seen_gen) {
                shared.start_cond.wait(&shared.mutex);
            }

            if (shared.stop) {
                shared.mutex.unlock();
                return;
            }

            // Snapshot job
            const ctx: *anyopaque = shared.ctx;
            const n: usize = shared.n;
            const grain: usize = shared.grain;
            const func = shared.func;
            const job_gen: u64 = shared.gen;
            const tcount: usize = shared.thread_count_total;
            seen_gen = job_gen;
            shared.mutex.unlock();

            const did_work: bool = runForTid(ctx, n, grain, func, tid, tcount);

            shared.mutex.lock();
            if (did_work and shared.gen == job_gen and shared.pending_workers > 0) {
                shared.pending_workers -= 1;
                if (shared.pending_workers == 0) shared.done_cond.signal();
            }
            shared.mutex.unlock();
        }
    }

    fn runForTid(
        ctx: *anyopaque,
        n: usize,
        grain: usize,
        func: *const fn (ctx: *anyopaque, start: usize, end: usize, tid: usize) void,
        tid: usize,
        thread_count_total: usize,
    ) bool {
        // Static contiguous partition:
        // start = floor(n*tid/tcount), end = floor(n*(tid+1)/tcount)
        const start0: usize = (n * tid) / thread_count_total;
        const end0: usize = (n * (tid + 1)) / thread_count_total;
        if (start0 >= end0) return false;

        var start: usize = start0;
        const g: usize = if (grain == 0) (end0 - start0) else grain;
        while (start < end0) {
            const end: usize = @min(end0, start + g);
            func(ctx, start, end, tid);
            start = end;
        }

        return true;
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
