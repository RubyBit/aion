// Keys a query may attend to, as `[lo, hi)`. Mirrors `graph.AttentionWindow.keys`;
// passed as three u32 rather than a struct to avoid uniform-struct alignment rules.
fn window_keys(left: u32, right: u32, chunk: u32, query_pos: u32, kv_end: u32) -> vec2<u32> {
    var anchor = query_pos;
    var span = 1u;
    if (chunk > 0u) {
        anchor = (query_pos / chunk) * chunk;
        span = chunk;
    }
    var hi = kv_end;
    if (right < kv_end) { hi = min(kv_end, anchor + span + right); }
    return vec2<u32>(anchor - min(anchor, left), hi);
}
