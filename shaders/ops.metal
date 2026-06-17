// ops.metal — small GPU-resident elementwise kernels so residual/swiglu/
// accumulate/kv-append never round-trip through the CPU.
#include <metal_stdlib>
using namespace metal;

struct N { uint n; };
struct ScaledN { uint n; float scale; };
struct KvP { uint n_kv; uint head_dim; uint max_len; uint pos; };

// out = a + b
kernel void ew_add(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]],
                   device half* out [[buffer(2)]], constant N& p [[buffer(3)]],
                   uint t [[thread_position_in_grid]]) {
    if (t < p.n) out[t] = (half)((float)a[t] + (float)b[t]);
}

// out = silu(g) * u
kernel void ew_swiglu(device const half* g [[buffer(0)]], device const half* u [[buffer(1)]],
                      device half* out [[buffer(2)]], constant N& p [[buffer(3)]],
                      uint t [[thread_position_in_grid]]) {
    if (t < p.n) { float x = (float)g[t]; out[t] = (half)((x / (1.0f + exp(-x))) * (float)u[t]); }
}

// acc += scale * y   (in place)
kernel void ew_scaled_add(device half* acc [[buffer(0)]], device const half* y [[buffer(1)]],
                          constant ScaledN& p [[buffer(2)]],
                          uint t [[thread_position_in_grid]]) {
    if (t < p.n) acc[t] = (half)((float)acc[t] + p.scale * (float)y[t]);
}

// copy a token's K/V [n_kv, head_dim] into the cache [n_kv, max_len, head_dim] at pos
kernel void kv_append(device const half* src [[buffer(0)]], device half* dst [[buffer(1)]],
                      constant KvP& p [[buffer(2)]], uint t [[thread_position_in_grid]]) {
    if (t >= p.n_kv * p.head_dim) return;
    uint kv = t / p.head_dim, d = t % p.head_dim;
    dst[kv * p.max_len * p.head_dim + p.pos * p.head_dim + d] = src[kv * p.head_dim + d];
}
