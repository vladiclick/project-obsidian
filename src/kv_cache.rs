// kv_cache.rs — per-layer rotating K/V buffers (fp16, StorageModeShared).
// Layout: [n_kv, max_len, head_dim]. forward.rs writes each token at seq_len
// on the GPU (kv_append_gpu).

use metal::{Buffer, Device, MTLResourceOptions};

pub struct KvCache {
    pub k: Buffer,
    pub v: Buffer,
    pub seq_len: usize,
    pub max_len: usize,
}

impl KvCache {
    pub fn new(dev: &Device, n_kv: usize, head_dim: usize, max_len: usize) -> Self {
        let bytes = (n_kv * max_len * head_dim * 2) as u64; // fp16
        let opt = MTLResourceOptions::StorageModeShared;
        KvCache {
            k: dev.new_buffer(bytes, opt),
            v: dev.new_buffer(bytes, opt),
            seq_len: 0,
            max_len,
        }
    }
}
