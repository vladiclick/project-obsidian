// kv_cache.rs — per-layer rotating K/V buffers (fp16, StorageModeShared).
// Layout: [n_kv, max_len, head_dim]. Decode appends one token at seq_len.

use metal::{Buffer, Device, MTLResourceOptions};

pub struct KvCache {
    pub k: Buffer,
    pub v: Buffer,
    pub seq_len: usize,
    pub max_len: usize,
    pub n_kv: usize,
    pub head_dim: usize,
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
            n_kv,
            head_dim,
        }
    }

    /// Append one token's K/V (each [n_kv, head_dim], fp16) at the current
    /// position. StorageModeShared -> direct CPU memcpy into the cache.
    pub fn append(&mut self, k_tok: &[u16], v_tok: &[u16]) {
        let pos = self.seq_len.min(self.max_len - 1);
        let kp = self.k.contents() as *mut u16;
        let vp = self.v.contents() as *mut u16;
        for kv in 0..self.n_kv {
            let dst = kv * self.max_len * self.head_dim + pos * self.head_dim;
            let src = kv * self.head_dim;
            unsafe {
                std::ptr::copy_nonoverlapping(k_tok.as_ptr().add(src), kp.add(dst), self.head_dim);
                std::ptr::copy_nonoverlapping(v_tok.as_ptr().add(src), vp.add(dst), self.head_dim);
            }
        }
        self.seq_len += 1;
    }
}
