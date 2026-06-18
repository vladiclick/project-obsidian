// attention.rs — GQA SDPA decode. q/k norm + RoPE are applied in forward.rs
// (before append), so this is pure attention over the KV cache.

use crate::executor::ObsidianExecutor;
use crate::kv_cache::KvCache;
use metal::{Buffer, MTLResourceOptions, MTLSize};

#[repr(C)]
#[derive(Clone, Copy)]
struct SdpaParams { n_heads: u32, n_kv: u32, head_dim: u32, seq_len: u32, kv_stride: u32, scale: f32 }

pub struct Attention {
    pub n_heads: u32,
    pub n_kv: u32,
    pub head_dim: u32,
    pub theta: f32,
    /// device score scratch [n_heads * max_len] f32, allocated once + reused.
    pub scratch: Buffer,
}

impl Attention {
    /// SDPA over the cache -> [n_heads*head_dim] fp16. q already normed+roped;
    /// the cache already holds normed+roped k and raw v.
    pub fn sdpa(&self, exec: &ObsidianExecutor, q: &Buffer, kv: &KvCache) -> Result<Buffer, String> {
        let out = exec.device().new_buffer((self.n_heads * self.head_dim * 2) as u64,
                                           MTLResourceOptions::StorageModeShared);
        let sp = SdpaParams {
            n_heads: self.n_heads, n_kv: self.n_kv, head_dim: self.head_dim,
            seq_len: kv.seq_len.max(1) as u32, kv_stride: kv.max_len as u32,
            scale: (self.head_dim as f32).powf(-0.5),
        };
        let pso = exec.pipeline("sdpa_decode").ok_or("sdpa_decode missing")?;
        let cb = exec.queue().new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pso);
        for (i, b) in [q, &kv.k, &kv.v, &out, &self.scratch].iter().enumerate() { enc.set_buffer(i as u64, Some(*b), 0); }
        enc.set_bytes(5, std::mem::size_of::<SdpaParams>() as u64, &sp as *const _ as *const _);
        enc.dispatch_thread_groups(MTLSize::new(self.n_heads as u64, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
        cb.commit();   // no wait: same-queue ordering syncs with later dispatches
        Ok(out)
    }
}
