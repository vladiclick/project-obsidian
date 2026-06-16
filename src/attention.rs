// attention.rs — GQA attention for decode: RoPE(q) then SDPA over the KV cache.
// QKV projection (4-bit dequant-matvec) belongs to the weight loader / forward
// wiring; here q is already projected. KV-cache stride caveat noted below.

use crate::executor::ObsidianExecutor;
use crate::kv_cache::KvCache;
use metal::{Buffer, MTLResourceOptions, MTLSize};

#[repr(C)]
#[derive(Clone, Copy)]
struct RopeParams { n_heads: u32, head_dim: u32, pos: u32, theta: f32 }

#[repr(C)]
#[derive(Clone, Copy)]
struct SdpaParams { n_heads: u32, n_kv: u32, head_dim: u32, seq_len: u32, kv_stride: u32, scale: f32 }

pub struct Attention {
    pub n_heads: u32,
    pub n_kv: u32,
    pub head_dim: u32,
    pub theta: f32,
}

impl Attention {
    /// RoPE(q) in place, then single-query SDPA over the cache -> [n_heads*head_dim] fp16.
    /// NOTE: sdpa_decode treats the cache as [n_kv, seq_len, head_dim]; the
    /// KvCache buffer is allocated at max_len stride, so for seq_len < max_len
    /// the stride must match (refinement: pass max_len as a separate stride).
    pub fn forward(&self, exec: &ObsidianExecutor, q: &Buffer, kv: &KvCache) -> Result<Buffer, String> {
        let dev = exec.device();
        let pos = (kv.seq_len.max(1) - 1) as u32;

        // 1) RoPE on q (buffer 0 = q, params at slot 1)
        let rp = RopeParams { n_heads: self.n_heads, head_dim: self.head_dim, pos, theta: self.theta };
        let n_pairs = (self.n_heads * self.head_dim / 2) as u64;
        self.run(exec, "rope_apply", &[q], 1,
                 &rp as *const _ as *const std::ffi::c_void,
                 std::mem::size_of::<RopeParams>(), n_pairs, false)?;

        // 2) SDPA: one threadgroup per query head
        let out = dev.new_buffer((self.n_heads * self.head_dim * 2) as u64,
                                 MTLResourceOptions::StorageModeShared);
        let sp = SdpaParams {
            n_heads: self.n_heads, n_kv: self.n_kv, head_dim: self.head_dim,
            seq_len: kv.seq_len.max(1) as u32,
            kv_stride: kv.max_len as u32,
            scale: (self.head_dim as f32).powf(-0.5),
        };
        let pso = exec.pipeline("sdpa_decode").ok_or("sdpa_decode missing")?;
        let cb = exec.queue().new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pso);
        enc.set_buffer(0, Some(q), 0);
        enc.set_buffer(1, Some(&kv.k), 0);
        enc.set_buffer(2, Some(&kv.v), 0);
        enc.set_buffer(3, Some(&out), 0);
        enc.set_bytes(4, std::mem::size_of::<SdpaParams>() as u64,
                      &sp as *const _ as *const std::ffi::c_void);
        enc.dispatch_thread_groups(MTLSize::new(self.n_heads as u64, 1, 1),
                                   MTLSize::new(256, 1, 1));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        Ok(out)
    }

    /// Dispatch a 1-D kernel: bind `bufs` at slots 0.., params at slot
    /// `params_slot`, grid = ceil(n/256) threadgroups of 256.
    fn run(&self, exec: &ObsidianExecutor, kernel: &str, bufs: &[&Buffer],
           params_slot: u64, params: *const std::ffi::c_void, plen: usize,
           n: u64, _wait: bool) -> Result<(), String> {
        let pso = exec.pipeline(kernel).ok_or(format!("{kernel} missing"))?;
        let cb = exec.queue().new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pso);
        for (i, b) in bufs.iter().enumerate() {
            enc.set_buffer(i as u64, Some(*b), 0);
        }
        enc.set_bytes(params_slot, plen as u64, params);
        let tg = 256u64;
        enc.dispatch_thread_groups(MTLSize::new((n + tg - 1) / tg, 1, 1),
                                   MTLSize::new(tg, 1, 1));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        Ok(())
    }
}
