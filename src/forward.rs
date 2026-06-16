// forward.rs — Model + forward_layer: wires one full Qwen3-MoE decoder layer
// out of the metal kernels. SCAFFOLD: the dispatch sequence is real, but exact
// dims/dtypes still need validation against live weights (iterative).

use crate::attention::Attention;
use crate::disk_io::ExpertIndex;
use crate::executor::ObsidianExecutor;
use crate::kv_cache::KvCache;
use crate::model::Qwen3MoeConfig;
use crate::router::MoeRouter;
use crate::weight_loader::Weights;
use metal::{Buffer, MTLResourceOptions, MTLSize};

#[repr(C)]
#[derive(Clone, Copy)]
struct DequantParams { in_features: u32, out_features: u32, group_size: u32 }

pub struct Model {
    pub cfg: Qwen3MoeConfig,
    pub weights: Weights,
    pub experts: ExpertIndex,
    pub attn: Attention,
    pub router: MoeRouter,
    pub caches: Vec<KvCache>,
}

impl Model {
    /// 4-bit dequant + matvec: out[out_features] = dequant(W) @ x. Reads the
    /// `{name}.weight/.scales/.biases` tensors from the loaded weights.
    fn dequant_matvec(&self, exec: &ObsidianExecutor, name: &str, x: &Buffer,
                      in_features: u32, out_features: u32) -> Result<Buffer, String> {
        let w = self.weights.get(&format!("{name}.weight")).ok_or(format!("{name}.weight"))?;
        let s = self.weights.get(&format!("{name}.scales")).ok_or(format!("{name}.scales"))?;
        let b = self.weights.get(&format!("{name}.biases")).ok_or(format!("{name}.biases"))?;
        let out = exec.device().new_buffer((out_features * 2) as u64,
                                           MTLResourceOptions::StorageModeShared);
        let p = DequantParams { in_features, out_features, group_size: 64 };
        let pso = exec.pipeline("fused_dequant_matvec_4bit").ok_or("dequant kernel")?;
        let cb = exec.queue().new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pso);
        enc.set_buffer(0, Some(w), 0);
        enc.set_buffer(1, Some(s), 0);
        enc.set_buffer(2, Some(b), 0);
        enc.set_buffer(3, Some(x), 0);
        enc.set_buffer(4, Some(&out), 0);
        enc.set_bytes(5, std::mem::size_of::<DequantParams>() as u64,
                      &p as *const _ as *const std::ffi::c_void);
        let tg = 256u64;
        enc.dispatch_thread_groups(MTLSize::new((out_features as u64 + tg - 1) / tg, 1, 1),
                                   MTLSize::new(tg, 1, 1));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        Ok(out)
    }

    /// One decoder layer. Returns the new hidden-state buffer.
    pub fn forward_layer(&mut self, exec: &ObsidianExecutor, il: usize, x: &Buffer)
        -> Result<Buffer, String> {
        let pre = format!("model.layers.{il}.");
        let hidden = self.cfg.hidden_size as u32;
        let hd = self.cfg.head_dim() as u32;
        let nq = self.cfg.num_attention_heads as u32;
        let nkv = self.cfg.num_key_value_heads as u32;

        // QKV projections (4-bit dequant-matvec)
        let q = self.dequant_matvec(exec, &format!("{pre}self_attn.q_proj"), x, hidden, nq * hd)?;
        let _k = self.dequant_matvec(exec, &format!("{pre}self_attn.k_proj"), x, hidden, nkv * hd)?;
        let _v = self.dequant_matvec(exec, &format!("{pre}self_attn.v_proj"), x, hidden, nkv * hd)?;
        // TODO: q_norm/k_norm, append _k/_v to caches[il], then attention.

        let attn_out = self.attn.forward(exec, &q, &self.caches[il])?;
        let o = self.dequant_matvec(exec, &format!("{pre}self_attn.o_proj"), &attn_out, nq * hd, hidden)?;

        // MoE: router top-8 -> page experts off SSD (compute/combine = TODO)
        let gate = self.dequant_matvec(exec, &format!("{pre}mlp.gate"), &o,
                                       hidden, self.cfg.num_experts as u32)?;
        let _experts = self.router.route_and_page(exec, &gate, &self.experts, il)?;
        // TODO: per-expert dequant_matvec(gate/up/down) + fused_moe_combine + residual.

        Ok(o)
    }
}
