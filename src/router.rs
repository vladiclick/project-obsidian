// router.rs — MoE routing (GPU top-k) + expert paging trigger.
// Dispatches moe_routing_topk, reads the selected expert ids back, then fires
// disk_io::read_experts_parallel to page only those experts off SSD.

use crate::disk_io::{read_experts_parallel, ExpertIndex, RawTensor};
use crate::executor::ObsidianExecutor;
use metal::{Buffer, MTLResourceOptions, MTLSize};

#[repr(C)]
#[derive(Clone, Copy)]
struct RoutingParams {
    n_experts: u32,
    top_k: u32,
    norm_topk: u32,
}

pub struct MoeRouter {
    pub n_experts: u32,
    pub top_k: u32,
    pub norm_topk: bool,
}

impl MoeRouter {
    /// Dispatch moe_routing_topk on `gate_logits` ([n_experts] fp16), read back
    /// the top-k expert ids. (weights buffer also produced, kept on GPU.)
    pub fn route(&self, exec: &ObsidianExecutor, gate_logits: &Buffer) -> Result<Vec<u32>, String> {
        let dev = exec.device();
        let k = self.top_k as u64;
        let idx_buf = dev.new_buffer(k * 4, MTLResourceOptions::StorageModeShared);
        let w_buf = dev.new_buffer(k * 2, MTLResourceOptions::StorageModeShared);
        let params = RoutingParams {
            n_experts: self.n_experts,
            top_k: self.top_k,
            norm_topk: self.norm_topk as u32,
        };

        let pso = exec.pipeline("moe_routing_topk").ok_or("moe_routing_topk missing")?;
        let cb = exec.queue().new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pso);
        enc.set_buffer(0, Some(gate_logits), 0);
        enc.set_buffer(1, Some(&idx_buf), 0);
        enc.set_buffer(2, Some(&w_buf), 0);
        enc.set_bytes(3, std::mem::size_of::<RoutingParams>() as u64,
                      &params as *const _ as *const std::ffi::c_void);
        enc.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        let ptr = idx_buf.contents() as *const i32;
        let ids: Vec<u32> = (0..self.top_k as isize)
            .map(|i| unsafe { *ptr.offset(i) } as u32)
            .collect();
        Ok(ids)
    }

    /// Full step: route, then page the selected experts off disk.
    pub fn route_and_page(
        &self,
        exec: &ObsidianExecutor,
        gate_logits: &Buffer,
        index: &ExpertIndex,
        layer: usize,
    ) -> Result<Vec<RawTensor>, String> {
        let ids = self.route(exec, gate_logits)?;
        Ok(read_experts_parallel(index, layer, &ids))
    }
}
