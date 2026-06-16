// executor.rs — Metal-native foundation (Option C pivot).
//
// Pure metal-rs: Metal device + runtime MSL compilation + one
// ComputePipelineState per custom kernel. No mlx-native. mlx-rs (tensor math)
// is feature-gated behind `--features mlx`; this base needs neither it nor the
// offline Metal CLI toolchain (runtime compilation uses the Metal framework).

use metal::{CompileOptions, ComputePipelineState, Device, Library};
use std::collections::HashMap;

const SHADERS: [(&str, &str); 4] = [
    ("fused_dequant_matvec_4bit",
        include_str!("../shaders/fused_dequant_matvec_4bit.metal")),
    ("moe_routing_topk",
        include_str!("../shaders/moe_routing_topk.metal")),
    ("fused_moe_combine",
        include_str!("../shaders/fused_moe_combine.metal")),
    ("fused_residual_rms_norm",
        include_str!("../shaders/fused_residual_rms_norm.metal")),
];

pub struct ObsidianExecutor {
    device: Device,
    queue: metal::CommandQueue,
    pipelines: HashMap<&'static str, ComputePipelineState>,
}

impl ObsidianExecutor {
    /// Initialize the Metal device, compile every embedded .metal shader at
    /// runtime, and build a compute pipeline per kernel.
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("no Metal device found")?;
        let queue = device.new_command_queue();

        let mut pipelines = HashMap::new();
        for (name, src) in SHADERS {
            let opts = CompileOptions::new();
            let lib: Library = device.new_library_with_source(src, &opts)?;
            let func = lib.get_function(name, None)?;
            let pso = device.new_compute_pipeline_state_with_function(&func)?;
            pipelines.insert(name, pso);
        }

        log::info!(
            "Metal device '{}' ready; {} custom kernels compiled",
            device.name(),
            pipelines.len()
        );
        Ok(Self { device, queue, pipelines })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn queue(&self) -> &metal::CommandQueue {
        &self.queue
    }

    pub fn pipeline(&self, name: &str) -> Option<&ComputePipelineState> {
        self.pipelines.get(name)
    }
}
