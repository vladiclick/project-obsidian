// Project Obsidian — Phase 1 skeleton (CLI + module wiring only).
// Generation (generate.rs), router, attention, kv_cache = Phase 2, gated on a
// successful `cargo build` of this base (see BUILD.md for current blockers).

mod executor;
mod disk_io;
mod model;
mod router;
mod kv_cache;
mod attention;
mod weight_loader;
mod forward;

use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "obsidian")]
struct Args {
    #[arg(long)]
    model: String,
    #[arg(long, default_value = "")]
    config: String,
    #[arg(long, default_value = "")]
    prompt: String,
    #[arg(long, default_value_t = 50)]
    max_tokens: usize,
    #[arg(long, default_value_t = false)]
    benchmark: bool,
}

fn main() {
    env_logger::init();
    let args = Args::parse();
    log::info!("Project Obsidian Phase 1 skeleton. model={}", args.model);
    let _ = (&args.prompt, args.max_tokens, args.benchmark);

    // Phase 1 self-test: init Metal device + runtime-compile all 4 .metal kernels.
    match executor::ObsidianExecutor::new() {
        Ok(exec) => {
            println!("[obsidian] OK: Metal device '{}' up.", exec.device().name());
            for k in ["fused_dequant_matvec_4bit", "moe_routing_topk",
                      "fused_moe_combine", "fused_residual_rms_norm",
                      "rope_apply", "sdpa_decode"] {
                let ok = exec.pipeline(k).is_some();
                println!("[obsidian]   kernel {k}: {}", if ok { "compiled ✓" } else { "MISSING" });
            }
        }
        Err(e) => {
            eprintln!("[obsidian] executor init FAILED: {e}");
            std::process::exit(1);
        }
    }

    if !args.config.is_empty() {
        match model::Qwen3MoeConfig::from_file(std::path::Path::new(&args.config)) {
            Ok(c) => println!(
                "[obsidian] config: {} layers, hidden {}, {} experts top-{}, \
                 head_dim {}, rope_theta {}, vocab {}",
                c.num_hidden_layers, c.hidden_size, c.num_experts,
                c.num_experts_per_tok, c.head_dim(), c.rope_theta, c.vocab_size
            ),
            Err(e) => eprintln!("[obsidian] config parse failed: {e}"),
        }
    }
}
