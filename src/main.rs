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
    expert_dir: String,
    #[arg(long, default_value_t = false)]
    run: bool,
    #[arg(long, default_value_t = 9707)]
    token: u32,
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

    if args.run {
        let exec = match executor::ObsidianExecutor::new() {
            Ok(e) => e, Err(e) => { eprintln!("[obsidian] executor: {e}"); std::process::exit(1); }
        };
        let model_dir = std::path::Path::new(&args.config).parent()
            .expect("--config must point at config.json").to_path_buf();
        let expert_dir = if args.expert_dir.is_empty() {
            model_dir.join("splitted_moe")
        } else { std::path::PathBuf::from(&args.expert_dir) };
        println!("[obsidian] loading model from {} ...", model_dir.display());
        let t0 = std::time::Instant::now();
        let mut m = match forward::Model::new(&model_dir, &expert_dir, &exec) {
            Ok(m) => m, Err(e) => { eprintln!("[obsidian] model load FAILED: {e}"); std::process::exit(1); }
        };
        let n = args.max_tokens.max(1);
        println!("[obsidian] loaded in {:.1}s", t0.elapsed().as_secs_f32());

        if !args.prompt.is_empty() {
            // ---- prompt mode: real chat-templated generation ----
            let tk = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
                .expect("load tokenizer.json");
            let wrapped = format!(
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", args.prompt);
            let ids: Vec<u32> = tk.encode(wrapped, false).expect("encode").get_ids().to_vec();
            let eos = tk.token_to_id("<|im_end|>").unwrap_or(151645);
            println!("[obsidian] prompt = {:?} ({} tokens); generating {} ...", args.prompt, ids.len(), n);
            let t1 = std::time::Instant::now();
            let mut next = 0u32;
            for &t in &ids { next = m.run_one_token(&exec, t).expect("prefill"); }
            let mut gen = vec![next];
            for _ in 0..n - 1 {
                if next == eos { break; }
                next = m.run_one_token(&exec, next).expect("decode");
                gen.push(next);
            }
            let secs = t1.elapsed().as_secs_f32();
            let text = tk.decode(&gen, true).unwrap_or_default();
            println!("[obsidian] ---- OUTPUT ----\n{}\n----------------", text);
            println!("[obsidian] {} tokens in {:.2}s = {:.2} tok/s", gen.len(), secs, gen.len() as f32 / secs);
        } else {
            // ---- token mode: timing only ----
            println!("[obsidian] generating {} tokens (input id {}) ...", n, args.token);
            let mut tok = args.token;
            let mut times = Vec::new();
            for i in 0..n {
                let t1 = std::time::Instant::now();
                let next = m.run_one_token(&exec, tok).expect("forward");
                let dt = t1.elapsed().as_secs_f32();
                println!("[obsidian] tok {i}: id={next} ({:.3}s, {:.1}ms/layer)", dt, dt * 1000.0 / m.cfg.num_hidden_layers as f32);
                times.push(dt); tok = next;
            }
            if times.len() > 1 {
                let ss: f32 = times[1..].iter().sum::<f32>() / (times.len() - 1) as f32;
                println!("[obsidian] STEADY-STATE: {:.3}s/token = {:.2} tok/s, {:.1}ms/layer",
                         ss, 1.0 / ss, ss * 1000.0 / m.cfg.num_hidden_layers as f32);
            }
        }
    }
}
