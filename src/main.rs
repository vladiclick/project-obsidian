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
    #[arg(long, default_value_t = false)]
    serve: bool,
    #[arg(long, default_value_t = 8000)]
    port: u16,
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

    if args.serve {
        serve(&args);
    }
}

fn serve(args: &Args) {
    use std::io::Read;
    let exec = executor::ObsidianExecutor::new().expect("metal");
    let model_dir = std::path::Path::new(&args.config).parent().expect("--config").to_path_buf();
    let expert_dir = if args.expert_dir.is_empty() { model_dir.join("splitted_moe") }
                     else { std::path::PathBuf::from(&args.expert_dir) };
    let mut m = forward::Model::new(&model_dir, &expert_dir, &exec).expect("model");
    let tk = tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json")).expect("tokenizer");
    let eos = tk.token_to_id("<|im_end|>").unwrap_or(151645);

    let addr = format!("0.0.0.0:{}", args.port);
    let server = tiny_http::Server::http(&addr).unwrap_or_else(|e| { eprintln!("bind {addr}: {e}"); std::process::exit(1); });
    println!("[obsidian] API LIVE on http://{addr}/v1/chat/completions");

    for mut req in server.incoming_requests() {
        let mut body = String::new();
        let _ = req.as_reader().read_to_string(&mut body);
        let v: serde_json::Value = serde_json::from_str(&body).unwrap_or(serde_json::json!({}));
        let max_tokens = v["max_tokens"].as_u64().unwrap_or(128).clamp(1, 1024) as usize;

        let mut prompt = String::new();
        if let Some(msgs) = v["messages"].as_array() {
            for msg in msgs {
                let role = msg["role"].as_str().unwrap_or("user");
                let content = msg["content"].as_str().unwrap_or("");
                prompt.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
            }
        }
        prompt.push_str("<|im_start|>assistant\n");

        let ids: Vec<u32> = tk.encode(prompt, false).expect("encode").get_ids().to_vec();
        m.reset_cache();
        let mut next = 0u32;
        for &t in &ids { next = m.run_one_token(&exec, t).expect("prefill"); }
        let mut gen = vec![next];
        for _ in 0..max_tokens - 1 {
            if next == eos { break; }
            next = m.run_one_token(&exec, next).expect("decode");
            gen.push(next);
        }
        let text = tk.decode(&gen, true).unwrap_or_default();
        let out = serde_json::json!({
            "id": "chatcmpl-obsidian", "object": "chat.completion",
            "created": 0, "model": "obsidian-qwen3-30b-a3b",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": ids.len(), "completion_tokens": gen.len(), "total_tokens": ids.len() + gen.len()}
        });
        println!("[obsidian] req: {} prompt tok -> {} gen tok", ids.len(), gen.len());
        let resp = tiny_http::Response::from_string(out.to_string())
            .with_header("Content-Type: application/json".parse::<tiny_http::Header>().unwrap());
        let _ = req.respond(resp);
    }
}
