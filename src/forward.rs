// forward.rs — Model + full single-token forward over Qwen3-30B-A3B.
// First-run goals: load the real model, run end-to-end, produce a token id, and
// surface concrete dim/alignment errors. Correctness (esp. 4-bit packing vs MLX,
// RoPE, q/k-norm) is UNVERIFIED. q_norm/k_norm currently SKIPPED (flagged).

use crate::attention::Attention;
use crate::disk_io::{ExpertIndex, RawTensor};
use crate::executor::ObsidianExecutor;
use crate::kv_cache::KvCache;
use crate::model::Qwen3MoeConfig;
use crate::router::MoeRouter;
use crate::weight_loader::Weights;
use half::f16;
use metal::{Buffer, MTLResourceOptions, MTLSize};
use std::collections::HashMap;
use std::path::Path;

const SHARED: MTLResourceOptions = MTLResourceOptions::StorageModeShared;

#[repr(C)] #[derive(Clone, Copy)] struct DequantParams { in_features: u32, out_features: u32, group_size: u32 }
#[repr(C)] #[derive(Clone, Copy)] struct RmsParams { dim: u32, eps: f32 }

pub struct Model {
    pub cfg: Qwen3MoeConfig,
    pub weights: Weights,
    pub experts: ExpertIndex,
    pub attn: Attention,
    pub router: MoeRouter,
    pub caches: Vec<KvCache>,
}

impl Model {
    pub fn new(model_dir: &Path, expert_dir: &Path, exec: &ObsidianExecutor) -> Result<Self, String> {
        let cfg = Qwen3MoeConfig::from_file(&model_dir.join("config.json"))?;
        let weights = Weights::load_non_expert(model_dir, exec.device()).map_err(|e| format!("{e:?}"))?;
        let experts = ExpertIndex::open(expert_dir, cfg.num_hidden_layers).map_err(|e| format!("{e:?}"))?;
        let attn = Attention {
            n_heads: cfg.num_attention_heads as u32, n_kv: cfg.num_key_value_heads as u32,
            head_dim: cfg.head_dim() as u32, theta: cfg.rope_theta,
        };
        let router = MoeRouter {
            n_experts: cfg.num_experts as u32, top_k: cfg.num_experts_per_tok as u32,
            norm_topk: cfg.norm_topk_prob,
        };
        let caches = (0..cfg.num_hidden_layers)
            .map(|_| KvCache::new(exec.device(), cfg.num_key_value_heads, cfg.head_dim(), 4096))
            .collect();
        Ok(Self { cfg, weights, experts, attn, router, caches })
    }

    // ---- low-level helpers ----
    fn buf(&self, exec: &ObsidianExecutor, bytes: usize) -> Buffer {
        exec.device().new_buffer(bytes as u64, SHARED)
    }
    fn buf_from(&self, exec: &ObsidianExecutor, data: &[u8]) -> Buffer {
        let b = self.buf(exec, data.len());
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), b.contents() as *mut u8, data.len()); }
        b
    }
    fn read_f16(buf: &Buffer, n: usize) -> Vec<f32> {
        let p = buf.contents() as *const u16;
        (0..n).map(|i| f16::from_bits(unsafe { *p.add(i) }).to_f32()).collect()
    }
    fn f16_buf(&self, exec: &ObsidianExecutor, v: &[f32]) -> Buffer {
        let bits: Vec<u16> = v.iter().map(|&x| f16::from_f32(x).to_bits()).collect();
        self.buf_from(exec, bytemuck::cast_slice(&bits))
    }

    fn dequant_matvec_bufs(&self, exec: &ObsidianExecutor, w: &Buffer, s: &Buffer, b: &Buffer,
                           x: &Buffer, in_f: u32, out_f: u32) -> Result<Buffer, String> {
        let out = self.buf(exec, out_f as usize * 2);
        let p = DequantParams { in_features: in_f, out_features: out_f, group_size: 64 };
        let pso = exec.pipeline("fused_dequant_matvec_4bit").ok_or("dequant kernel")?;
        let cb = exec.queue().new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pso);
        for (i, bf) in [w, s, b, x, &out].iter().enumerate() { enc.set_buffer(i as u64, Some(*bf), 0); }
        enc.set_bytes(5, std::mem::size_of::<DequantParams>() as u64, &p as *const _ as *const _);
        let tg = 256u64;
        enc.dispatch_thread_groups(MTLSize::new((out_f as u64 + tg - 1) / tg, 1, 1), MTLSize::new(tg, 1, 1));
        enc.end_encoding(); cb.commit(); cb.wait_until_completed();
        Ok(out)
    }
    fn dequant_named(&self, exec: &ObsidianExecutor, name: &str, x: &Buffer, in_f: u32, out_f: u32) -> Result<Buffer, String> {
        let w = self.weights.get(&format!("{name}.weight")).ok_or(format!("missing {name}.weight"))?;
        let s = self.weights.get(&format!("{name}.scales")).ok_or(format!("missing {name}.scales"))?;
        let b = self.weights.get(&format!("{name}.biases")).ok_or(format!("missing {name}.biases"))?;
        self.dequant_matvec_bufs(exec, &w.clone(), &s.clone(), &b.clone(), x, in_f, out_f)
    }
    fn rmsnorm(&self, exec: &ObsidianExecutor, x: &Buffer, w: &Buffer, dim: u32, rows: u32) -> Result<Buffer, String> {
        let out = self.buf(exec, (dim * rows) as usize * 2);
        let p = RmsParams { dim, eps: self.cfg.rms_norm_eps };
        let pso = exec.pipeline("rmsnorm").ok_or("rmsnorm kernel")?;
        let cb = exec.queue().new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pso);
        for (i, bf) in [x, w, &out].iter().enumerate() { enc.set_buffer(i as u64, Some(*bf), 0); }
        enc.set_bytes(3, std::mem::size_of::<RmsParams>() as u64, &p as *const _ as *const _);
        enc.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding(); cb.commit(); cb.wait_until_completed();
        Ok(out)
    }

    pub fn forward_layer(&mut self, exec: &ObsidianExecutor, il: usize, x: &Buffer) -> Result<Buffer, String> {
        let pre = format!("model.layers.{il}.");
        let h = self.cfg.hidden_size as u32;
        let hd = self.cfg.head_dim() as u32;
        let nq = self.cfg.num_attention_heads as u32;
        let nkv = self.cfg.num_key_value_heads as u32;

        // input norm
        let xn = self.rmsnorm(exec, x, &self.weights.get(&format!("{pre}input_layernorm.weight")).ok_or("input_ln")?.clone(), h, 1)?;
        // QKV  (q_norm/k_norm SKIPPED for first run)
        let q = self.dequant_named(exec, &format!("{pre}self_attn.q_proj"), &xn, h, nq * hd)?;
        let k = self.dequant_named(exec, &format!("{pre}self_attn.k_proj"), &xn, h, nkv * hd)?;
        let v = self.dequant_named(exec, &format!("{pre}self_attn.v_proj"), &xn, h, nkv * hd)?;
        // append projected k/v to cache (CPU readback of fp16)
        let kbits: Vec<u16> = (0..(nkv*hd) as usize).map(|i| unsafe { *(k.contents() as *const u16).add(i) }).collect();
        let vbits: Vec<u16> = (0..(nkv*hd) as usize).map(|i| unsafe { *(v.contents() as *const u16).add(i) }).collect();
        self.caches[il].append(&kbits, &vbits);
        // attention (rope(q) + sdpa)
        let attn_out = self.attn.forward(exec, &q, &self.caches[il])?;
        let o = self.dequant_named(exec, &format!("{pre}self_attn.o_proj"), &attn_out, nq * hd, h)?;
        // residual
        let mut hs = Self::read_f16(x, h as usize);
        let ov = Self::read_f16(&o, h as usize);
        for i in 0..h as usize { hs[i] += ov[i]; }
        let x1 = self.f16_buf(exec, &hs);

        // MoE
        let x1n = self.rmsnorm(exec, &x1, &self.weights.get(&format!("{pre}post_attention_layernorm.weight")).ok_or("post_ln")?.clone(), h, 1)?;
        let gate = self.dequant_named(exec, &format!("{pre}mlp.gate"), &x1n, h, self.cfg.num_experts as u32)?;
        let (ids, wts) = self.router.route(exec, &gate)?;
        let paged = crate::disk_io::read_experts_parallel(&self.experts, il, &ids);
        let mut by_e: HashMap<u32, HashMap<String, Vec<u8>>> = HashMap::new();
        for RawTensor { eid, key, bytes } in paged { by_e.entry(eid).or_default().insert(key, bytes); }

        let moe_inter = self.cfg.moe_intermediate_size as u32;
        let mut acc = vec![0f32; h as usize];
        for (rank, &e) in ids.iter().enumerate() {
            let t = by_e.get(&e).ok_or(format!("expert {e} not paged"))?;
            let gw = self.buf_from(exec, &t["gate_proj.weight"]); let gs = self.buf_from(exec, &t["gate_proj.scales"]); let gb = self.buf_from(exec, &t["gate_proj.biases"]);
            let uw = self.buf_from(exec, &t["up_proj.weight"]);   let us = self.buf_from(exec, &t["up_proj.scales"]);   let ub = self.buf_from(exec, &t["up_proj.biases"]);
            let dw = self.buf_from(exec, &t["down_proj.weight"]); let ds = self.buf_from(exec, &t["down_proj.scales"]); let db = self.buf_from(exec, &t["down_proj.biases"]);
            let g = self.dequant_matvec_bufs(exec, &gw, &gs, &gb, &x1n, h, moe_inter)?;
            let u = self.dequant_matvec_bufs(exec, &uw, &us, &ub, &x1n, h, moe_inter)?;
            let gv = Self::read_f16(&g, moe_inter as usize);
            let uv = Self::read_f16(&u, moe_inter as usize);
            let sg: Vec<f32> = (0..moe_inter as usize).map(|i| { let x = gv[i]; (x / (1.0 + (-x).exp())) * uv[i] }).collect();
            let sgb = self.f16_buf(exec, &sg);
            let y = self.dequant_matvec_bufs(exec, &dw, &ds, &db, &sgb, moe_inter, h)?;
            let yv = Self::read_f16(&y, h as usize);
            let w = wts[rank];
            for i in 0..h as usize { acc[i] += w * yv[i]; }
        }
        for i in 0..h as usize { acc[i] += hs[i]; } // residual
        Ok(self.f16_buf(exec, &acc))
    }

    /// CPU-dequant one embedding row -> fp16 buffer [hidden].
    /// NOTE: assumes f16 scales/biases; mlx repos store them BF16 for non-expert
    /// tensors -> this (and the dequant shader) will be WRONG until bf16 is
    /// converted at load. Flagged; first run is for plumbing/dim errors.
    pub fn embed(&self, exec: &ObsidianExecutor, tok: u32) -> Buffer {
        let h = self.cfg.hidden_size;
        let (words, groups) = (h / 8, h / 64);
        let w = self.weights.get("model.embed_tokens.weight").unwrap();
        let s = self.weights.get("model.embed_tokens.scales").unwrap();
        let b = self.weights.get("model.embed_tokens.biases").unwrap();
        let (wq, sp, bp) = (w.contents() as *const u32, s.contents() as *const u16, b.contents() as *const u16);
        let mut row = vec![0f32; h];
        for i in 0..h {
            let word = unsafe { *wq.add(tok as usize * words + i / 8) };
            let nib = ((word >> (4 * (i % 8))) & 0xF) as f32;
            let gi = tok as usize * groups + i / 64;
            let sc = f16::from_bits(unsafe { *sp.add(gi) }).to_f32();
            let bi = f16::from_bits(unsafe { *bp.add(gi) }).to_f32();
            row[i] = nib * sc + bi;
        }
        self.f16_buf(exec, &row)
    }

    fn stat(label: &str, buf: &Buffer, n: usize) {
        let v = Self::read_f16(buf, n);
        let (mut mn, mut mx, mut nan) = (f32::INFINITY, f32::NEG_INFINITY, 0usize);
        for &x in &v { if x.is_nan() { nan += 1 } else { mn = mn.min(x); mx = mx.max(x) } }
        eprintln!("[stat] {label}: min={mn:.4} max={mx:.4} nan={nan}/{n}");
    }

    pub fn run_one_token(&mut self, exec: &ObsidianExecutor, tok: u32) -> Result<u32, String> {
        let mut x = self.embed(exec, tok);
        Self::stat("embed", &x, self.cfg.hidden_size);
        for il in 0..self.cfg.num_hidden_layers {
            x = self.forward_layer(exec, il, &x)?;
            if il == 0 || il == 1 { Self::stat(&format!("layer{il}"), &x, self.cfg.hidden_size); }
        }
        let h = self.cfg.hidden_size as u32;
        let nw = self.weights.get("model.norm.weight").ok_or("model.norm.weight")?.clone();
        let xn = self.rmsnorm(exec, &x, &nw, h, 1)?;
        Self::stat("final_norm", &xn, self.cfg.hidden_size);
        let logits = self.dequant_named(exec, "lm_head", &xn, h, self.cfg.vocab_size as u32)?;
        Self::stat("logits", &logits, self.cfg.vocab_size);
        let lv = Self::read_f16(&logits, self.cfg.vocab_size);
        let mut best = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for (i, &l) in lv.iter().enumerate() { if l > bv { bv = l; best = i; } }
        Ok(best as u32)
    }
}
