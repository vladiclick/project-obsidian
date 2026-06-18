// forward.rs — GPU-resident single-token forward over Qwen3-30B-A3B.
// Dispatches commit WITHOUT wait (Metal serializes same-queue command buffers),
// elementwise ops (residual/swiglu/accumulate/kv-append) run on the GPU, and the
// only per-layer CPU sync is the router top-k readback (needed to page experts
// off SSD). Validated to still produce token 271.

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
use std::ffi::c_void;
use std::path::Path;

const SHARED: MTLResourceOptions = MTLResourceOptions::StorageModeShared;

#[repr(C)] #[derive(Clone, Copy)] struct DequantParams { in_features: u32, out_features: u32, group_size: u32 }
#[repr(C)] #[derive(Clone, Copy)] struct RmsParams { dim: u32, eps: f32 }
#[repr(C)] #[derive(Clone, Copy)] struct RopeParams { n_heads: u32, head_dim: u32, pos: u32, theta: f32 }
#[repr(C)] #[derive(Clone, Copy)] struct N { n: u32 }
#[repr(C)] #[derive(Clone, Copy)] struct ScaledN { n: u32, scale: f32 }
#[repr(C)] #[derive(Clone, Copy)] struct KvP { n_kv: u32, head_dim: u32, max_len: u32, pos: u32 }

pub struct Model {
    pub cfg: Qwen3MoeConfig,
    pub weights: Weights,
    pub experts: ExpertIndex,
    pub attn: Attention,
    pub router: MoeRouter,
    pub caches: Vec<KvCache>,
    /// Pre-allocated expert scratch: top_k slots x 9 tensors (gate/up/down x
    /// weight/scales/biases). Reused across layers (router drain syncs them).
    expert_pool: Vec<Buffer>,
    /// Full token sequence currently materialized in the KV cache
    /// (prompt + generated). Enables cross-request prefix reuse: a new turn
    /// whose prompt shares this prefix skips re-prefilling the common part.
    last_seq: Vec<u32>,
    /// Last-turn prefix-reuse metrics (exposed via /health).
    pub last_reused: usize,
    pub last_prefilled: usize,
    pub last_prefill_ms: u128,
    pub turns: u64,
    /// Batched-prefill expert pool (n_experts x 9 buffers), allocated lazily.
    /// Lets the whole layer's activated experts stay resident so a batch of
    /// prompt tokens reuses them (load once/layer instead of once/token).
    batch_pool: Option<Vec<Buffer>>,
}

impl Model {
    pub fn new(model_dir: &Path, expert_dir: &Path, exec: &ObsidianExecutor) -> Result<Self, String> {
        let cfg = Qwen3MoeConfig::from_file(&model_dir.join("config.json"))?;
        let weights = Weights::load_non_expert(model_dir, exec.device()).map_err(|e| format!("{e:?}"))?;
        let experts = ExpertIndex::open(expert_dir, cfg.num_hidden_layers).map_err(|e| format!("{e:?}"))?;
        // KV context window. Coding sessions need far more than the old 4096;
        // 16384 fits comfortably in 16GB (n_kv 4 * 16384 * head_dim 128 * 2B *
        // 2(k,v) * 48 layers ≈ 1.6GB). Override with OBSIDIAN_MAX_LEN.
        let max_len = std::env::var("OBSIDIAN_MAX_LEN").ok()
            .and_then(|s| s.parse::<usize>().ok()).unwrap_or(16384);
        let attn = Attention {
            n_heads: cfg.num_attention_heads as u32, n_kv: cfg.num_key_value_heads as u32,
            head_dim: cfg.head_dim() as u32, theta: cfg.rope_theta,
            // device score scratch [n_heads * max_len] f32, allocated once
            scratch: exec.device().new_buffer(
                (cfg.num_attention_heads * max_len * 4) as u64, SHARED),
        };
        let router = MoeRouter {
            n_experts: cfg.num_experts as u32, top_k: cfg.num_experts_per_tok as u32,
            norm_topk: cfg.norm_topk_prob,
        };
        let caches = (0..cfg.num_hidden_layers)
            .map(|_| KvCache::new(exec.device(), cfg.num_key_value_heads, cfg.head_dim(), max_len))
            .collect();
        // expert scratch pool: top_k slots x 9 tensors (w=moe_inter*hidden/2 bytes
        // for 4-bit; scales/biases = moe_inter*hidden/64*2)
        let w_size = (cfg.moe_intermediate_size * cfg.hidden_size / 2) as u64;
        let sb_size = (cfg.moe_intermediate_size * cfg.hidden_size / 64 * 2) as u64;
        let mut expert_pool = Vec::with_capacity(cfg.num_experts_per_tok * 9);
        for _ in 0..cfg.num_experts_per_tok {
            for k in 0..9 {
                let sz = if k % 3 == 0 { w_size } else { sb_size };
                expert_pool.push(exec.device().new_buffer(sz, SHARED));
            }
        }
        Ok(Self { cfg, weights, experts, attn, router, caches, expert_pool, last_seq: Vec::new(),
                  last_reused: 0, last_prefilled: 0, last_prefill_ms: 0, turns: 0, batch_pool: None })
    }

    /// Lazily allocate the batched-prefill expert pool (n_experts x 9 buffers).
    fn ensure_batch_pool(&mut self, exec: &ObsidianExecutor) {
        if self.batch_pool.is_some() { return; }
        let w_size = (self.cfg.moe_intermediate_size * self.cfg.hidden_size / 2) as u64;
        let sb_size = (self.cfg.moe_intermediate_size * self.cfg.hidden_size / 64 * 2) as u64;
        let mut pool = Vec::with_capacity(self.cfg.num_experts * 9);
        for _ in 0..self.cfg.num_experts {
            for k in 0..9 { pool.push(exec.device().new_buffer(if k % 3 == 0 { w_size } else { sb_size }, SHARED)); }
        }
        self.batch_pool = Some(pool);
    }

    /// BATCHED PREFILL (MEGAPROMPT V2 Phase 1): process `tokens` layer-major.
    /// For each layer: run all T tokens' attention (filling KV), then ONE drain
    /// for all T gate+route dispatches, load the union of activated experts ONCE
    /// off SSD, and apply experts per token from the resident pool. Reuses the
    /// existing per-token Metal kernels -> bit-exact with the per-token path,
    /// but cuts the per-layer router CPU sync from N×layers to layers, and
    /// expert SSD reads from per-token to once/layer. Returns the next token id.
    /// Errors are recoverable: the caller falls back to per-token prefill.
    pub fn prefill_batched(&mut self, exec: &ObsidianExecutor, tokens: &[u32]) -> Result<u32, String> {
        let t = tokens.len();
        if t == 0 { return Err("empty prompt".into()); }
        let (h, hd, nq, nkv) = (self.cfg.hidden_size as u32, self.cfg.head_dim() as u32,
                                self.cfg.num_attention_heads as u32, self.cfg.num_key_value_heads as u32);
        let moe_inter = self.cfg.moe_intermediate_size as u32;
        let base_pos = self.caches[0].seq_len;
        if base_pos + t > self.caches[0].max_len { return Err("context overflow".into()); }
        self.ensure_batch_pool(exec);

        const KEYS: [&str; 9] = ["gate_proj.weight", "gate_proj.scales", "gate_proj.biases",
                                 "up_proj.weight", "up_proj.scales", "up_proj.biases",
                                 "down_proj.weight", "down_proj.scales", "down_proj.biases"];

        let mut xs: Vec<Buffer> = tokens.iter().map(|&tok| self.embed(exec, tok)).collect();

        for il in 0..self.cfg.num_hidden_layers {
            let pre = format!("model.layers.{il}.");
            let in_ln = self.weights.get(&format!("{pre}input_layernorm.weight")).ok_or("in_ln")?.clone();
            let qn = self.weights.get(&format!("{pre}self_attn.q_norm.weight")).ok_or("qn")?.clone();
            let kn = self.weights.get(&format!("{pre}self_attn.k_norm.weight")).ok_or("kn")?.clone();
            let post_ln = self.weights.get(&format!("{pre}post_attention_layernorm.weight")).ok_or("post")?.clone();
            let ml = self.caches[il].max_len as u32;

            // --- attention block: per token in order, fills KV, no per-token sync ---
            let mut x1s: Vec<Buffer> = Vec::with_capacity(t);
            for i in 0..t {
                let pos = (base_pos + i) as u32;
                self.caches[il].seq_len = base_pos + i;
                let x = &xs[i];
                let xn = self.rmsnorm(exec, x, &in_ln, h, 1)?;
                let q = self.dequant_named(exec, &format!("{pre}self_attn.q_proj"), &xn, h, nq * hd)?;
                let k = self.dequant_named(exec, &format!("{pre}self_attn.k_proj"), &xn, h, nkv * hd)?;
                let v = self.dequant_named(exec, &format!("{pre}self_attn.v_proj"), &xn, h, nkv * hd)?;
                let q = self.rmsnorm(exec, &q, &qn, hd, nq)?;
                let k = self.rmsnorm(exec, &k, &kn, hd, nkv)?;
                self.rope(exec, &q, nq, hd, pos)?;
                self.rope(exec, &k, nkv, hd, pos)?;
                self.kv_append_gpu(exec, &k, &self.caches[il].k.clone(), nkv, hd, ml, pos)?;
                self.kv_append_gpu(exec, &v, &self.caches[il].v.clone(), nkv, hd, ml, pos)?;
                self.caches[il].seq_len = base_pos + i + 1;
                let attn = self.attn.sdpa(exec, &q, &self.caches[il])?;
                let o = self.dequant_named(exec, &format!("{pre}self_attn.o_proj"), &attn, nq * hd, h)?;
                x1s.push(self.add(exec, x, &o, h)?);
            }

            // --- MoE: batched gate + route (no wait), ONE drain ---
            let mut x1ns: Vec<Buffer> = Vec::with_capacity(t);
            let mut routes: Vec<(Buffer, Buffer)> = Vec::with_capacity(t);
            for i in 0..t {
                let x1n = self.rmsnorm(exec, &x1s[i], &post_ln, h, 1)?;
                let gate = self.gate_gpu(exec, &pre, &x1n)?;
                routes.push(self.router.route_dispatch(exec, &gate)?);
                x1ns.push(x1n);
            }
            Self::drain(exec);

            // read routes, gather union of experts
            let mut per_tok: Vec<(Vec<u32>, Vec<f32>)> = Vec::with_capacity(t);
            let mut union: Vec<u32> = Vec::new();
            for (idx_buf, w_buf) in &routes {
                let (ids, wts) = self.router.read_route(idx_buf, w_buf);
                for &e in &ids { if !union.contains(&e) { union.push(e); } }
                per_tok.push((ids, wts));
            }

            // load union experts ONCE, copy into resident pool slots
            let paged = crate::disk_io::read_experts_parallel(&self.experts, il, &union);
            let mut by_e: HashMap<u32, HashMap<String, Vec<u8>>> = HashMap::new();
            for RawTensor { eid, key, bytes } in paged { by_e.entry(eid).or_default().insert(key, bytes); }
            let mut slot_of: HashMap<u32, usize> = HashMap::new();
            for (slot, &e) in union.iter().enumerate() {
                let tmap = by_e.get(&e).ok_or(format!("expert {e} not paged"))?;
                let base = slot * 9;
                for (kk, key) in KEYS.iter().enumerate() {
                    self.copy_into(&self.batch_pool.as_ref().unwrap()[base + kk],
                                   tmap.get(*key).ok_or(format!("expert {e} missing {key}"))?);
                }
                slot_of.insert(e, slot);
            }

            // apply experts per token (no sync), accumulate into x1s[i]
            for i in 0..t {
                let (ids, wts) = &per_tok[i];
                for (rank, &e) in ids.iter().enumerate() {
                    let base = slot_of[&e] * 9;
                    let p = self.batch_pool.as_ref().unwrap();
                    let g = self.dequant(exec, &p[base], &p[base + 1], &p[base + 2], &x1ns[i], h, moe_inter)?;
                    let u = self.dequant(exec, &p[base + 3], &p[base + 4], &p[base + 5], &x1ns[i], h, moe_inter)?;
                    let sg = self.swiglu(exec, &g, &u, moe_inter)?;
                    let y = self.dequant(exec, &p[base + 6], &p[base + 7], &p[base + 8], &sg, moe_inter, h)?;
                    self.scaled_add(exec, &x1s[i], &y, wts[rank], h)?;
                }
            }
            xs = x1s;
        }
        self.caches.iter_mut().for_each(|c| c.seq_len = base_pos + t);

        // last token logits -> next token
        let nw = self.weights.get("model.norm.weight").ok_or("model.norm")?.clone();
        let xn = self.rmsnorm(exec, &xs[t - 1], &nw, h, 1)?;
        let logits = self.dequant_named(exec, "lm_head", &xn, h, self.cfg.vocab_size as u32)?;
        Self::drain(exec);
        let lv = Self::read_f16(&logits, self.cfg.vocab_size);
        let mut best = 0usize; let mut bv = f32::NEG_INFINITY;
        for (i, &l) in lv.iter().enumerate() { if l > bv { bv = l; best = i; } }
        Ok(best as u32)
    }

    // ---- buffers ----
    fn buf(&self, exec: &ObsidianExecutor, bytes: usize) -> Buffer { exec.device().new_buffer(bytes as u64, SHARED) }
    fn buf_from(&self, exec: &ObsidianExecutor, data: &[u8]) -> Buffer {
        let b = self.buf(exec, data.len());
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), b.contents() as *mut u8, data.len()); }
        b
    }
    fn copy_into(&self, buf: &Buffer, data: &[u8]) {
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), buf.contents() as *mut u8, data.len()); }
    }
    fn read_f16(buf: &Buffer, n: usize) -> Vec<f32> {
        let p = buf.contents() as *const u16;
        (0..n).map(|i| f16::from_bits(unsafe { *p.add(i) }).to_f32()).collect()
    }
    fn f16_buf(&self, exec: &ObsidianExecutor, v: &[f32]) -> Buffer {
        let bits: Vec<u16> = v.iter().map(|&x| f16::from_f32(x).to_bits()).collect();
        self.buf_from(exec, bytemuck::cast_slice(&bits))
    }
    /// Drain the queue (empty cb + wait) — the only place the CPU blocks.
    fn drain(exec: &ObsidianExecutor) {
        let cb = exec.queue().new_command_buffer();
        cb.commit();
        cb.wait_until_completed();
    }

    // ---- generic 1-D dispatch: commit, NO wait. params at slot bufs.len() ----
    fn disp(&self, exec: &ObsidianExecutor, kernel: &str, bufs: &[&Buffer],
            params: *const c_void, plen: usize, n: u64) -> Result<(), String> {
        let pso = exec.pipeline(kernel).ok_or(format!("{kernel} missing"))?;
        let cb = exec.queue().new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pso);
        for (i, b) in bufs.iter().enumerate() { enc.set_buffer(i as u64, Some(*b), 0); }
        enc.set_bytes(bufs.len() as u64, plen as u64, params);
        enc.dispatch_thread_groups(MTLSize::new((n + 255) / 256, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
        cb.commit();
        Ok(())
    }

    fn dequant(&self, exec: &ObsidianExecutor, w: &Buffer, s: &Buffer, b: &Buffer, x: &Buffer,
               in_f: u32, out_f: u32) -> Result<Buffer, String> {
        let out = self.buf(exec, out_f as usize * 2);
        let p = DequantParams { in_features: in_f, out_features: out_f, group_size: 64 };
        self.disp(exec, "fused_dequant_matvec_4bit", &[w, s, b, x, &out],
                  &p as *const _ as *const c_void, std::mem::size_of::<DequantParams>(), out_f as u64)?;
        Ok(out)
    }
    fn dequant_named(&self, exec: &ObsidianExecutor, name: &str, x: &Buffer, in_f: u32, out_f: u32) -> Result<Buffer, String> {
        let w = self.weights.get(&format!("{name}.weight")).ok_or(format!("missing {name}.weight"))?.clone();
        let s = self.weights.get(&format!("{name}.scales")).ok_or(format!("missing {name}.scales"))?.clone();
        let b = self.weights.get(&format!("{name}.biases")).ok_or(format!("missing {name}.biases"))?.clone();
        self.dequant(exec, &w, &s, &b, x, in_f, out_f)
    }
    fn rmsnorm(&self, exec: &ObsidianExecutor, x: &Buffer, w: &Buffer, dim: u32, rows: u32) -> Result<Buffer, String> {
        let out = self.buf(exec, (dim * rows) as usize * 2);
        let p = RmsParams { dim, eps: self.cfg.rms_norm_eps };
        let pso = exec.pipeline("rmsnorm").ok_or("rmsnorm")?;
        let cb = exec.queue().new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pso);
        for (i, b) in [x, w, &out].iter().enumerate() { enc.set_buffer(i as u64, Some(*b), 0); }
        enc.set_bytes(3, std::mem::size_of::<RmsParams>() as u64, &p as *const _ as *const c_void);
        enc.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
        cb.commit();
        Ok(out)
    }
    fn rope(&self, exec: &ObsidianExecutor, buf: &Buffer, n_heads: u32, head_dim: u32, pos: u32) -> Result<(), String> {
        let p = RopeParams { n_heads, head_dim, pos, theta: self.cfg.rope_theta };
        self.disp(exec, "rope_apply", &[buf], &p as *const _ as *const c_void,
                  std::mem::size_of::<RopeParams>(), (n_heads * head_dim / 2) as u64)
    }
    fn add(&self, exec: &ObsidianExecutor, a: &Buffer, b: &Buffer, n: u32) -> Result<Buffer, String> {
        let out = self.buf(exec, n as usize * 2);
        let p = N { n };
        self.disp(exec, "ew_add", &[a, b, &out], &p as *const _ as *const c_void, 4, n as u64)?;
        Ok(out)
    }
    fn swiglu(&self, exec: &ObsidianExecutor, g: &Buffer, u: &Buffer, n: u32) -> Result<Buffer, String> {
        let out = self.buf(exec, n as usize * 2);
        let p = N { n };
        self.disp(exec, "ew_swiglu", &[g, u, &out], &p as *const _ as *const c_void, 4, n as u64)?;
        Ok(out)
    }
    fn scaled_add(&self, exec: &ObsidianExecutor, acc: &Buffer, y: &Buffer, scale: f32, n: u32) -> Result<(), String> {
        let p = ScaledN { n, scale };
        self.disp(exec, "ew_scaled_add", &[acc, y], &p as *const _ as *const c_void,
                  std::mem::size_of::<ScaledN>(), n as u64)
    }
    fn kv_append_gpu(&self, exec: &ObsidianExecutor, src: &Buffer, dst: &Buffer, n_kv: u32, head_dim: u32, max_len: u32, pos: u32) -> Result<(), String> {
        let p = KvP { n_kv, head_dim, max_len, pos };
        self.disp(exec, "kv_append", &[src, dst], &p as *const _ as *const c_void,
                  std::mem::size_of::<KvP>(), (n_kv * head_dim) as u64)
    }
    fn gate_gpu(&self, exec: &ObsidianExecutor, pre: &str, x: &Buffer) -> Result<Buffer, String> {
        let h = self.cfg.hidden_size as u32;
        let ne = self.cfg.num_experts as u32;
        let w = self.weights.get(&format!("{pre}mlp.gate.weight")).ok_or("gate.w")?.clone();
        let s = self.weights.get(&format!("{pre}mlp.gate.scales")).ok_or("gate.s")?.clone();
        let b = self.weights.get(&format!("{pre}mlp.gate.biases")).ok_or("gate.b")?.clone();
        let out = self.buf(exec, ne as usize * 2);
        let p = DequantParams { in_features: h, out_features: ne, group_size: 64 };
        self.disp(exec, "dequant_matvec_8bit", &[&w, &s, &b, x, &out],
                  &p as *const _ as *const c_void, std::mem::size_of::<DequantParams>(), ne as u64)?;
        Ok(out)
    }
    fn forward_layer(&mut self, exec: &ObsidianExecutor, il: usize, x: &Buffer) -> Result<Buffer, String> {
        let pre = format!("model.layers.{il}.");
        let (h, hd, nq, nkv) = (self.cfg.hidden_size as u32, self.cfg.head_dim() as u32,
                                self.cfg.num_attention_heads as u32, self.cfg.num_key_value_heads as u32);
        let moe_inter = self.cfg.moe_intermediate_size as u32;

        let xn = self.rmsnorm(exec, x, &self.weights.get(&format!("{pre}input_layernorm.weight")).ok_or("in_ln")?.clone(), h, 1)?;
        let q = self.dequant_named(exec, &format!("{pre}self_attn.q_proj"), &xn, h, nq * hd)?;
        let k = self.dequant_named(exec, &format!("{pre}self_attn.k_proj"), &xn, h, nkv * hd)?;
        let v = self.dequant_named(exec, &format!("{pre}self_attn.v_proj"), &xn, h, nkv * hd)?;
        let q = self.rmsnorm(exec, &q, &self.weights.get(&format!("{pre}self_attn.q_norm.weight")).ok_or("qn")?.clone(), hd, nq)?;
        let k = self.rmsnorm(exec, &k, &self.weights.get(&format!("{pre}self_attn.k_norm.weight")).ok_or("kn")?.clone(), hd, nkv)?;
        let pos = self.caches[il].seq_len as u32;
        self.rope(exec, &q, nq, hd, pos)?;
        self.rope(exec, &k, nkv, hd, pos)?;
        let ml = self.caches[il].max_len as u32;
        self.kv_append_gpu(exec, &k, &self.caches[il].k.clone(), nkv, hd, ml, pos)?;
        self.kv_append_gpu(exec, &v, &self.caches[il].v.clone(), nkv, hd, ml, pos)?;
        self.caches[il].seq_len += 1;
        let attn = self.attn.sdpa(exec, &q, &self.caches[il])?;
        let o = self.dequant_named(exec, &format!("{pre}self_attn.o_proj"), &attn, nq * hd, h)?;
        let x1 = self.add(exec, x, &o, h)?;                      // residual (GPU)

        let x1n = self.rmsnorm(exec, &x1, &self.weights.get(&format!("{pre}post_attention_layernorm.weight")).ok_or("post_ln")?.clone(), h, 1)?;
        let gate = self.gate_gpu(exec, &pre, &x1n)?;
        let (ids, wts) = self.router.route(exec, &gate)?;        // <-- only CPU sync/layer (commit+wait)
        let paged = crate::disk_io::read_experts_parallel(&self.experts, il, &ids);
        let mut by_e: HashMap<u32, HashMap<String, Vec<u8>>> = HashMap::new();
        for RawTensor { eid, key, bytes } in paged { by_e.entry(eid).or_default().insert(key, bytes); }

        const KEYS: [&str; 9] = ["gate_proj.weight", "gate_proj.scales", "gate_proj.biases",
                                 "up_proj.weight", "up_proj.scales", "up_proj.biases",
                                 "down_proj.weight", "down_proj.scales", "down_proj.biases"];
        for (rank, &e) in ids.iter().enumerate() {
            let t = by_e.get(&e).ok_or(format!("expert {e} not paged"))?;
            let b = rank * 9;
            for (k, key) in KEYS.iter().enumerate() { self.copy_into(&self.expert_pool[b + k], &t[*key]); }
            let p = &self.expert_pool;
            let g = self.dequant(exec, &p[b], &p[b + 1], &p[b + 2], &x1n, h, moe_inter)?;
            let u = self.dequant(exec, &p[b + 3], &p[b + 4], &p[b + 5], &x1n, h, moe_inter)?;
            let sg = self.swiglu(exec, &g, &u, moe_inter)?;
            let y = self.dequant(exec, &p[b + 6], &p[b + 7], &p[b + 8], &sg, moe_inter, h)?;
            self.scaled_add(exec, &x1, &y, wts[rank], h)?;       // x1 += w*y (GPU)
        }
        Ok(x1)
    }

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

    pub fn reset_cache(&mut self) {
        for c in &mut self.caches { c.seq_len = 0; }
        self.last_seq.clear();
    }

    /// Run one agentic turn with cross-request PREFIX REUSE.
    /// The KV cache persists across calls; we reuse the K/V of the longest
    /// common prefix between this prompt and what's already cached (exact:
    /// position i's K/V depends only on tokens 0..=i, so an identical prefix
    /// yields identical K/V), and only prefill the divergent suffix. For
    /// agentic loops (stable system+tools+history prefix) this turns an
    /// O(full-context) prefill into O(new-tokens) — the 60x lever.
    pub fn run_turn(&mut self, exec: &ObsidianExecutor, prompt: &[u32],
                    max_new: usize, eos: u32) -> Result<Vec<u32>, String> {
        if prompt.is_empty() { return Ok(Vec::new()); }
        let cap = self.caches[0].max_len;
        // If the prompt alone would overflow the cache, keep only its tail
        // (lose prefix reuse for the dropped head) to stay correct/bounded.
        let prompt: &[u32] = if prompt.len() + max_new > cap {
            let keep = cap.saturating_sub(max_new + 1).min(prompt.len());
            &prompt[prompt.len() - keep..]
        } else { prompt };

        // longest common prefix with the currently-cached sequence
        let mut lcp = 0usize;
        let maxcmp = self.last_seq.len().min(prompt.len());
        while lcp < maxcmp && self.last_seq[lcp] == prompt[lcp] { lcp += 1; }
        // leave at least the final prompt token so we get next-token logits
        if lcp >= prompt.len() { lcp = prompt.len() - 1; }

        for c in &mut self.caches { c.seq_len = lcp; }
        let reused = lcp;

        let t0 = std::time::Instant::now();
        let suffix: Vec<u32> = prompt[lcp..].to_vec();
        // Big suffix -> BATCHED prefill (layer-major, ~few syncs). Small suffix
        // or any failure -> per-token path (regression anchor / fallback).
        let mut next: u32;
        let use_batch = suffix.len() >= 24 && std::env::var("OBSIDIAN_NO_BATCH").is_err();
        if use_batch {
            match self.prefill_batched(exec, &suffix) {
                Ok(n) => { next = n; }
                Err(e) => {
                    log::warn!("[prefix] batched prefill failed ({e}); per-token fallback");
                    for c in &mut self.caches { c.seq_len = lcp; }
                    for &tk in &suffix[..suffix.len() - 1] { self.fill_kv(exec, tk)?; }
                    next = self.run_one_token(exec, *suffix.last().unwrap())?;
                }
            }
        } else {
            for &tk in &suffix[..suffix.len() - 1] { self.fill_kv(exec, tk)?; }
            next = self.run_one_token(exec, *suffix.last().unwrap())?;
        }
        let pf_ms = t0.elapsed().as_millis();

        let mut gen = vec![next];
        for _ in 0..max_new.saturating_sub(1) {
            if next == eos { break; }
            next = self.run_one_token(exec, next)?;
            gen.push(next);
        }

        self.last_seq.clear();
        self.last_seq.extend_from_slice(prompt);
        self.last_seq.extend_from_slice(&gen);

        self.last_reused = reused;
        self.last_prefilled = prompt.len() - reused;
        self.last_prefill_ms = pf_ms;
        self.turns += 1;

        log::info!("[prefix] reused {reused} tok, prefilled {} tok in {pf_ms}ms (~{}s saved)",
                   prompt.len() - reused, reused as f32 * 0.6);
        println!("[obsidian] turn: reused {reused} cached tok, prefilled {} new ({pf_ms}ms), gen {}",
                 prompt.len() - reused, gen.len());
        Ok(gen)
    }

    /// Prefill-only forward: advance the KV cache for `tok` WITHOUT computing
    /// the lm_head logits / argmax / CPU readback. During prefill we discard
    /// all but the final token's logits, so skipping the 2048->151936 lm_head
    /// matvec + vocab readback + argmax for every prefill token is a big, exact
    /// saving. Same-queue FIFO ordering keeps KV writes correct for the next
    /// token's attention (no drain needed between prefill tokens).
    pub fn fill_kv(&mut self, exec: &ObsidianExecutor, tok: u32) -> Result<(), String> {
        let mut x = self.embed(exec, tok);
        for il in 0..self.cfg.num_hidden_layers {
            x = self.forward_layer(exec, il, &x)?;
        }
        let _ = x;
        Ok(())
    }

    pub fn run_one_token(&mut self, exec: &ObsidianExecutor, tok: u32) -> Result<u32, String> {
        let mut x = self.embed(exec, tok);
        for il in 0..self.cfg.num_hidden_layers {
            x = self.forward_layer(exec, il, &x)?;
        }
        let h = self.cfg.hidden_size as u32;
        let nw = self.weights.get("model.norm.weight").ok_or("model.norm")?.clone();
        let xn = self.rmsnorm(exec, &x, &nw, h, 1)?;
        let logits = self.dequant_named(exec, "lm_head", &xn, h, self.cfg.vocab_size as u32)?;
        Self::drain(exec);                                       // final CPU sync
        let lv = Self::read_f16(&logits, self.cfg.vocab_size);
        let mut best = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for (i, &l) in lv.iter().enumerate() { if l > bv { bv = l; best = i; } }
        Ok(best as u32)
    }
}
