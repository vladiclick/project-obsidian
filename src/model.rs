// model.rs — Qwen3-MoE config (parsed from the HF config.json).
// Phase 2 step 1: just the config; weight loading + forward come next.

use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct Qwen3MoeConfig {
    pub num_hidden_layers: usize,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub vocab_size: usize,
    #[serde(default)]
    pub norm_topk_prob: bool,
}

impl Qwen3MoeConfig {
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let s = std::fs::read_to_string(path)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        serde_json::from_str(&s).map_err(|e| format!("parse config: {e}"))
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
}
