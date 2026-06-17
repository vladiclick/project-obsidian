# Project Obsidian

Bare-metal **Rust + Metal** MoE inference engine for Apple Silicon. Runs
**Qwen3-30B-A3B (4-bit MoE)** on a **16 GB M4** by streaming only the active
experts off SSD per token.

- **3.71 tok/s** decode steady-state (~4.6 ms/layer warm) — ~12× the Python
  layer-streaming baseline.
- **Correct**: dequant bit-exact vs `mx.dequantize`; full forward matches the
  reference MLX engine (same tokens).
- Hand-written `.metal` kernels (4/8-bit dequant-matvec, RMSNorm, RoPE, GQA
  SDPA, MoE top-k routing, elementwise). GPU-resident forward, expert buffer
  pooling, per-layer expert paging via `pread()`.

## Requirements
- Apple Silicon mac, macOS, Xcode CLI tools.
- Rust (stable): `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- A 4-bit MLX MoE checkpoint cached locally, e.g.
  `mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit`, pre-ingested into per-layer
  expert files (`splitted_moe/`). Metal shaders compile at runtime — no Metal CLI
  toolchain needed.

## Build
```bash
cargo build --release
```

## Run a prompt
```bash
CFG=~/.cache/huggingface/hub/models--mlx-community--Qwen3-30B-A3B-Instruct-2507-4bit/snapshots/*/config.json
./target/release/obsidian --run --config $CFG --prompt "Write a hello world in Rust" --max-tokens 48
```

## Serve (OpenAI-compatible HTTP API)
```bash
./target/release/obsidian --serve --config $CFG --port 8000
# -> [obsidian] API LIVE on http://0.0.0.0:8000/v1/chat/completions
```

Test it:
```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"obsidian","messages":[{"role":"user","content":"Write a hello world in Rust"}],"max_tokens":48}'
```

## Use from an OpenAI-compatible client (incl. Claude Code OpenAI mode)
```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=sk-obsidian-local   # any non-empty value; auth is ignored
```

## Notes / limits
- Single-sequence greedy decode (no batching, no speculative decoding, no
  streaming responses yet).
- `>5 tok/s` is not reachable for 30B-A3B on this SSD — bounded by expert paging
  I/O (~6 ms/layer floor). 3.71 tok/s is near that physical ceiling.
- Flags: `--port` (default 8000), `--max-tokens`, `--expert-dir`.
