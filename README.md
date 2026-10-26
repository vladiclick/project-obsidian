# project-obsidian

Rust + Metal inference engine that runs **Qwen3-30B-A3B** (4-bit mixture-of-experts) on a **16 GB Apple Silicon Mac** by streaming only the active experts off the SSD for each token.

## How it works

The 4-bit checkpoint is about 17 GB, which does not fit in 16 GB of unified memory. In a mixture-of-experts model each token only uses a few experts per layer, so Obsidian keeps the forward pass on the GPU and pages in just the experts the router picks, layer by layer. The experts are pre-split into one safetensors file per layer, indexed by byte offset, and read with parallel `pread()` into pooled Metal buffers.

## Features

- **3.71 tok/s** steady-state decode (~4.6 ms/layer warm) — about 12× the Python layer-streaming baseline.
- Checked against MLX: 4-bit dequantization is bit-exact with `mx.dequantize`, and the full forward pass produces the same tokens as the reference engine.
- Hand-written Metal kernels: 4/8-bit dequant mat-vec, grouped dequant GEMM for the expert FFN, RMSNorm, RoPE, GQA attention, MoE top-k routing, elementwise ops.
- Batched prefill (3.6× faster cold start).
- Real prompts: Hugging Face tokenizer + chat template.
- OpenAI-compatible `/v1/chat/completions` endpoint.

## Tech stack

Rust · Metal ([metal-rs](https://github.com/gfx-rs/metal-rs)) · safetensors + memmap2 · rayon · tokenizers · tiny_http · Python + MLX for the validation scripts

## Getting started

Requirements: an Apple Silicon Mac, Xcode Command Line Tools, stable Rust, and the [`mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit`](https://huggingface.co/mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit) checkpoint in the Hugging Face cache. Shaders compile at runtime, so no Metal toolchain is needed.

```bash
cargo build --release

# One-time: split the stacked experts into per-layer files (<model_dir>/splitted_moe/)
MODEL_DIR=$(echo ~/.cache/huggingface/hub/models--mlx-community--Qwen3-30B-A3B-Instruct-2507-4bit/snapshots/*)
python3 tools/split_experts.py "$MODEL_DIR"      # needs: pip install mlx safetensors
```

Run a prompt:

```bash
./target/release/obsidian --model qwen3-30b-a3b --config "$MODEL_DIR/config.json" \
  --run --prompt "Write a hello world in Rust" --max-tokens 48
```

Serve an OpenAI-compatible API:

```bash
./target/release/obsidian --model qwen3-30b-a3b --config "$MODEL_DIR/config.json" --serve --port 8000

curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"obsidian","messages":[{"role":"user","content":"Write a hello world in Rust"}],"max_tokens":48}'
```

Any OpenAI-compatible client works with `OPENAI_BASE_URL=http://localhost:8000/v1` and any non-empty API key (auth is ignored).

Other flags: `--expert-dir` (default `<model_dir>/splitted_moe`), `--max-tokens` (default 50), `--port` (default 8000).

Environment variables (all optional):

| Variable | Effect |
|---|---|
| `OBSIDIAN_MAX_LEN` | KV-cache context length in tokens (default 16384, about 1.6 GB) |
| `OBSIDIAN_NO_GEMM` | Set to use the per-token mat-vec expert path instead of the grouped GEMM |
| `OBSIDIAN_NO_BATCH` | Set to disable batched prefill |

## Validation scripts

- `validate_dequant.py` — replicates the shader's 4-bit unpacking in NumPy and compares it with `mx.dequantize` on a real weight.
- `reference_logits.py` — produces a greedy reference sequence with a layer-streamed Python MoE engine built on AirLLM (not included here; set `AIRLLM_PATH` to its `air_llm` directory) to compare against Obsidian's tokens.

## Status

v1.0 — works end to end on a 16 GB M4. Limits:

- Single-sequence greedy decode: no batching, no speculative decoding, no streaming responses yet.
- Decode is bound by expert-paging I/O (~6 ms/layer floor on this SSD), so more than ~5 tok/s is not reachable for this model on this hardware; 3.71 tok/s is close to that ceiling.

Notes from the first build attempt are in [docs/phase-1-build-log.md](docs/phase-1-build-log.md).

## License

[MIT](LICENSE)
