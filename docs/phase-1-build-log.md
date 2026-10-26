> Historical notes from the first build attempt (June 2026), kept for reference. The build issues listed here were resolved; see the README for current instructions.

# Project Obsidian — Build Status (Phase 1)

## Structure created
```
project_obsidian/
├── Cargo.toml                # deps (mlx-native UNVERIFIED), release profile, M4 notes
├── BUILD.md                  # this file
├── src/
│   ├── main.rs               # CLI stub (Phase 1 wiring only)
│   ├── executor.rs           # single-CommandBuffer graph (blueprint §1.2, UNVERIFIED)
│   └── disk_io.rs            # parallel pread() expert streaming (blueprint §2.2)
└── shaders/
    ├── fused_dequant_matvec_4bit.metal
    ├── moe_routing_topk.metal
    ├── fused_moe_combine.metal
    └── fused_residual_rms_norm.metal
```

## ⛔ Build is BLOCKED — `cargo build` cannot run yet

Three hard blockers on this machine:

1. **No Rust toolchain.** `cargo` and `rustc` are not installed.
   Fix: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`

2. **No Metal toolchain.** `xcrun metal` reports *"missing Metal Toolchain"*.
   The four `.metal` shaders cannot compile without it.
   Fix: `xcodebuild -downloadComponent MetalToolchain` (large Apple download;
   may require full Xcode, not just CLI tools).

3. **`mlx-native` crate UNVERIFIED.** crates.io API is blocked from this
   environment (HTTP 403), so the crate's existence, version (`0.6`), and API
   (`GraphExecutor`/`DispatchRecord`/`MlxBufferPool`/`KernelRegistry`) could not
   be confirmed. The blueprint's claim ("60-70 tok/s") and the very specific API
   are consistent with LLM-fabricated content. **`executor.rs` is built entirely
   on this API.** If the crate is fictional or different, `executor.rs` must be
   rewritten against `mlx-rs` (the real MLX Rust binding) or raw `metal`.

## Blueprint fixes already applied to Cargo.toml
- `edition = "2024"` + `rust-version = "1.82"` → **incompatible** (edition 2024
  needs rustc ≥ 1.85). Set to `edition = "2021"`.
- `[profile.release.build-overrides."*"] rustflags = [...]` → **invalid Cargo**
  (rustflags is not a profile key). Removed; documented `.cargo/config.toml`
  alternative.
- `target-cpu=apple-m4` → may be unknown to LLVM; suggested `apple-m2`.
- Code-level: blueprint's `fused_dequant_matvec_4bit.metal` matvec was wrong
  (used `input[0]` for every term); corrected to a real dot product.

## To proceed (in order)
1. Install Rust toolchain (rustup).
2. **Confirm `mlx-native` is a real crate** (provide crates.io/git source). If
   not, decide: rewrite against `mlx-rs`, or build the Metal pipeline directly
   on the `metal` crate (no mlx-native).
3. Install the Metal toolchain.
4. `cargo build --release` → iterate on real compiler errors.
