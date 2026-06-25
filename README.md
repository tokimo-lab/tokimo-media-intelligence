# tokimo-media-intelligence

> On-device perception stack for the [Tokimo](https://github.com/tokimo-lab/tokimo) web desktop OS.
> OCR, CLIP image/text embedding, face detection & recognition, and speech-to-text — all via ONNX Runtime + sherpa-onnx, exposed as a library crate *and* a standalone sidecar worker.

[![license](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![rust](https://img.shields.io/badge/rust-edition%202024-orange.svg)](./Cargo.toml)

English · [简体中文](./README.zh-CN.md)

---

## Why another ML crate?

Most Rust ML crates pick one task (OCR, face, STT, …) and stop there. Tokimo needs all of them to coexist in a single process *and* needs them to get out of the way when idle — a long-running desktop server can't afford to hold 2 GB of ONNX sessions in memory forever.

`tokimo-media-intelligence` is the answer:

- **One crate, many models.** OCR (PaddleOCR / PP-OCRv5 / RapidOCR), CLIP (OpenAI ViT-B/32 & ViT-L/14, multilingual variants), face detection + embedding, and streaming STT (sherpa-onnx Zipformer / Whisper).
- **Lazy + self-evicting.** Models are loaded on first call and automatically dropped from memory after 3 minutes of inactivity. GPU VRAM comes back the moment you stop using it.
- **Runs in-process *or* as a sidecar.** Use it as a plain Rust library, or spawn the bundled `tokimo-media-intelligence-worker` binary and talk to it over a Unix domain socket / HTTP. The sidecar exits on its own when idle — perfect when you want the OS to reclaim the whole process's RSS.
- **One execution-provider abstraction.** CUDA, ROCm, CoreML, DirectML, or CPU — picked automatically at startup with graceful fallback.

## Features

| Capability | Backend | Notes |
|---|---|---|
| Text detection + recognition | PaddleOCR mobile v4 · PP-OCRv5 · RapidOCR | Chinese + English out of the box, layout-aware line merging |
| Image-text embedding | OpenAI CLIP ViT-B/32 · ViT-L/14 · multilingual M-CLIP | For reverse image search & zero-shot tagging |
| Face detection + ArcFace embedding | RetinaFace + ArcFace ONNX | 512-d embeddings suitable for FAISS / pgvector |
| Streaming speech-to-text | sherpa-onnx Zipformer · Whisper | 16 kHz PCM in, timestamped tokens out |
| Model management | built-in | Download, verify (SHA256), cache under `$HOME/.cache/tokimo` |

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Your Rust program                                           │
│                                                              │
│   ┌──────────────────────┐       ┌──────────────────────┐    │
│   │  as a library        │       │  as a worker client  │    │
│   │  tokimo_media_intelligence   │       │  MediaIntelligenceWorkerClient      │    │
│   └──────────┬───────────┘       └──────────┬───────────┘    │
└──────────────┼──────────────────────────────┼────────────────┘
               │                              │
               │                              │  UDS / HTTP frames
               ▼                              ▼
     ┌──────────────────┐          ┌─────────────────────────┐
     │  in-process      │          │ tokimo-media-intelligence-      │
     │  ONNX Runtime    │          │ worker  (separate proc) │
     │  sessions        │          │  auto-exits when idle   │
     └──────────────────┘          └─────────────────────────┘
```

The worker speaks a simple length-prefixed RPC (`src/worker/protocol/`) using either Unix domain sockets (local, zero-copy for blobs) or framed HTTP (remote, Docker-friendly). Both transports expose the same `MediaIntelligenceWorkerClient` surface, so you can develop against in-process inference and flip a config switch to move the models into a separate container later.

## Quick start

### As a library

```toml
# Cargo.toml
[dependencies]
tokimo-media-intelligence = { git = "https://github.com/tokimo-lab/tokimo-media-intelligence" }
```

```rust
use tokimo_media_intelligence::{config::MediaIntelligenceConfig, MediaIntelligenceService};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let svc = MediaIntelligenceService::new(MediaIntelligenceConfig::default());
    let png = std::fs::read("invoice.png")?;
    // Second argument picks an OCR model; `None` uses the default.
    let items = svc.ocr(&png, None).await.map_err(anyhow::Error::msg)?;
    for it in items {
        println!("{:.2}  {:>3}x{:<3} @ ({:.0},{:.0})  {}",
                 it.score, it.w as i32, it.h as i32, it.x, it.y, it.text);
    }
    Ok(())
}
```

### As a sidecar worker

```bash
# Build once
cargo build --release --bin tokimo-media-intelligence-worker

# Run with auto-eviction after 5 minutes idle
./target/release/tokimo-media-intelligence-worker \
    --socket /tmp/tokimo-media-intelligence.sock \
    --idle-exit-secs 300
```

Then from your server:

```rust
use std::sync::Arc;
use tokimo_media_intelligence::worker::client::MediaIntelligenceWorkerClient;
use tokimo_media_intelligence::worker::protocol::transport::{AnyTransport, UdsTransport};

let transport = Arc::new(AnyTransport::Uds(
    UdsTransport::new("/tmp/tokimo-media-intelligence.sock"),
));
let client = MediaIntelligenceWorkerClient::new(transport);
let embedding = client.clip_image(jpeg_bytes).await?;
```

When you pair the client with the built-in supervisor
(`worker::client::supervisor::Supervisor`), exited workers are respawned
transparently on the next call.

## Runtime requirements

- **Rust**: edition 2024 (stable 1.85+)
- **ONNX Runtime**: loaded dynamically. On Linux set `ORT_DYLIB_PATH=/path/to/libonnxruntime.so`, or place it on `LD_LIBRARY_PATH`. macOS / Windows use the same environment variable or system library search.
- **sherpa-onnx**: prebuilt libraries are auto-downloaded by the `sherpa-onnx` crate during build.
- **GPU (optional)**:
  - CUDA 12 + cuDNN 9 (Linux / Windows)
  - ROCm with `/dev/kfd` (Linux)
  - CoreML (macOS 10.15+, automatic)
  - DirectML (Windows 10+, automatic)

## Project layout

```
src/
  lib.rs               — MediaIntelligenceService facade + EP selection + lazy unloading
  ocr*.rs              — Paddle / RapidOCR detector + recognizer + layout merge
  clip.rs              — ViT image encoder + text encoder + cosine search
  face.rs              — RetinaFace + ArcFace pipeline
  stt.rs               — sherpa-onnx streaming wrapper
  models.rs            — download / verify / cache manifest
  worker/
    protocol/          — length-prefixed msgpack RPC (UDS + HTTP transports)
    client/            — MediaIntelligenceWorkerClient + supervisor (auto-respawn)
  bin/
    tokimo-media-intelligence-worker.rs — standalone sidecar binary
config/                 — PaddleOCR character dict + CLIP vocab
```

## Status

This crate is extracted from the Tokimo monorepo with its full history preserved (81 commits going back to the `rust-ai` → `rust-ocr` → `rust-models` → `tokimo-media-intelligence` rename chain). The public API is still evolving as Tokimo itself evolves; until `1.0` we will cut breaking releases whenever the monorepo needs them.

If you just want a stable OCR / CLIP / face / STT Rust crate *today*, this probably isn't it — you'll be pinning commits. But if you're building something similar and want a reference for how to ship multi-model ONNX workloads with sane lifecycle management, dig in.

## Contributing

Issues and PRs are welcome. Please note:

- The primary consumer is [tokimo-lab/tokimo](https://github.com/tokimo-lab/tokimo) — breaking changes that don't make sense for Tokimo will probably be rejected.
- Run `cargo clippy --all-targets -- -D warnings` and `cargo fmt` before submitting.
- CUDA / ROCm GPU paths can't be easily tested in CI; include the EP you tested against in your PR description.

## License

[MIT](./LICENSE) © Tokimo contributors.

Bundled dictionaries under `config/` originate from the PaddleOCR project and are redistributed under the [Apache License 2.0](https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE).
