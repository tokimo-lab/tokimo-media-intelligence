# tokimo-media-intelligence

> [Tokimo](https://github.com/tokimo-lab/tokimo) Web 桌面 OS 的端侧感知套件。
> OCR、CLIP 图文向量、人脸检测与识别、语音转文字 —— 基于 ONNX Runtime + sherpa-onnx，既可作为库 crate 嵌入，也可作为独立的 sidecar 进程。

[![license](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![rust](https://img.shields.io/badge/rust-edition%202024-orange.svg)](./Cargo.toml)

[English](./README.md) · 简体中文

---

## 为什么又造一个轮子？

绝大多数 Rust 机器学习 crate 只聚焦一个任务（OCR、人脸、STT……），解决完就停下。Tokimo 需要把它们**同时塞进一个进程**，并且要求它们在空闲时**自觉从内存里滚出去** —— 一个长时间运行的桌面 server，不可能永久常驻 2 GB 的 ONNX session。

`tokimo-media-intelligence` 就是为此而生：

- **一个 crate，多个模型。** OCR（PaddleOCR / PP-OCRv5 / RapidOCR）、CLIP（OpenAI ViT-B/32 与 ViT-L/14、M-CLIP 多语言版本）、人脸检测 + 向量抽取、流式 STT（sherpa-onnx Zipformer / Whisper）。
- **懒加载 + 自动回收。** 模型首次调用时加载，空闲 3 分钟后自动从内存卸载。不用它，GPU 显存立刻还回去。
- **支持进程内使用，也支持 sidecar。** 既可以当普通 Rust 库用，也可以拉起内置的 `tokimo-media-intelligence-worker` 二进制，通过 Unix domain socket 或 HTTP 通信。worker 空闲时会自己退出 —— 当你希望操作系统回收整个进程的 RSS 时，这点特别有用。
- **统一的执行后端抽象。** CUDA / ROCm / CoreML / DirectML / CPU 启动时自动选择，带优雅降级。

## 能力一览

| 能力 | 后端 | 说明 |
|---|---|---|
| 文本检测 + 识别 | PaddleOCR mobile v4 · PP-OCRv5 · RapidOCR | 中英文开箱即用，带版面感知的行合并 |
| 图文向量 | OpenAI CLIP ViT-B/32 · ViT-L/14 · 多语言 M-CLIP | 用于以图搜图、零样本打标 |
| 人脸检测 + ArcFace 向量 | RetinaFace + ArcFace ONNX | 512 维向量，可直接送入 FAISS / pgvector |
| 流式语音转文字 | sherpa-onnx Zipformer · Whisper | 输入 16 kHz PCM，输出带时间戳的 token |
| 模型管理 | 内置 | 下载、SHA256 校验、缓存到 `$HOME/.cache/tokimo` |

## 架构

```
┌──────────────────────────────────────────────────────────────┐
│  你的 Rust 程序                                              │
│                                                              │
│   ┌──────────────────────┐       ┌──────────────────────┐    │
│   │  作为库直接使用      │       │  作为 worker 客户端  │    │
│   │  tokimo_media_intelligence   │       │  MediaIntelligenceWorkerClient      │    │
│   └──────────┬───────────┘       └──────────┬───────────┘    │
└──────────────┼──────────────────────────────┼────────────────┘
               │                              │
               │                              │  UDS / HTTP 帧
               ▼                              ▼
     ┌──────────────────┐          ┌─────────────────────────┐
     │  进程内          │          │ tokimo-media-intelligence-      │
     │  ONNX Runtime    │          │ worker （独立进程）     │
     │  session         │          │ 空闲时自动退出          │
     └──────────────────┘          └─────────────────────────┘
```

worker 使用简单的长度前缀 RPC 协议（见 `src/worker/protocol/`），传输层支持 Unix domain socket（本地，大 payload 零拷贝）或 HTTP 帧（远程、对 Docker 友好）。两种传输底下共用同一个 `MediaIntelligenceWorkerClient`，所以开发时可以先跑进程内推理，之后只要切一个配置开关就能把模型挪到独立容器里。

## 快速上手

### 作为库使用

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
    // 第二个参数选择 OCR 模型；传 `None` 表示使用默认模型。
    let items = svc.ocr(&png, None).await.map_err(anyhow::Error::msg)?;
    for it in items {
        println!("{:.2}  {:>3}x{:<3} @ ({:.0},{:.0})  {}",
                 it.score, it.w as i32, it.h as i32, it.x, it.y, it.text);
    }
    Ok(())
}
```

### 作为 sidecar 使用

```bash
# 只编译一次
cargo build --release --bin tokimo-media-intelligence-worker

# 空闲 5 分钟后自动退出
./target/release/tokimo-media-intelligence-worker \
    --socket /tmp/tokimo-media-intelligence.sock \
    --idle-exit-secs 300
```

server 侧：

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

搭配 `worker::client::supervisor::Supervisor` 使用时，worker 退出后在下次调用时会被自动重启。

## 运行时要求

- **Rust**：edition 2024（stable 1.85+）
- **ONNX Runtime**：动态加载。Linux 上设置 `ORT_DYLIB_PATH=/path/to/libonnxruntime.so`，或放到 `LD_LIBRARY_PATH` 里。macOS / Windows 同样识别 `ORT_DYLIB_PATH`，或走系统库搜索路径。
- **sherpa-onnx**：构建时由 `sherpa-onnx` crate 自动下载预编译库。
- **GPU（可选）**：
  - CUDA 12 + cuDNN 9（Linux / Windows）
  - ROCm，需要 `/dev/kfd`（Linux）
  - CoreML（macOS 10.15+，自动启用）
  - DirectML（Windows 10+，自动启用）

## 代码结构

```
src/
  lib.rs               — MediaIntelligenceService 门面 + EP 选择 + 懒卸载
  ocr*.rs              — Paddle / RapidOCR 检测器、识别器、版面合并
  clip.rs              — ViT 图像编码器 + 文本编码器 + 余弦检索
  face.rs              — RetinaFace + ArcFace 流水线
  stt.rs               — sherpa-onnx 流式封装
  models.rs            — 模型下载、校验、缓存清单
  worker/
    protocol/          — 长度前缀 msgpack RPC（UDS + HTTP 传输）
    client/            — MediaIntelligenceWorkerClient + supervisor（自动重启）
  bin/
    tokimo-media-intelligence-worker.rs — 独立 sidecar 二进制
config/                 — PaddleOCR 字符字典 + CLIP 词表
```

## 状态

本 crate 从 Tokimo 单体仓库中拆分而来，完整历史被保留（81 个 commit 一路追溯到 `rust-ai` → `rust-ocr` → `rust-models` → `tokimo-media-intelligence` 的改名链）。公共 API 仍会随着 Tokimo 本身演进；在 `1.0` 之前，只要主仓需要，我们就会切破坏性版本。

如果你只是想找个**今天就能直接用的稳定 OCR / CLIP / 人脸 / STT Rust crate**，这可能不是首选 —— 你得 pin commit。但如果你在做类似的项目，想参考一个真实在用的"多模型 ONNX 负载 + 正经生命周期管理"实现，那这里值得一看。

## 贡献

欢迎提 Issue / PR，不过请注意：

- 主要消费者是 [tokimo-lab/tokimo](https://github.com/tokimo-lab/tokimo)，对 Tokimo 没意义的破坏性改动通常不会接受。
- 提交前请先跑 `cargo clippy --all-targets -- -D warnings` 和 `cargo fmt`。
- CI 里无法方便地覆盖 CUDA / ROCm 的 GPU 路径，请在 PR 描述中说明自己测了哪个 EP。

## 协议

[MIT](./LICENSE) © Tokimo contributors.

`config/` 下的词典文件来自 PaddleOCR 项目，以 [Apache License 2.0](https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE) 重新分发。
