//! RPC protocol types for the tokimo AI worker sidecar.
//!
//! This module is shared between the worker binary (out-of-process host) and
//! the in-proc client + supervisor used by `tokimo-server`.
//!
//! Wire types here **mirror** tokimo-media-intelligence' public types but add `serde` derive so they
//! can travel over UDS / HTTP. Conversion helpers live in the worker binary.

#![allow(clippy::match_same_arms)] // read-loop arms are logically distinct (EOF vs error)

pub mod error;
pub mod frame;
pub mod transport;
pub mod types;

pub use error::{RpcError, RpcResult};
pub use types::*;

/// Single source of truth for RPC route paths. Both transports use the same strings.
pub mod routes {
    pub const PING: &str = "/v1/ping";
    pub const INFO: &str = "/v1/info";
    pub const SHUTDOWN: &str = "/v1/shutdown";
    pub const STATUS: &str = "/v1/models/status";
    pub const LIST_OCR_MODELS: &str = "/v1/models/list_ocr";
    pub const STT_MODELS_STATUS: &str = "/v1/models/stt_status";
    pub const ENSURE_CATEGORY: &str = "/v1/models/ensure_category"; // streaming progress
    pub const DOWNLOAD_STT: &str = "/v1/models/download_stt"; // streaming progress

    // --- Unified model catalog (settings UI) ---
    pub const CATALOG: &str = "/v1/catalog";
    pub const MODEL_DOWNLOAD: &str = "/v1/catalog/download"; // streaming progress by model_id
    pub const MODEL_UNLOAD: &str = "/v1/catalog/unload"; // unary

    pub const OCR: &str = "/v1/ocr";
    pub const OCR_HYBRID: &str = "/v1/ocr_hybrid";

    pub const CLIP_IMAGE: &str = "/v1/clip/image";
    pub const CLIP_TEXT: &str = "/v1/clip/text";
    pub const CLIP_CLASSIFY: &str = "/v1/clip/classify";

    pub const FACE_DETECT: &str = "/v1/face/detect";

    pub const STT_TRANSCRIBE: &str = "/v1/stt/transcribe";
    pub const STT_TRANSCRIBE_PCM: &str = "/v1/stt/transcribe_pcm";
    pub const STT_STREAM: &str = "/v1/stt/stream"; // bidirectional

    /// Terminate in-flight ORT inference(s) registered under the given id.
    pub const CANCEL: &str = "/v1/cancel";
}
