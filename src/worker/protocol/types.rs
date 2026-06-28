//! Wire types — mirror tokimo-media-intelligence' public types with serde + pure-data shape.
//!
//! Conversion back to tokimo-media-intelligence types happens inside the worker binary.

use serde::{Deserialize, Serialize};

// ---------- acceleration / status ----------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccelProvider {
    Cuda,
    Rocm,
    CoreMl,
    DirectMl,
    Cpu,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccelerationProfile {
    #[default]
    Balanced,
    LowVram,
}

impl AccelerationProfile {
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Balanced => "balanced",
            Self::LowVram => "low_vram",
        }
    }
}

impl AccelProvider {
    /// Short human-readable name (matches the strings previously returned by
    /// `tokimo_media_intelligence::config::AccelProvider::name()`).
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Cuda => "CUDA",
            Self::Rocm => "ROCm",
            Self::CoreMl => "CoreML",
            Self::DirectMl => "DirectML",
            Self::Cpu => "CPU",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediaIntelligenceStatus {
    pub accel_provider: AccelProvider,
    pub ocr_loaded: bool,
    pub clip_loaded: bool,
    pub face_loaded: bool,
    pub stt_loaded: bool,
}

/// Static + readiness info snapshot. Returned from `/v1/info`; cached by the
/// client and used to back the sync `is_*_enabled` / `*_ready` / `config()`
/// APIs that mirror `tokimo_media_intelligence::MediaIntelligenceService`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerInfo {
    pub accel_provider: AccelProvider,
    pub acceleration_profile: AccelerationProfile,
    pub models_dir: String,
    pub ocr_det_max_side: Option<u32>,
    pub ocr_enabled: bool,
    pub clip_enabled: bool,
    pub face_enabled: bool,
    pub stt_enabled: bool,
    pub models_ready: bool,
    pub ocr_models_ready: bool,
    pub ocr_server_models_ready: bool,
    pub ocr_mobile_models_ready: bool,
    pub clip_models_ready: bool,
    pub face_models_ready: bool,
    pub stt_model_ready: bool,
    pub streaming_stt_model_ready: bool,
    pub ocr_loaded: bool,
    pub clip_loaded: bool,
    pub face_loaded: bool,
    pub stt_loaded: bool,
}

// ---------- OCR ----------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrItem {
    pub text: String,
    pub score: f32,
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub angle: f32,
    pub corners: Option<[(f32, f32); 4]>,
    pub paragraph_id: u32,
    pub char_positions: Option<Vec<(f32, f32)>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrModelInfo {
    pub id: String,
    pub display_name: String,
    pub loaded: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrRequest {
    /// Raw image bytes (any format supported by `image` crate).
    #[serde(with = "serde_bytes")]
    pub image: Vec<u8>,
    /// Model id override (None = use configured default).
    pub model: Option<String>,
    /// Optional cancellation id — when set, ORT `run_async` sites register
    /// their `RunOptions` under this id so the server can terminate in-flight
    /// inference via the `/v1/cancel` route.
    #[serde(default)]
    pub request_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrHybridRequest {
    #[serde(with = "serde_bytes")]
    pub image: Vec<u8>,
    pub det_model: Option<String>,
    pub vlm_model: Option<String>,
    #[serde(default)]
    pub request_id: Option<String>,
}

// serde_bytes crate provides #[serde(with = "serde_bytes")] for Vec<u8>.

// ---------- CLIP ----------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipImageRequest {
    #[serde(with = "serde_bytes")]
    pub image: Vec<u8>,
    #[serde(default)]
    pub request_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipTextRequest {
    pub text: String,
    #[serde(default)]
    pub request_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClipClassifyRequest {
    pub vector: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagResult {
    pub category: String,
    pub icon: String,
    pub subcategory: String,
    pub score: f32,
}

// ---------- Face ----------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceRequest {
    #[serde(with = "serde_bytes")]
    pub image: Vec<u8>,
    #[serde(default)]
    pub request_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceDetection {
    pub embedding: Vec<f32>,
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
    pub confidence: f32,
}

// ---------- STT ----------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SttTranscribeRequest {
    #[serde(with = "serde_bytes")]
    pub wav: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SttTranscribeResponse {
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SttTranscribePcmRequest {
    pub samples: Vec<f32>,
    pub sample_rate: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SttModelStatus {
    pub id: String,
    pub name: String,
    pub ready: bool,
}

/// Client → worker frames on the streaming STT channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SttClientFrame {
    /// One chunk of 16 kHz mono PCM.
    Audio { pcm: Vec<i16> },
    /// Reset decoder state without closing the stream.
    Reset,
    /// Close the stream (half-close from client side).
    Close,
}

/// Worker → client frames on the streaming STT channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SttServerFrame {
    /// Partial hypothesis (interim).
    Partial { text: String },
    /// Final hypothesis (endpoint triggered).
    Final { text: String },
    /// Endpoint detected — client should start a new utterance.
    Endpoint,
    /// Fatal error on the stream — client should reconnect.
    Error { message: String },
    /// Worker is about to shut down; client must reconnect on next request.
    ShuttingDown,
}

// ---------- Model downloads (streaming progress) ----------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelCategory {
    OcrServer,
    OcrMobile,
    Clip,
    Face,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsureCategoryRequest {
    /// `None` = ensure ALL categories (equivalent to the old
    /// `ensure_models_with_progress` entry point).
    pub category: Option<ModelCategory>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadSttRequest {
    pub model_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ProgressFrame {
    Progress {
        file_name: String,
        status: String,
        percent: u32,
        downloaded_bytes: u64,
        total_bytes: u64,
    },
    Done,
    Error {
        message: String,
    },
}

// ---------- Model catalog (declarative metadata for settings UI) ----------

/// Ordered, localized list of model sections. Returned by `/v1/catalog`.
/// All user-visible strings (`title`, `description`, `name`, attribute labels)
/// are already localized per the `Accept-Language` header supplied by the caller.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCatalog {
    pub sections: Vec<CatalogSection>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogSection {
    /// Stable id (`ocr` / `clip` / `face` / `stt` / `sidecar`).
    pub id: String,
    pub title: String,
    pub description: String,
    /// Lucide icon key (e.g. `"scan-text"`, `"image"`, `"user-round"`).
    pub icon: String,
    pub models: Vec<CatalogModel>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CatalogModel {
    /// Globally unique id in `<section>.<slug>` form (e.g. `ocr.ppocrv5-mobile`).
    pub id: String,
    pub name: String,
    pub description: String,
    pub size_mb: Option<u64>,
    /// Display-only key/label/value tag list (speed, accuracy, …).
    pub attrs: Vec<CatalogAttr>,
    /// Capability tags (`"text"`, `"blocks"`, `"formula"`, …).
    pub capabilities: Vec<String>,
    /// `"rust-native"` | `"python-sidecar"`.
    pub provider: String,
    pub state: ModelState,
    pub actions: Vec<ModelAction>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogAttr {
    pub key: String,
    pub label: String,
    pub value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ModelState {
    NotDownloaded,
    #[serde(rename_all = "camelCase")]
    Downloading {
        percent: u32,
        downloaded: u64,
        total: u64,
    },
    Loading,
    Ready,
    Error {
        message: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelAction {
    Download,
    Unload,
    Remove,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogRequest {
    /// BCP-47 language tags in preference order. Empty = worker default.
    #[serde(default)]
    pub languages: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelActionRequest {
    pub model_id: String,
}

// ---------- simple response wrappers ----------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pong {
    pub version: String,
    pub accel_provider: AccelProvider,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShutdownResponse {
    pub ok: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VecResponse {
    pub data: Vec<f32>,
}

// ---------- pure helpers (no AI runtime needed) ----------

const NO_BLOCK_MODELS: &[&str] = &["got-ocr-2"];

/// Whether a given OCR model returns bounding-box block info.
#[must_use]
pub fn ocr_model_supports_blocks(model_name: &str) -> bool {
    !NO_BLOCK_MODELS.contains(&model_name)
}

/// Convert Int16 little-endian PCM bytes to normalized f32 samples.
#[must_use]
pub fn int16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            f32::from(sample) / 32768.0
        })
        .collect()
}

// ---------- Cancel ----------

/// Request to terminate all in-flight ORT inference associated with a given id.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelRequest {
    pub request_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancelResponse {
    /// True if at least one in-flight RunOptions was terminated.
    pub cancelled: bool,
}
