use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use base64::Engine;
use serde::{Deserialize, Serialize};
use tokio::sync::{OnceCell, RwLock};

use crate::ocr::OcrItem;
use crate::ocr_backend::{OcrBackend, PaddleOcrVariant};
use crate::sidecar::SidecarManager;

/// Known model identifiers.
pub const MODEL_PP_OCRV5_MOBILE: &str = "pp-ocrv5-mobile";
pub const MODEL_GOT_OCR_2: &str = "got-ocr-2";
pub const MODEL_RAPID_OCR_RUST: &str = "rapid-ocr-rust";

/// Default model when none specified.
pub const DEFAULT_MODEL: &str = MODEL_RAPID_OCR_RUST;

/// Models that do NOT provide bounding boxes (text-only output).
const NO_BLOCK_MODELS: &[&str] = &[MODEL_GOT_OCR_2];

/// Metadata about an available OCR model.
#[derive(Debug, Clone)]
pub struct OcrModelInfo {
    pub id: &'static str,
    pub display_name: &'static str,
    pub loaded: bool,
}

/// Per-backend lazy-init slot.
///
/// `OnceCell::get_or_try_init` is the standard Tokio "init once" primitive:
/// the first caller runs the init future, all concurrent callers wait for it,
/// no double-initialization possible. The outer `RwLock` is only needed for
/// eviction — a write lock replaces the cell with a fresh one to free memory.
struct BackendSlot {
    cell: RwLock<OnceCell<Arc<dyn OcrBackend>>>,
}

impl BackendSlot {
    fn new() -> Self {
        Self {
            cell: RwLock::new(OnceCell::new()),
        }
    }

    async fn get_or_init(
        &self,
        model_name: &str,
        models_dir: &str,
        det_max_side: Option<u32>,
    ) -> Result<Arc<dyn OcrBackend>, String> {
        let guard = self.cell.read().await;
        let name = model_name.to_string();
        let dir = models_dir.to_string();
        let backend = guard
            .get_or_try_init(|| async move {
                tokio::task::spawn_blocking(move || create_backend_sync(&name, &dir, det_max_side))
                    .await
                    .map_err(|e| format!("Backend init task panicked: {e}"))?
            })
            .await?;
        Ok(Arc::clone(backend))
    }

    async fn evict(&self, name: &str) -> bool {
        let mut guard = self.cell.write().await;
        let was_loaded = guard.get().is_some();
        if was_loaded {
            *guard = OnceCell::new();
            tracing::info!("Evicted OCR backend: {name}");
        }
        was_loaded
    }

    fn is_loaded(&self) -> bool {
        self.cell.try_read().is_ok_and(|g| g.get().is_some())
    }
}

/// Routes OCR requests to the appropriate backend. Backends are lazy-loaded
/// and can be evicted to free memory after idle timeout.
pub struct OcrManager {
    models_dir: String,
    backends: HashMap<&'static str, BackendSlot>,
    sidecar: Arc<SidecarManager>,
    last_use: AtomicU64,
    /// Detection resolution limit (longest side in pixels).
    det_max_side: Option<u32>,
}

impl OcrManager {
    pub fn new(models_dir: String, sidecar: Arc<SidecarManager>) -> Self {
        Self::with_options(models_dir, sidecar, None)
    }

    pub fn with_max_side(models_dir: String, sidecar: Arc<SidecarManager>, det_max_side: Option<u32>) -> Self {
        Self::with_options(models_dir, sidecar, det_max_side)
    }

    pub fn with_options(models_dir: String, sidecar: Arc<SidecarManager>, det_max_side: Option<u32>) -> Self {
        let mut backends: HashMap<&'static str, BackendSlot> = HashMap::new();
        backends.insert(MODEL_PP_OCRV5_MOBILE, BackendSlot::new());
        backends.insert(MODEL_RAPID_OCR_RUST, BackendSlot::new());
        // VLM models (GOT-OCR-2) are handled via HTTP sidecar, not backends
        Self {
            models_dir,
            backends,
            sidecar,
            last_use: AtomicU64::new(0),
            det_max_side,
        }
    }

    /// Run OCR with the given model. Lazy-initializes the backend on first call.
    /// For VLM sidecar models, makes an HTTP call to the Python sidecar.
    pub async fn ocr(&self, model_name: &str, image_bytes: &[u8]) -> Result<Vec<OcrItem>, String> {
        self.last_use.store(crate::epoch_secs(), Ordering::Relaxed);

        // VLM models: ensure sidecar is running, then call directly
        if model_name == MODEL_GOT_OCR_2 {
            let sidecar_url = self.sidecar.ensure_running().await?;
            return vlm_ocr_via_sidecar(&sidecar_url, model_name, image_bytes).await;
        }

        // Local Paddle models: decode image (fast CPU op) then run async ONNX inference.
        // No spawn_blocking needed — sessions use tokio::sync::Mutex + run_async.
        let bytes = image_bytes.to_vec();
        let backend = self.get_or_init_backend(model_name).await?;
        let img = image::load_from_memory(&bytes).map_err(|e| format!("Invalid image: {e}"))?;
        backend.recognize(&img).await
    }

    /// Hybrid OCR: use `det_model` for bounding-box detection, `vlm_model` for
    /// accurate text recognition, then merge via sidecar.
    /// Returns (`merged_items`, `optional_debug_info`).
    pub async fn ocr_hybrid(
        &self,
        det_model: &str,
        vlm_model: &str,
        image_bytes: &[u8],
    ) -> Result<(Vec<OcrItem>, Option<serde_json::Value>), String> {
        let sidecar_url = self.sidecar.ensure_running().await?;

        // 1. Run detection model for bounding boxes
        let det_items = self.ocr(det_model, image_bytes).await?;

        // 2. Send image + det_blocks to sidecar for VLM recognition + merge
        hybrid_ocr_via_sidecar(&sidecar_url, image_bytes, &det_items, vlm_model).await
    }

    /// Return the epoch timestamp of last use (0 if never used).
    pub fn last_use_epoch(&self) -> u64 {
        self.last_use.load(Ordering::Relaxed)
    }

    /// Evict all loaded backends to free memory.
    pub async fn evict_all(&self) {
        let mut evicted = 0usize;
        for (name, slot) in &self.backends {
            if slot.evict(name).await {
                evicted += 1;
            }
        }
        if evicted > 0 {
            #[cfg(target_os = "linux")]
            #[allow(unsafe_code)]
            unsafe {
                libc::malloc_trim(0);
            }
            tracing::info!("Evicted {evicted} OCR backend(s), called malloc_trim");
        }
    }

    /// Check whether a model provides bounding-box coordinates.
    pub fn model_supports_blocks(model_name: &str) -> bool {
        !NO_BLOCK_MODELS.contains(&model_name)
    }

    /// List all known models with their current status.
    pub fn available_models(&self) -> Vec<OcrModelInfo> {
        Self::known_models()
            .into_iter()
            .map(|mut info| {
                info.loaded = self.is_loaded(info.id);
                info
            })
            .collect()
    }

    /// Static list of all known OCR models (loaded flag is always false).
    /// Callable without an initialized manager instance.
    pub fn known_models() -> Vec<OcrModelInfo> {
        vec![
            OcrModelInfo {
                id: MODEL_RAPID_OCR_RUST,
                display_name: "RapidOCR",
                loaded: false,
            },
            OcrModelInfo {
                id: MODEL_PP_OCRV5_MOBILE,
                display_name: "PP-OCRv5 Mobile",
                loaded: false,
            },
            OcrModelInfo {
                id: MODEL_GOT_OCR_2,
                display_name: "GOT-OCR 2.0",
                loaded: false,
            },
        ]
    }

    fn is_loaded(&self, model: &str) -> bool {
        self.backends.get(model).is_some_and(BackendSlot::is_loaded)
    }

    /// Whether any OCR backend is currently loaded in memory.
    pub fn has_loaded_backends(&self) -> bool {
        self.backends.values().any(BackendSlot::is_loaded)
    }

    async fn get_or_init_backend(&self, model_name: &str) -> Result<Arc<dyn OcrBackend>, String> {
        let slot = self
            .backends
            .get(model_name)
            .ok_or_else(|| format!("Unknown OCR model: {model_name}"))?;
        slot.get_or_init(model_name, &self.models_dir, self.det_max_side).await
    }
}

/// Create an OCR backend synchronously (called from `spawn_blocking`).
fn create_backend_sync(
    model_name: &str,
    models_dir: &str,
    det_max_side: Option<u32>,
) -> Result<Arc<dyn OcrBackend>, String> {
    match model_name {
        MODEL_PP_OCRV5_MOBILE => {
            let svc = crate::ocr::OcrService::new_with_options(
                models_dir,
                PaddleOcrVariant::Mobile,
                crate::ocr::DetectionMode::Components,
                det_max_side,
            )?;
            Ok(Arc::new(svc))
        }
        MODEL_RAPID_OCR_RUST => {
            let svc = crate::ocr::OcrService::new_with_options(
                models_dir,
                PaddleOcrVariant::Server,
                crate::ocr::DetectionMode::Contours,
                det_max_side,
            )?;
            Ok(Arc::new(svc))
        }
        _ => Err(format!("Unknown OCR model: {model_name}")),
    }
}

// ── VLM sidecar HTTP call ─────────────────────────────────────────────────────

/// Response block from the Python sidecar `/ocr` endpoint.
#[derive(Debug, Deserialize)]
struct SidecarOcrBlock {
    text: String,
    x: Option<f64>,
    y: Option<f64>,
    w: Option<f64>,
    h: Option<f64>,
    score: f64,
    #[serde(default)]
    paragraph_id: u32,
}

/// Response from the Python sidecar `/ocr` endpoint.
#[derive(Debug, Deserialize)]
struct SidecarOcrResponse {
    #[allow(dead_code)]
    model: String,
    blocks: Vec<SidecarOcrBlock>,
    #[allow(dead_code)]
    processing_time_ms: f64,
    debug: Option<serde_json::Value>,
}

/// Call the Python OCR sidecar via HTTP to run VLM-based OCR.
async fn vlm_ocr_via_sidecar(sidecar_url: &str, model_name: &str, image_bytes: &[u8]) -> Result<Vec<OcrItem>, String> {
    let b64 = base64::engine::general_purpose::STANDARD.encode(image_bytes);
    let body = serde_json::json!({
        "model": model_name,
        "image": b64,
    });

    let client = reqwest::Client::new();
    let url = format!("{}/ocr", sidecar_url.trim_end_matches('/'));

    let resp = client
        .post(&url)
        .json(&body)
        .timeout(std::time::Duration::from_mins(2))
        .send()
        .await
        .map_err(|e| format!("Sidecar HTTP error: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("Sidecar returned {status}: {text}"));
    }

    let result: SidecarOcrResponse = resp
        .json()
        .await
        .map_err(|e| format!("Failed to parse sidecar response: {e}"))?;

    Ok(result
        .blocks
        .into_iter()
        .map(|b| OcrItem {
            text: b.text,
            score: b.score as f32,
            // Use -1.0 sentinel when sidecar returns null coordinates
            // (e.g. GOT-OCR which doesn't provide bounding boxes)
            x: b.x.map_or(-1.0, |v| v as f32),
            y: b.y.map_or(-1.0, |v| v as f32),
            w: b.w.map_or(-1.0, |v| v as f32),
            h: b.h.map_or(-1.0, |v| v as f32),
            angle: 0.0,
            corners: None,
            paragraph_id: b.paragraph_id,
            char_positions: None,
        })
        .collect())
}

// ── Hybrid OCR (PP-OCRv5 detection + GOT-OCR recognition) ────────────────────

/// Detection block sent to the sidecar for hybrid merge.
#[derive(Debug, Serialize)]
struct SidecarDetBlock {
    text: String,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    score: f32,
    paragraph_id: u32,
}

/// Call the sidecar `/ocr/hybrid` endpoint: send `det_blocks` + image,
/// sidecar runs the VLM model and merges text with detection coordinates.
async fn hybrid_ocr_via_sidecar(
    sidecar_url: &str,
    image_bytes: &[u8],
    det_items: &[OcrItem],
    vlm_model: &str,
) -> Result<(Vec<OcrItem>, Option<serde_json::Value>), String> {
    let b64 = base64::engine::general_purpose::STANDARD.encode(image_bytes);

    let det_blocks: Vec<SidecarDetBlock> = det_items
        .iter()
        .filter(|item| item.x >= 0.0 && item.y >= 0.0)
        .map(|item| SidecarDetBlock {
            text: item.text.clone(),
            x: item.x,
            y: item.y,
            w: item.w,
            h: item.h,
            score: item.score,
            paragraph_id: item.paragraph_id,
        })
        .collect();

    let body = serde_json::json!({
        "image": b64,
        "det_blocks": det_blocks,
        "vlm_model": vlm_model,
    });

    let client = reqwest::Client::new();
    let url = format!("{}/ocr/hybrid", sidecar_url.trim_end_matches('/'));

    let resp = client
        .post(&url)
        .json(&body)
        .timeout(std::time::Duration::from_mins(3))
        .send()
        .await
        .map_err(|e| format!("Sidecar hybrid HTTP error: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("Sidecar hybrid returned {status}: {text}"));
    }

    let result: SidecarOcrResponse = resp
        .json()
        .await
        .map_err(|e| format!("Failed to parse sidecar hybrid response: {e}"))?;

    let items = result
        .blocks
        .into_iter()
        .map(|b| OcrItem {
            text: b.text,
            score: b.score as f32,
            x: b.x.map_or(-1.0, |v| v as f32),
            y: b.y.map_or(-1.0, |v| v as f32),
            w: b.w.map_or(-1.0, |v| v as f32),
            h: b.h.map_or(-1.0, |v| v as f32),
            angle: 0.0,
            corners: None,
            paragraph_id: b.paragraph_id,
            char_positions: None,
        })
        .collect();

    Ok((items, result.debug))
}
