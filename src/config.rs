/// AI service configuration.
///
/// Constructed by the host application (tokimo-server) and passed to `MediaIntelligenceService::new`.
#[derive(Debug, Clone)]
pub struct MediaIntelligenceConfig {
    pub models_dir: String,
    pub disable_hardware_acceleration: bool,
    pub enable_ocr: bool,
    pub enable_clip: bool,
    pub enable_face: bool,
    pub enable_stt: bool,
    /// Optional override for the embedded Python OCR sidecar source directory.
    /// `None` resolves to `<worker-dir>/python` or `CARGO_MANIFEST_DIR/python`.
    pub python_sidecar_dir: Option<String>,
    /// Detection resolution limit — longest image side in pixels.
    /// `None` uses the built-in default (4096). Lower values speed up detection
    /// at the cost of missing small text.
    pub ocr_det_max_side: Option<u32>,
}

pub fn data_local_path() -> String {
    std::env::var("TOKIMO_DATA_LOCAL_PATH").unwrap_or_else(|_| "./.data".to_string())
}

pub fn hardware_acceleration_disabled_by_env() -> bool {
    std::env::var("TOKIMO_MEDIA_INTELLIGENCE_DISABLE_ACCEL")
        .is_ok_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
}

impl Default for MediaIntelligenceConfig {
    fn default() -> Self {
        let data_local_path = data_local_path();
        Self {
            models_dir: format!("{data_local_path}/media-intelligence"),
            disable_hardware_acceleration: hardware_acceleration_disabled_by_env(),
            enable_ocr: true,
            enable_clip: true,
            enable_face: true,
            enable_stt: true,
            python_sidecar_dir: None,
            ocr_det_max_side: None,
        }
    }
}
