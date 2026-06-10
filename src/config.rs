/// AI service configuration.
///
/// Constructed by the host application (tokimo-server) and passed to `AiService::new`.
#[derive(Debug, Clone)]
pub struct AiConfig {
    pub models_dir: String,
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

impl Default for AiConfig {
    fn default() -> Self {
        let data_local_path = std::env::var("DATA_LOCAL_PATH").unwrap_or_else(|_| "./.data".to_string());
        Self {
            models_dir: format!("{data_local_path}/perception"),
            enable_ocr: true,
            enable_clip: true,
            enable_face: true,
            enable_stt: true,
            python_sidecar_dir: None,
            ocr_det_max_side: None,
        }
    }
}
