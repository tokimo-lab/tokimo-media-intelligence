/// AI service configuration.
///
/// Constructed by the host application (tokimo-server) and passed to `MediaIntelligenceService::new`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccelerationProfile {
    #[default]
    Balanced,
    LowVram,
}

impl AccelerationProfile {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Balanced => "balanced",
            Self::LowVram => "low_vram",
        }
    }

    #[must_use]
    pub fn parse(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "low_vram" | "low-vram" | "lowvram" | "low" => Self::LowVram,
            _ => Self::Balanced,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MediaIntelligenceConfig {
    pub models_dir: String,
    pub disable_hardware_acceleration: bool,
    pub acceleration_profile: AccelerationProfile,
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

pub fn acceleration_profile_from_env() -> AccelerationProfile {
    std::env::var("TOKIMO_MEDIA_INTELLIGENCE_ACCEL_PROFILE")
        .ok()
        .map_or(AccelerationProfile::Balanced, |v| AccelerationProfile::parse(&v))
}

impl Default for MediaIntelligenceConfig {
    fn default() -> Self {
        let data_local_path = data_local_path();
        let acceleration_profile = acceleration_profile_from_env();
        Self {
            models_dir: format!("{data_local_path}/media-intelligence"),
            disable_hardware_acceleration: hardware_acceleration_disabled_by_env(),
            acceleration_profile,
            enable_ocr: true,
            enable_clip: true,
            enable_face: true,
            enable_stt: true,
            python_sidecar_dir: None,
            ocr_det_max_side: None,
        }
    }
}
