#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::needless_pass_by_value,
    clippy::too_many_lines,
    clippy::manual_filter
)]

//! Unified AI service: OCR + CLIP + Face recognition + STT via ONNX Runtime / sherpa-onnx.
//!
//! This crate is used as a library by `tokimo-server` — no HTTP server.
//! Models are lazy-loaded on first use and automatically evicted after 3 minutes
//! of inactivity to free memory.

pub mod cancel;
pub mod clip;
pub mod clip_categories;
pub mod config;
pub mod face;
pub mod models;

/// GPU acceleration execution provider selected at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccelProvider {
    /// NVIDIA CUDA — Linux / Windows (requires CUDA 12 + cuDNN 9 runtime).
    Cuda,
    /// AMD ROCm — Linux (requires ROCm runtime + `/dev/kfd`).
    ROCm,
    /// Apple CoreML — macOS Intel + Apple Silicon (system framework, always available on macOS ≥ 10.15).
    CoreML,
    /// Microsoft DirectML — Windows (system component, always available on Windows 10+).
    DirectML,
    /// CPU fallback — always available.
    Cpu,
}

impl AccelProvider {
    pub fn name(self) -> &'static str {
        match self {
            Self::Cuda => "CUDA",
            Self::ROCm => "ROCm",
            Self::CoreML => "CoreML",
            Self::DirectML => "DirectML",
            Self::Cpu => "CPU",
        }
    }

    pub fn is_gpu(self) -> bool {
        !matches!(self, Self::Cpu)
    }
}

/// Snapshot of AI service status for the system-info API.
#[derive(Debug, Clone)]
pub struct MediaIntelligenceStatus {
    /// The active execution provider (CPU = none available).
    pub accel_provider: AccelProvider,
    pub ocr_loaded: bool,
    pub clip_loaded: bool,
    pub face_loaded: bool,
    pub stt_loaded: bool,
}
pub mod ocr;
pub mod ocr_backend;
pub mod ocr_detector;
pub mod ocr_manager;
pub mod sidecar;
pub mod stt;
mod tokenizer;
pub mod worker;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use config::{MediaIntelligenceConfig, data_local_path};
use tokio::sync::{OnceCell, RwLock};

/// How long an idle model stays in memory before eviction.
const MODEL_IDLE_TIMEOUT: Duration = Duration::from_mins(3); // 3 minutes

/// Max intra-op threads per ONNX Runtime session.
/// Use half the logical CPUs (min 4, max 16) — sweet spot for parallel CPU inference.
/// The old deadlock risk (with `std::sync::Mutex` + blocking) is gone now that all
/// inference uses `tokio::sync::Mutex` + `run_async`.
fn ort_intra_op_threads() -> usize {
    let cpus = std::thread::available_parallelism().map_or(4, std::num::NonZero::get);
    (cpus / 2).clamp(4, 16)
}

fn cuda_device_id() -> i32 {
    std::env::var("TOKIMO_MEDIA_INTELLIGENCE_CUDA_DEVICE_ID")
        .ok()
        .and_then(|v| v.parse::<i32>().ok())
        .unwrap_or(0)
}

/// Build an ONNX Runtime session from a model file.
/// Uses the execution provider selected at init (CUDA / ROCm / CoreML / DirectML / CPU).
/// If the selected GPU EP fails to register, falls back to CPU with a warning.
pub fn build_session(path: impl AsRef<std::path::Path>) -> ort::Result<ort::session::Session> {
    use ort::session::Session;

    let path = path.as_ref();
    let filename = path.file_name().unwrap_or_default().to_string_lossy();
    let ep = active_ep();

    if ep == AccelProvider::Cpu {
        tracing::info!("[ort] Building session for {filename} (CPU)");
        return Session::builder()?
            .with_intra_threads(ort_intra_op_threads())?
            .commit_from_file(path);
    }

    tracing::info!("[ort] Building session for {filename} with {} EP", ep.name());

    // macOS: CoreML
    #[cfg(target_os = "macos")]
    if ep == AccelProvider::CoreML {
        let r = Session::builder()?
            .with_intra_threads(ort_intra_op_threads())?
            .with_execution_providers([ort::ep::CoreML::default().build().error_on_failure()]);
        match r {
            Ok(mut b) => {
                let s = b.commit_from_file(path)?;
                tracing::info!("[ort] ✓ Session {filename} loaded with CoreML EP");
                return Ok(s);
            }
            Err(e) => {
                tracing::warn!("[ort] CoreML EP failed for {filename}: {e} — falling back to CPU");
                return Session::builder()?
                    .with_intra_threads(ort_intra_op_threads())?
                    .commit_from_file(path);
            }
        }
    }

    // Windows: DirectML
    #[cfg(target_os = "windows")]
    if ep == AccelProvider::DirectML {
        let r = Session::builder()?
            .with_intra_threads(ort_intra_op_threads())?
            .with_execution_providers([ort::ep::DirectML::default().build().error_on_failure()]);
        match r {
            Ok(mut b) => {
                let s = b.commit_from_file(path)?;
                tracing::info!("[ort] ✓ Session {filename} loaded with DirectML EP");
                return Ok(s);
            }
            Err(e) => {
                tracing::warn!("[ort] DirectML EP failed for {filename}: {e} — falling back to CPU");
                return Session::builder()?
                    .with_intra_threads(ort_intra_op_threads())?
                    .commit_from_file(path);
            }
        }
    }

    // Linux AMD: ROCm
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    if ep == AccelProvider::ROCm {
        let r = Session::builder()?
            .with_intra_threads(ort_intra_op_threads())?
            .with_execution_providers([ort::ep::ROCm::default().build().error_on_failure()]);
        match r {
            Ok(mut b) => {
                let s = b.commit_from_file(path)?;
                tracing::info!("[ort] ✓ Session {filename} loaded with ROCm EP");
                return Ok(s);
            }
            Err(e) => {
                tracing::warn!("[ort] ROCm EP failed for {filename}: {e} — falling back to CPU");
                return Session::builder()?
                    .with_intra_threads(ort_intra_op_threads())?
                    .commit_from_file(path);
            }
        }
    }

    // NVIDIA CUDA (Linux / Windows)
    if ep == AccelProvider::Cuda {
        let device_id = cuda_device_id();
        let r = Session::builder()?
            .with_intra_threads(ort_intra_op_threads())?
            .with_execution_providers([ort::ep::CUDA::default()
                .with_device_id(device_id)
                .build()
                .error_on_failure()]);
        match r {
            Ok(mut b) => {
                let s = b.commit_from_file(path)?;
                tracing::info!("[ort] ✓ Session {filename} loaded with CUDA EP (device {device_id})");
                return Ok(s);
            }
            Err(e) => {
                tracing::warn!("[ort] CUDA EP registration failed for {filename}: {e} — falling back to CPU");
            }
        }
    }

    // CPU fallback
    Session::builder()?
        .with_intra_threads(ort_intra_op_threads())?
        .commit_from_file(path)
}

/// The GPU execution provider selected at init (CPU = disabled / not available).
static ACTIVE_EP: std::sync::OnceLock<AccelProvider> = std::sync::OnceLock::new();

/// Returns the active execution provider (set once during `MediaIntelligenceService::new`).
pub fn active_ep() -> AccelProvider {
    ACTIVE_EP.get().copied().unwrap_or(AccelProvider::Cpu)
}

/// Backward-compatible helper — true only when CUDA EP is active.
pub fn is_cuda_enabled() -> bool {
    active_ep() == AccelProvider::Cuda
}

// ── Platform-gated detection helpers ─────────────────────────────────────────

/// Check whether a shared library is loadable by the system linker.
/// Linux: queries `ldconfig` + common hard-coded dirs.
/// macOS: checks common Homebrew / system install paths.
/// Windows: not needed (DirectML is a system component, no extra dylib check).
#[cfg(not(target_os = "windows"))]
fn lib_available(lib: &str) -> bool {
    #[cfg(not(target_os = "macos"))]
    {
        let lib_dirs = [
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib/aarch64-linux-gnu",
            "/lib/x86_64-linux-gnu",
            "/usr/local/lib",
            "/usr/local/cuda/lib64",
            "/usr/local/cuda-12/lib64",
            "/usr/local/cuda-12/targets/x86_64-linux/lib",
            "/opt/rocm/lib",
        ];
        let ldconfig = std::process::Command::new("ldconfig")
            .arg("-p")
            .output()
            .ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
            .unwrap_or_default();
        let ld_library_path = std::env::var_os("LD_LIBRARY_PATH")
            .map(|paths| std::env::split_paths(&paths).collect::<Vec<_>>())
            .unwrap_or_default();
        ldconfig.contains(lib)
            || lib_dirs.iter().any(|d| std::path::Path::new(d).join(lib).exists())
            || ld_library_path.iter().any(|d| d.join(lib).exists())
    }
    #[cfg(target_os = "macos")]
    {
        let lib_dirs = [
            "/usr/local/lib",
            "/usr/lib",
            "/opt/homebrew/lib",
            "/opt/homebrew/opt/onnxruntime/lib",
        ];
        lib_dirs.iter().any(|d| std::path::Path::new(d).join(lib).exists())
    }
}

/// Check that at least one NVIDIA GPU device is accessible.
/// Guards against CUDA base images running on CPU-only hosts without `--gpus`.
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
fn gpu_device_present() -> bool {
    std::path::Path::new("/dev/nvidia0").exists()
        || std::process::Command::new("nvidia-smi")
            .arg("--query-gpu=name")
            .arg("--format=csv,noheader")
            .output()
            .is_ok_and(|o| o.status.success() && !o.stdout.is_empty())
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
fn char_device_status(path: &str) -> Result<(), String> {
    use std::os::unix::fs::FileTypeExt;

    match std::fs::metadata(path) {
        Ok(meta) if meta.file_type().is_char_device() => Ok(()),
        Ok(_) => Err(format!("{path} is not a character device")),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(format!("{path} is not accessible: {e}")),
    }
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
fn cuda_runtime_device_count() -> Result<i32, String> {
    use std::ffi::{CStr, CString};

    type CudaGetDeviceCount = unsafe extern "C" fn(*mut libc::c_int) -> libc::c_int;
    type CudaGetErrorString = unsafe extern "C" fn(libc::c_int) -> *const libc::c_char;

    let lib_name = CString::new("libcudart.so.12").expect("static library name has no nul");
    let count_symbol = CString::new("cudaGetDeviceCount").expect("static symbol has no nul");
    let error_symbol = CString::new("cudaGetErrorString").expect("static symbol has no nul");

    // SAFETY: We only call well-known CUDA Runtime API symbols with their C ABI
    // signatures, and close the library before returning from this probe.
    unsafe {
        let lib = libc::dlopen(lib_name.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL);
        if lib.is_null() {
            return Err(dl_error("dlopen libcudart.so.12 failed"));
        }

        let count_ptr = libc::dlsym(lib, count_symbol.as_ptr());
        let error_ptr = libc::dlsym(lib, error_symbol.as_ptr());
        if count_ptr.is_null() || error_ptr.is_null() {
            let reason = dl_error("dlsym CUDA Runtime API failed");
            libc::dlclose(lib);
            return Err(reason);
        }

        let cuda_get_device_count: CudaGetDeviceCount = std::mem::transmute(count_ptr);
        let cuda_get_error_string: CudaGetErrorString = std::mem::transmute(error_ptr);

        let mut count = -1;
        let rc = cuda_get_device_count(&mut count);
        if rc != 0 {
            let msg = CStr::from_ptr(cuda_get_error_string(rc)).to_string_lossy().into_owned();
            libc::dlclose(lib);
            return Err(format!("cudaGetDeviceCount failed: {msg} ({rc})"));
        }

        libc::dlclose(lib);
        Ok(count)
    }
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
fn dl_error(prefix: &str) -> String {
    // SAFETY: `dlerror` returns a thread-local C string pointer owned by libc.
    unsafe {
        let err = libc::dlerror();
        if err.is_null() {
            return prefix.to_string();
        }
        format!("{prefix}: {}", std::ffi::CStr::from_ptr(err).to_string_lossy())
    }
}

/// Detect NVIDIA CUDA — Linux / Windows.
/// Returns Ok(()) if ready, Err with the first missing piece otherwise.
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
fn detect_cuda() -> Result<(), String> {
    let checks: &[(&str, &str)] = &[
        (
            "libonnxruntime_providers_cuda.so",
            "ORT CUDA provider not in linker path",
        ),
        ("libcudart.so.12", "libcudart.so.12 missing"),
        ("libcublas.so.12", "libcublas.so.12 missing"),
        ("libcublasLt.so.12", "libcublasLt.so.12 missing"),
        ("libcufft.so.11", "libcufft.so.11 missing"),
        ("libcurand.so.10", "libcurand.so.10 missing"),
        ("libcudnn.so.9", "libcudnn.so.9 missing"),
    ];
    for (lib, reason) in checks {
        if !lib_available(lib) {
            return Err(reason.to_string());
        }
    }
    #[cfg(not(target_os = "windows"))]
    if !gpu_device_present() {
        return Err("no GPU device (/dev/nvidia0 absent, nvidia-smi failed)".to_string());
    }
    char_device_status("/dev/nvidia0")?;
    char_device_status("/dev/nvidiactl")?;
    char_device_status("/dev/nvidia-uvm")?;
    let device_count = cuda_runtime_device_count()?;
    if device_count <= 0 {
        return Err("cudaGetDeviceCount returned no CUDA devices".to_string());
    }
    Ok(())
}

/// Detect AMD ROCm — Linux only.
#[cfg(not(any(target_os = "macos", target_os = "windows")))]
fn detect_rocm() -> Result<(), &'static str> {
    if !std::path::Path::new("/dev/kfd").exists() {
        return Err("/dev/kfd not found");
    }
    let checks: &[(&str, &str)] = &[
        (
            "libonnxruntime_providers_rocm.so",
            "ORT ROCm provider not in linker path",
        ),
        ("libamdhip64.so.6", "libamdhip64.so.6 missing"),
        ("libMIOpen.so.1", "libMIOpen.so.1 missing"),
        ("librocblas.so.4", "librocblas.so.4 missing"),
    ];
    for (lib, reason) in checks {
        if !lib_available(lib) {
            return Err(reason);
        }
    }
    Ok(())
}

/// Detect Apple CoreML — macOS only.
#[cfg(target_os = "macos")]
fn detect_coreml() -> Result<(), &'static str> {
    if lib_available("libonnxruntime_providers_coreml.dylib") {
        Ok(())
    } else {
        Err("ORT CoreML provider dylib not in library path")
    }
}

/// Detect Microsoft DirectML — Windows only.
#[cfg(target_os = "windows")]
fn detect_directml() -> Result<(), &'static str> {
    let found = std::process::Command::new("where")
        .arg("onnxruntime_providers_directml.dll")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if found {
        Ok(())
    } else {
        Err("onnxruntime_providers_directml.dll not in PATH")
    }
}

/// Detect the best available EP and return it with a human-readable description for logging.
fn detect_best_ep(disable_hardware_acceleration: bool) -> (AccelProvider, String) {
    if disable_hardware_acceleration {
        return (AccelProvider::Cpu, "CPU (hardware acceleration disabled)".into());
    }

    #[cfg(target_os = "macos")]
    match detect_coreml() {
        Ok(()) => return (AccelProvider::CoreML, "CoreML".into()),
        Err(r) => return (AccelProvider::Cpu, format!("CPU (CoreML unavailable: {r})")),
    }

    #[cfg(target_os = "windows")]
    match detect_directml() {
        Ok(()) => return (AccelProvider::DirectML, "DirectML".into()),
        Err(r) => {
            return (AccelProvider::Cpu, format!("CPU (DirectML unavailable: {r})"));
        }
    }

    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        let cuda = detect_cuda();
        if cuda.is_ok() {
            return (AccelProvider::Cuda, "CUDA".into());
        }
        let rocm = detect_rocm();
        if rocm.is_ok() {
            return (AccelProvider::ROCm, "ROCm".into());
        }
        let reason = format!("CPU (CUDA: {}; ROCm: {})", cuda.unwrap_err(), rocm.unwrap_err());
        return (AccelProvider::Cpu, reason);
    }

    #[allow(unreachable_code)]
    (AccelProvider::Cpu, "CPU".into())
}

/// Epoch-based timestamp (seconds since an arbitrary point).
fn epoch_secs() -> u64 {
    use std::time::SystemTime;
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Resolve the path to the embedded Python sidecar source directory.
///
/// Resolution order:
/// 1. `config.python_sidecar_dir` (explicit override)
/// 2. `$TOKIMO_MEDIA_INTELLIGENCE_PYTHON_DIR`
/// 3. `<exe_dir>/../../packages/tokimo-media-intelligence/python` (dev builds under `target/`)
/// 4. `CARGO_MANIFEST_DIR/python` (library tests)
fn python_sidecar_dir(cfg: &MediaIntelligenceConfig) -> std::path::PathBuf {
    use std::path::PathBuf;

    if let Some(p) = &cfg.python_sidecar_dir {
        return PathBuf::from(p);
    }
    if let Ok(p) = std::env::var("TOKIMO_MEDIA_INTELLIGENCE_PYTHON_DIR") {
        return PathBuf::from(p);
    }
    if let Ok(exe) = std::env::current_exe() {
        // exe at <repo>/target/debug/tokimo-media-intelligence-worker → climb to repo root
        let mut cur = exe.parent().map(std::path::Path::to_path_buf);
        for _ in 0..6 {
            let Some(dir) = cur else { break };
            let candidate = dir.join("packages/tokimo-media-intelligence/python");
            if candidate.exists() {
                return candidate;
            }
            cur = dir.parent().map(std::path::Path::to_path_buf);
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("python")
}

/// Shared AI service state. Holds lazy-loaded ONNX model sessions.
///
/// Wrap in `Arc` and store in your app state. All methods are `&self`.
/// Models are evicted from memory after [`MODEL_IDLE_TIMEOUT`] of inactivity.
pub struct MediaIntelligenceService {
    config: MediaIntelligenceConfig,
    ocr_manager: OnceCell<ocr_manager::OcrManager>,
    sidecar: Arc<sidecar::SidecarManager>,
    // Heavy models — evictable (wrapped in Arc for shared ownership during eviction)
    clip: RwLock<Option<Arc<clip::ClipService>>>,
    clip_last_use: AtomicU64,
    face: RwLock<Option<Arc<face::FaceService>>>,
    face_last_use: AtomicU64,
    stt: RwLock<Option<stt::SttService>>,
    stt_last_use: AtomicU64,
    streaming_stt: RwLock<Option<stt::StreamingSttService>>,
    streaming_stt_last_use: AtomicU64,
}

impl MediaIntelligenceService {
    pub fn new(config: MediaIntelligenceConfig) -> Arc<Self> {
        let (ep, ep_desc) = detect_best_ep(config.disable_hardware_acceleration);

        // OnceLock — first call wins; subsequent MediaIntelligenceService::new calls (tests) are no-ops.
        let _ = ACTIVE_EP.set(ep);

        let applied = {
            let data_local = std::path::PathBuf::from(data_local_path());
            match worker::client::resolve_ort_dylib_path(&data_local) {
                Some(dylib_path) => ort::init_from(dylib_path).map_or_else(
                    |e| {
                        tracing::warn!("init_from failed ({e}), falling back to default init");
                        ort::init().commit()
                    },
                    ort::environment::EnvironmentBuilder::commit,
                ),
                None => ort::init().commit(),
            }
        };
        if applied {
            tracing::info!("ONNX Runtime initialized — EP: {ep_desc}");
        }

        Arc::new(Self {
            sidecar: sidecar::SidecarManager::new(python_sidecar_dir(&config), config.models_dir.clone()),
            config,
            ocr_manager: OnceCell::new(),
            clip: RwLock::new(None),
            clip_last_use: AtomicU64::new(0),
            face: RwLock::new(None),
            face_last_use: AtomicU64::new(0),
            stt: RwLock::new(None),
            stt_last_use: AtomicU64::new(0),
            streaming_stt: RwLock::new(None),
            streaming_stt_last_use: AtomicU64::new(0),
        })
    }

    /// Access the Python OCR sidecar manager (lazy-spawned on first use).
    pub fn sidecar(&self) -> &Arc<sidecar::SidecarManager> {
        &self.sidecar
    }

    /// Start background eviction loop. Call once after creating the service.
    pub fn start_idle_eviction(self: &Arc<Self>) {
        let svc = Arc::clone(self);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));
            loop {
                interval.tick().await;
                svc.evict_idle().await;
            }
        });
    }

    /// Evict models that haven't been used for [`MODEL_IDLE_TIMEOUT`].
    async fn evict_idle(&self) {
        let now = epoch_secs();
        let threshold = MODEL_IDLE_TIMEOUT.as_secs();
        let mut evicted_any = false;

        // CLIP
        let clip_last = self.clip_last_use.load(Ordering::Relaxed);
        if clip_last > 0 && now.saturating_sub(clip_last) >= threshold {
            let mut guard = self.clip.write().await;
            if guard.is_some() {
                *guard = None;
                tracing::info!("Evicted idle CLIP model from memory");
                evicted_any = true;
            }
        }

        // Face
        let face_last = self.face_last_use.load(Ordering::Relaxed);
        if face_last > 0 && now.saturating_sub(face_last) >= threshold {
            let mut guard = self.face.write().await;
            if guard.is_some() {
                *guard = None;
                tracing::info!("Evicted idle Face model from memory");
                evicted_any = true;
            }
        }

        // OCR
        if let Some(mgr) = self.ocr_manager.get() {
            let ocr_last = mgr.last_use_epoch();
            if ocr_last > 0 && now.saturating_sub(ocr_last) >= threshold {
                mgr.evict_all().await;
                evicted_any = true;
            }
        }

        // STT (SenseVoice)
        let stt_last = self.stt_last_use.load(Ordering::Relaxed);
        if stt_last > 0 && now.saturating_sub(stt_last) >= threshold {
            let mut guard = self.stt.write().await;
            if guard.is_some() {
                *guard = None;
                tracing::info!("Evicted idle SenseVoice STT model from memory");
                evicted_any = true;
            }
        }

        // Streaming STT (Zipformer)
        let stream_last = self.streaming_stt_last_use.load(Ordering::Relaxed);
        if stream_last > 0 && now.saturating_sub(stream_last) >= threshold {
            let mut guard = self.streaming_stt.write().await;
            if guard.is_some() {
                *guard = None;
                tracing::info!("Evicted idle streaming STT model from memory");
                evicted_any = true;
            }
        }

        // After evicting models, ask glibc to return freed memory to the OS.
        // Without this, RSS stays high because glibc keeps freed pages in its arena.
        if evicted_any {
            #[cfg(target_os = "linux")]
            {
                #[allow(unsafe_code)]
                unsafe {
                    libc::malloc_trim(0)
                };
                tracing::info!("Called malloc_trim to release memory to OS");
            }
        }
    }

    pub fn config(&self) -> &MediaIntelligenceConfig {
        &self.config
    }

    pub fn models_dir(&self) -> &str {
        &self.config.models_dir
    }

    /// Report current AI service status for the system-info API.
    pub fn status(&self) -> MediaIntelligenceStatus {
        MediaIntelligenceStatus {
            accel_provider: active_ep(),
            ocr_loaded: self
                .ocr_manager
                .get()
                .is_some_and(ocr_manager::OcrManager::has_loaded_backends),
            clip_loaded: self.clip.try_read().is_ok_and(|g| g.is_some()),
            face_loaded: self.face.try_read().is_ok_and(|g| g.is_some()),
            stt_loaded: self.stt.try_read().is_ok_and(|g| g.is_some()),
        }
    }

    // ── Feature toggles ──────────────────────────────────────────────────

    pub fn is_ocr_enabled(&self) -> bool {
        self.config.enable_ocr
    }

    pub fn is_clip_enabled(&self) -> bool {
        self.config.enable_clip
    }

    pub fn is_face_enabled(&self) -> bool {
        self.config.enable_face
    }

    pub fn is_stt_enabled(&self) -> bool {
        self.config.enable_stt
    }

    // ── Model download ───────────────────────────────────────────────────

    /// Download any missing models. Call during startup.
    pub async fn ensure_models(&self) -> Result<(), String> {
        models::ensure_models(&self.config).await
    }

    /// Download any missing models with progress callback.
    /// Callback: (`file_name`, status, percent, `downloaded_bytes`, `total_bytes`)
    pub async fn ensure_models_with_progress(&self, on_progress: models::ProgressFn) -> Result<(), String> {
        models::ensure_models_with_progress(&self.config, Some(on_progress)).await
    }

    /// Download models for a single category with progress callback.
    pub async fn ensure_category_with_progress(
        &self,
        category: models::ModelCategory,
        on_progress: models::ProgressFn,
    ) -> Result<(), String> {
        models::ensure_category_with_progress(&self.config, category, Some(on_progress)).await
    }

    /// Check whether all enabled model files exist on disk.
    pub fn models_ready(&self) -> bool {
        models::all_models_present(&self.config)
    }

    /// Check whether CLIP model files exist on disk.
    pub fn clip_models_ready(&self) -> bool {
        models::clip_models_present(&self.config)
    }

    /// Check whether OCR model files exist on disk.
    pub fn ocr_models_ready(&self) -> bool {
        models::ocr_models_present(&self.config)
    }

    /// Check whether OCR Server model files exist on disk.
    pub fn ocr_server_models_ready(&self) -> bool {
        models::ocr_server_models_present(&self.config)
    }

    /// Check whether OCR Mobile model files exist on disk.
    pub fn ocr_mobile_models_ready(&self) -> bool {
        models::ocr_mobile_models_present(&self.config)
    }

    /// Check whether face model files exist on disk.
    pub fn face_models_ready(&self) -> bool {
        models::face_models_present(&self.config)
    }

    // ── OCR ──────────────────────────────────────────────────────────────

    /// Run OCR on raw image bytes. Returns detected text regions.
    ///
    /// `model_name` selects the backend (e.g. `"rapid-ocr-rust"`).
    /// Pass `None` to use the default server model.
    pub async fn ocr(
        &self,
        image_bytes: &[u8],
        model_name: Option<&str>,
        cancel_id: Option<&str>,
    ) -> Result<Vec<ocr::OcrItem>, String> {
        if !self.config.enable_ocr {
            return Err("OCR is disabled".into());
        }
        let manager = self
            .ocr_manager
            .get_or_try_init(|| async {
                Ok::<_, String>(ocr_manager::OcrManager::with_options(
                    self.config.models_dir.clone(),
                    Arc::clone(&self.sidecar),
                    self.config.ocr_det_max_side,
                ))
            })
            .await?;
        let model = model_name.unwrap_or(ocr_manager::DEFAULT_MODEL);
        cancel::with_cancel_id(cancel_id, manager.ocr(model, image_bytes)).await
    }

    /// Hybrid OCR: `det_model` provides bounding boxes, `vlm_model` provides
    /// accurate text. Results are merged by the sidecar.
    /// Returns (`merged_items`, `optional_debug_info`).
    pub async fn ocr_hybrid(
        &self,
        image_bytes: &[u8],
        det_model: &str,
        vlm_model: &str,
        cancel_id: Option<&str>,
    ) -> Result<(Vec<ocr::OcrItem>, Option<serde_json::Value>), String> {
        if !self.config.enable_ocr {
            return Err("OCR is disabled".into());
        }
        let manager = self
            .ocr_manager
            .get_or_try_init(|| async {
                Ok::<_, String>(ocr_manager::OcrManager::with_options(
                    self.config.models_dir.clone(),
                    Arc::clone(&self.sidecar),
                    self.config.ocr_det_max_side,
                ))
            })
            .await?;
        cancel::with_cancel_id(cancel_id, manager.ocr_hybrid(det_model, vlm_model, image_bytes)).await
    }

    /// List available OCR models and their status.
    pub fn ocr_available_models(&self) -> Vec<ocr_manager::OcrModelInfo> {
        match self.ocr_manager.get() {
            Some(mgr) => mgr.available_models(),
            None => ocr_manager::OcrManager::known_models(),
        }
    }

    // ── CLIP ─────────────────────────────────────────────────────────────

    /// Embed an image → 512-dim CLIP vector.
    pub async fn clip_image(&self, image_bytes: &[u8], cancel_id: Option<&str>) -> Result<Vec<f32>, String> {
        if !self.config.enable_clip {
            return Err("CLIP is disabled".into());
        }
        let img = image::load_from_memory(image_bytes).map_err(|e| format!("Invalid image: {e}"))?;
        let svc = self.get_or_init_clip().await?;
        cancel::with_cancel_id(cancel_id, svc.embed_image(&img)).await
    }

    /// Embed text → 512-dim CLIP vector.
    pub async fn clip_text(&self, text: &str, cancel_id: Option<&str>) -> Result<Vec<f32>, String> {
        if !self.config.enable_clip {
            return Err("CLIP is disabled".into());
        }
        let svc = self.get_or_init_clip().await?;
        cancel::with_cancel_id(cancel_id, svc.embed_text(text)).await
    }

    /// Classify an image vector against the built-in taxonomy using CLIP zero-shot.
    pub async fn clip_classify(&self, image_vec: &[f32]) -> Result<Vec<clip_categories::TagResult>, String> {
        if !self.config.enable_clip {
            return Err("CLIP is disabled".into());
        }
        let svc = self.get_or_init_clip().await?;
        svc.classify(image_vec)
    }

    async fn get_or_init_clip(&self) -> Result<Arc<clip::ClipService>, String> {
        self.clip_last_use.store(epoch_secs(), Ordering::Relaxed);
        {
            let guard = self.clip.read().await;
            if let Some(svc) = guard.as_ref() {
                return Ok(Arc::clone(svc));
            }
        }
        let mut guard = self.clip.write().await;
        if let Some(svc) = guard.as_ref() {
            return Ok(Arc::clone(svc));
        }
        let dir = self.config.models_dir.clone();
        let svc = Arc::new(
            tokio::task::spawn_blocking(move || clip::ClipService::new(&dir))
                .await
                .map_err(|e| format!("CLIP load panicked: {e}"))??,
        );
        *guard = Some(Arc::clone(&svc));
        Ok(svc)
    }

    // ── Face ─────────────────────────────────────────────────────────────

    /// Detect faces and extract 512-dim embeddings.
    pub async fn detect_faces(
        &self,
        image_bytes: &[u8],
        cancel_id: Option<&str>,
    ) -> Result<Vec<face::FaceDetection>, String> {
        if !self.config.enable_face {
            return Err("Face recognition is disabled".into());
        }
        let img = image::load_from_memory(image_bytes).map_err(|e| format!("Invalid image: {e}"))?;
        let svc = self.get_or_init_face().await?;
        cancel::with_cancel_id(cancel_id, svc.detect_faces(&img)).await
    }

    async fn get_or_init_face(&self) -> Result<Arc<face::FaceService>, String> {
        self.face_last_use.store(epoch_secs(), Ordering::Relaxed);
        {
            let guard = self.face.read().await;
            if let Some(svc) = guard.as_ref() {
                return Ok(Arc::clone(svc));
            }
        }
        let mut guard = self.face.write().await;
        if let Some(svc) = guard.as_ref() {
            return Ok(Arc::clone(svc));
        }
        let dir = self.config.models_dir.clone();
        let svc = Arc::new(
            tokio::task::spawn_blocking(move || face::FaceService::new(&dir))
                .await
                .map_err(|e| format!("Face load panicked: {e}"))??,
        );
        *guard = Some(Arc::clone(&svc));
        Ok(svc)
    }

    // ── STT ──────────────────────────────────────────────────────────────

    /// Transcribe WAV audio bytes to text using `SenseVoice` (sherpa-onnx).
    pub async fn transcribe_audio(&self, wav_bytes: &[u8]) -> Result<String, String> {
        if !self.config.enable_stt {
            return Err("STT is disabled".into());
        }
        let svc = self.get_or_init_stt().await?;
        let bytes = wav_bytes.to_vec();
        tokio::task::spawn_blocking(move || svc.transcribe(&bytes))
            .await
            .map_err(|e| format!("Transcription task panicked: {e}"))?
    }

    /// Transcribe raw f32 PCM samples using `SenseVoice` (for streaming refinement).
    pub async fn transcribe_pcm(&self, samples: Vec<f32>, sample_rate: i32) -> Result<String, String> {
        if !self.config.enable_stt {
            return Err("STT is disabled".into());
        }
        let svc = self.get_or_init_stt().await?;
        tokio::task::spawn_blocking(move || svc.transcribe_pcm(&samples, sample_rate))
            .await
            .map_err(|e| format!("Transcription task panicked: {e}"))?
    }

    async fn get_or_init_stt(&self) -> Result<stt::SttService, String> {
        self.stt_last_use.store(epoch_secs(), Ordering::Relaxed);
        {
            let guard = self.stt.read().await;
            if let Some(svc) = guard.as_ref() {
                return Ok(svc.clone());
            }
        }
        let mut guard = self.stt.write().await;
        if let Some(svc) = guard.as_ref() {
            return Ok(svc.clone());
        }
        let dir = self.config.models_dir.clone();
        let svc = tokio::task::spawn_blocking(move || stt::SttService::new(&dir, stt::DEFAULT_MODEL))
            .await
            .map_err(|e| format!("STT load panicked: {e}"))??;
        *guard = Some(svc.clone());
        Ok(svc)
    }

    /// Get STT model status information.
    pub fn stt_models_status(&self) -> Vec<stt::SttModelStatus> {
        stt::models_status(&self.config.models_dir)
    }

    /// Check if the default STT model is ready.
    pub fn stt_model_ready(&self) -> bool {
        stt::model_exists(&self.config.models_dir, stt::DEFAULT_MODEL)
    }

    /// Check if the streaming STT model is ready.
    pub fn streaming_stt_model_ready(&self) -> bool {
        stt::model_exists(&self.config.models_dir, stt::STREAMING_MODEL)
    }

    /// Get or init the streaming STT service.
    pub async fn streaming_stt(&self) -> Result<stt::StreamingSttService, String> {
        if !self.config.enable_stt {
            return Err("STT is disabled".into());
        }
        self.get_or_init_streaming_stt().await
    }

    async fn get_or_init_streaming_stt(&self) -> Result<stt::StreamingSttService, String> {
        self.streaming_stt_last_use.store(epoch_secs(), Ordering::Relaxed);
        {
            let guard = self.streaming_stt.read().await;
            if let Some(svc) = guard.as_ref() {
                return Ok(svc.clone());
            }
        }
        let mut guard = self.streaming_stt.write().await;
        if let Some(svc) = guard.as_ref() {
            return Ok(svc.clone());
        }
        let dir = self.config.models_dir.clone();
        let svc = tokio::task::spawn_blocking(move || stt::StreamingSttService::new(&dir))
            .await
            .map_err(|e| format!("Streaming STT load panicked: {e}"))??;
        *guard = Some(svc.clone());
        Ok(svc)
    }

    /// Download a specific STT model with progress reporting.
    ///
    /// `on_progress` is called with `(status, progress_0_100)`.
    pub async fn download_stt_model<F>(&self, model_id: &str, on_progress: F) -> Result<(), String>
    where
        F: Fn(&str, u8) + Send + 'static,
    {
        let model = stt::ALL_MODELS
            .iter()
            .find(|m| m.id() == model_id)
            .copied()
            .ok_or_else(|| format!("Unknown STT model: {model_id}"))?;

        let stt_dir = format!("{}/stt", self.config.models_dir);
        tokio::fs::create_dir_all(&stt_dir)
            .await
            .map_err(|e| format!("Failed to create stt dir: {e}"))?;

        let url = model.download_url();
        let archive_path = format!("{stt_dir}/_download_{}.tar.bz2", model.id());

        tracing::info!("Downloading STT model: {} → {stt_dir}", model.display_name());
        on_progress("downloading", 0);

        // Streaming download with progress
        let resp = reqwest::get(url)
            .await
            .map_err(|e| format!("HTTP request failed: {e}"))?;
        if !resp.status().is_success() {
            return Err(format!("HTTP {}: {url}", resp.status()));
        }

        let total_size = resp.content_length().unwrap_or(0);
        let mut downloaded: u64 = 0;
        let mut last_pct: u8 = 0;

        let mut file = tokio::fs::File::create(&archive_path)
            .await
            .map_err(|e| format!("Create file failed: {e}"))?;

        use tokio::io::AsyncWriteExt;
        let mut stream = resp.bytes_stream();
        use futures_util::StreamExt;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| format!("Download stream error: {e}"))?;
            file.write_all(&chunk).await.map_err(|e| format!("Write error: {e}"))?;
            downloaded += chunk.len() as u64;

            if total_size > 0 {
                let pct = ((downloaded as f64 / total_size as f64) * 100.0).min(100.0) as u8;
                if pct != last_pct {
                    last_pct = pct;
                    on_progress("downloading", pct);
                }
            }
        }
        file.flush().await.map_err(|e| format!("Flush error: {e}"))?;
        drop(file);

        let size_mb = downloaded as f64 / (1024.0 * 1024.0);
        tracing::info!("Download complete: {size_mb:.1} MB");
        on_progress("extracting", 0);

        // Extract using tar (tar.bz2 format) on blocking thread
        let archive_clone = archive_path.clone();
        let stt_dir_clone = stt_dir.clone();
        let extract_result = tokio::task::spawn_blocking(move || {
            std::process::Command::new("tar")
                .args(["xjf", &archive_clone, "-C", &stt_dir_clone])
                .output()
        })
        .await
        .map_err(|e| format!("Extract task panicked: {e}"))?
        .map_err(|e| format!("Failed to run tar: {e}"))?;
        if !extract_result.status.success() {
            let _ = tokio::fs::remove_file(&archive_path).await;
            let stderr = String::from_utf8_lossy(&extract_result.stderr).trim().to_string();
            if stderr.is_empty() {
                return Err(format!("tar extraction failed: exit status {}", extract_result.status));
            }
            return Err(format!("tar extraction failed: {stderr}"));
        }

        // Clean up archive
        let _ = tokio::fs::remove_file(&archive_path).await;

        on_progress("completed", 100);
        tracing::info!("STT model {} downloaded and extracted", model.display_name());
        Ok(())
    }
}
