//! `MediaIntelligenceWorkerClient` — drop-in replacement for `tokimo_media_intelligence::MediaIntelligenceService`.
//!
//! Wraps a [`Transport`](crate::worker::protocol::transport::Transport) implementation
//! (UDS or HTTP) and caches a [`WorkerInfo`] snapshot so that the many sync
//! getters on the old `MediaIntelligenceService` (e.g. `is_ocr_enabled()`, `models_ready()`)
//! can be served without awaiting.
//!
//! Conversion between `ai_worker_protocol` wire types and `rust-server`'s
//! internal types happens at the caller boundary — this crate intentionally
//! does **not** depend on `tokimo-media-intelligence`.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use arc_swap::ArcSwap;
use tokio::sync::mpsc;

use crate::worker::protocol::routes;
use crate::worker::protocol::transport::{AnyTransport, BidiStream, HttpTransport, UdsTransport};
use crate::worker::protocol::types as wire;
use crate::worker::protocol::{RpcError, RpcResult};

use super::settings::{MediaIntelligenceWorkerMode, MediaIntelligenceWorkerSettings};
use super::supervisor::{Supervisor, SupervisorConfig};

/// Result type returned from client methods. String error mirrors the
/// `Result<_, String>` contract of the previous `tokimo_media_intelligence::MediaIntelligenceService` so
/// caller sites need minimal rewrites.
pub type ClientResult<T> = Result<T, String>;

fn rpc_to_string(e: RpcError) -> String {
    e.to_string()
}

/// Stream handle returned by [`MediaIntelligenceWorkerClient::streaming_stt`].
pub struct StreamingSttSession {
    pub tx: mpsc::Sender<wire::SttClientFrame>,
    pub rx: mpsc::Receiver<RpcResult<wire::SttServerFrame>>,
}

pub struct MediaIntelligenceWorkerClient {
    transport: Arc<AnyTransport>,
    supervisor: Option<Arc<Supervisor>>,
    info: ArcSwap<wire::WorkerInfo>,
}

impl MediaIntelligenceWorkerClient {
    /// Construct a client around a raw transport. The caller is expected to
    /// have ensured the worker is reachable (or to install a [`Supervisor`]).
    pub fn new(transport: Arc<AnyTransport>) -> Arc<Self> {
        Arc::new(Self {
            transport,
            supervisor: None,
            info: ArcSwap::from_pointee(empty_info()),
        })
    }

    /// Construct a client with a supervisor that manages a local worker
    /// process (spawn / idle-exit / auto-respawn).
    pub fn with_supervisor(transport: Arc<AnyTransport>, supervisor: Arc<Supervisor>) -> Arc<Self> {
        Arc::new(Self {
            transport,
            supervisor: Some(supervisor),
            info: ArcSwap::from_pointee(empty_info()),
        })
    }

    /// Build an [`MediaIntelligenceWorkerClient`] from DB-persisted settings and the local
    /// data directory. Handles transport selection, worker binary resolution,
    /// and supervisor setup.
    pub fn from_settings(settings: &MediaIntelligenceWorkerSettings, data_local_path: &Path) -> Arc<Self> {
        let media_models_dir = data_local_path.join("media-intelligence");

        let socket_path = settings.socket_path.as_deref().map_or_else(
            || data_local_path.join("media-intelligence-worker.sock"),
            std::path::PathBuf::from,
        );

        let worker_binary = resolve_worker_binary(settings.worker_binary.as_deref());

        let transport: Arc<AnyTransport> = match settings.mode {
            MediaIntelligenceWorkerMode::Remote => {
                let url = settings
                    .remote_url
                    .clone()
                    .unwrap_or_else(|| "http://localhost:5679".to_string());
                let t = HttpTransport::new(url).expect("build HttpTransport");
                Arc::new(AnyTransport::Http(t))
            }
            MediaIntelligenceWorkerMode::Auto => Arc::new(AnyTransport::Uds(UdsTransport::new(socket_path.clone()))),
        };

        let client = match settings.mode {
            MediaIntelligenceWorkerMode::Remote => Self::new(Arc::clone(&transport)),
            MediaIntelligenceWorkerMode::Auto => {
                let mut extra_env = resolve_media_intelligence_python_dir();
                if !settings.hardware_acceleration_enabled {
                    extra_env.push(("TOKIMO_MEDIA_INTELLIGENCE_DISABLE_ACCEL".to_string(), "1".to_string()));
                }
                extra_env.push((
                    "TOKIMO_MEDIA_INTELLIGENCE_ACCEL_PROFILE".to_string(),
                    settings.acceleration_profile.as_env_value().to_string(),
                ));
                let cfg = SupervisorConfig {
                    worker_binary: PathBuf::from(worker_binary),
                    socket_path,
                    http_addr: None,
                    models_dir: Some(media_models_dir),
                    idle_secs: settings.effective_idle_secs(),
                    extra_env,
                    remote: false,
                };
                let sup = Supervisor::new(cfg, Arc::clone(&transport));
                Self::with_supervisor(Arc::clone(&transport), sup)
            }
        };
        client.spawn_info_refresher(Duration::from_secs(2));
        client
    }

    /// Start a background task that periodically refreshes the cached
    /// [`WorkerInfo`] snapshot. Call once after construction.
    pub fn spawn_info_refresher(self: &Arc<Self>, every: Duration) {
        let weak = Arc::downgrade(self);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(every).await;
                let Some(this) = weak.upgrade() else { break };
                let _ = this.refresh_info().await;
            }
        });
    }

    async fn ensure_up(&self) -> RpcResult<()> {
        if let Some(sv) = &self.supervisor {
            sv.ensure_up().await?;
        }
        Ok(())
    }

    fn mark_activity(&self) {
        if let Some(sv) = &self.supervisor {
            sv.mark_activity();
        }
    }

    async fn call<Req, Res>(&self, route: &str, req: &Req) -> RpcResult<Res>
    where
        Req: serde::Serialize + Sync,
        Res: serde::de::DeserializeOwned + Send,
    {
        self.ensure_up().await?;
        self.mark_activity();
        self.transport.call(route, req).await
    }

    /// Refresh the cached [`WorkerInfo`] snapshot by calling `/v1/info`.
    pub async fn refresh_info(&self) -> RpcResult<()> {
        let info: wire::WorkerInfo = self.transport.call(routes::INFO, &()).await?;
        self.info.store(Arc::new(info));
        Ok(())
    }

    // ---------------- Sync getters backed by cached `WorkerInfo` ----------------

    pub fn info(&self) -> Arc<wire::WorkerInfo> {
        self.info.load_full()
    }

    pub fn status(&self) -> wire::MediaIntelligenceStatus {
        let i = self.info.load();
        wire::MediaIntelligenceStatus {
            accel_provider: i.accel_provider,
            ocr_loaded: i.ocr_loaded,
            clip_loaded: i.clip_loaded,
            face_loaded: i.face_loaded,
            stt_loaded: i.stt_loaded,
        }
    }

    pub fn models_dir(&self) -> String {
        self.info.load().models_dir.clone()
    }

    pub fn ocr_det_max_side(&self) -> Option<u32> {
        self.info.load().ocr_det_max_side
    }

    pub fn is_ocr_enabled(&self) -> bool {
        self.info.load().ocr_enabled
    }
    pub fn is_clip_enabled(&self) -> bool {
        self.info.load().clip_enabled
    }
    pub fn is_face_enabled(&self) -> bool {
        self.info.load().face_enabled
    }
    pub fn is_stt_enabled(&self) -> bool {
        self.info.load().stt_enabled
    }

    pub fn models_ready(&self) -> bool {
        self.info.load().models_ready
    }
    pub fn ocr_models_ready(&self) -> bool {
        self.info.load().ocr_models_ready
    }
    pub fn ocr_server_models_ready(&self) -> bool {
        self.info.load().ocr_server_models_ready
    }
    pub fn ocr_mobile_models_ready(&self) -> bool {
        self.info.load().ocr_mobile_models_ready
    }
    pub fn clip_models_ready(&self) -> bool {
        self.info.load().clip_models_ready
    }
    pub fn face_models_ready(&self) -> bool {
        self.info.load().face_models_ready
    }
    pub fn stt_model_ready(&self) -> bool {
        self.info.load().stt_model_ready
    }
    pub fn streaming_stt_model_ready(&self) -> bool {
        self.info.load().streaming_stt_model_ready
    }

    // ---------------- OCR ----------------

    pub async fn ocr(
        &self,
        image: Vec<u8>,
        model: Option<String>,
        request_id: Option<String>,
    ) -> ClientResult<Vec<wire::OcrItem>> {
        self.call::<_, Vec<wire::OcrItem>>(
            routes::OCR,
            &wire::OcrRequest {
                image,
                model,
                request_id,
            },
        )
        .await
        .map_err(rpc_to_string)
    }

    pub async fn ocr_hybrid(
        &self,
        image: Vec<u8>,
        det_model: Option<String>,
        vlm_model: Option<String>,
        request_id: Option<String>,
    ) -> ClientResult<Vec<wire::OcrItem>> {
        self.call::<_, Vec<wire::OcrItem>>(
            routes::OCR_HYBRID,
            &wire::OcrHybridRequest {
                image,
                det_model,
                vlm_model,
                request_id,
            },
        )
        .await
        .map_err(rpc_to_string)
    }

    pub async fn ocr_available_models(&self) -> ClientResult<Vec<wire::OcrModelInfo>> {
        self.call::<_, Vec<wire::OcrModelInfo>>(routes::LIST_OCR_MODELS, &())
            .await
            .map_err(rpc_to_string)
    }

    // ---------------- CLIP ----------------

    pub async fn clip_image(&self, image: Vec<u8>, request_id: Option<String>) -> ClientResult<Vec<f32>> {
        self.call::<_, Vec<f32>>(routes::CLIP_IMAGE, &wire::ClipImageRequest { image, request_id })
            .await
            .map_err(rpc_to_string)
    }

    pub async fn clip_text(&self, text: String, request_id: Option<String>) -> ClientResult<Vec<f32>> {
        self.call::<_, Vec<f32>>(routes::CLIP_TEXT, &wire::ClipTextRequest { text, request_id })
            .await
            .map_err(rpc_to_string)
    }

    pub async fn clip_classify(&self, vector: Vec<f32>) -> ClientResult<Vec<wire::TagResult>> {
        self.call::<_, Vec<wire::TagResult>>(routes::CLIP_CLASSIFY, &wire::ClipClassifyRequest { vector })
            .await
            .map_err(rpc_to_string)
    }

    // ---------------- Face ----------------

    pub async fn detect_faces(
        &self,
        image: Vec<u8>,
        request_id: Option<String>,
    ) -> ClientResult<Vec<wire::FaceDetection>> {
        self.call::<_, Vec<wire::FaceDetection>>(routes::FACE_DETECT, &wire::FaceRequest { image, request_id })
            .await
            .map_err(rpc_to_string)
    }

    // ---------------- Cancel ----------------

    /// Terminate any in-flight ORT inference(s) registered under `request_id`.
    /// Returns `true` if at least one live inference was terminated.
    pub async fn cancel(&self, request_id: String) -> ClientResult<bool> {
        let res: wire::CancelResponse = self
            .call(routes::CANCEL, &wire::CancelRequest { request_id })
            .await
            .map_err(rpc_to_string)?;
        Ok(res.cancelled)
    }

    /// Hard-kill the worker process. Used as a last-resort escalation when
    /// cooperative cancel hasn't stopped in-flight inference after 5 s.
    /// The next inference RPC will transparently respawn the worker.
    /// No-op in Remote mode (no supervisor available).
    pub async fn kill_worker(&self) -> ClientResult<()> {
        if let Some(sv) = &self.supervisor {
            sv.kill_and_respawn().await.map_err(rpc_to_string)?;
        }
        Ok(())
    }

    // ---------------- STT (one-shot) ----------------

    pub async fn transcribe_audio(&self, wav: Vec<u8>) -> ClientResult<String> {
        let res: wire::SttTranscribeResponse = self
            .call(routes::STT_TRANSCRIBE, &wire::SttTranscribeRequest { wav })
            .await
            .map_err(rpc_to_string)?;
        Ok(res.text)
    }

    pub async fn transcribe_pcm(&self, samples: Vec<f32>, sample_rate: i32) -> ClientResult<String> {
        let res: wire::SttTranscribeResponse = self
            .call(
                routes::STT_TRANSCRIBE_PCM,
                &wire::SttTranscribePcmRequest { samples, sample_rate },
            )
            .await
            .map_err(rpc_to_string)?;
        Ok(res.text)
    }

    pub async fn stt_models_status(&self) -> ClientResult<Vec<wire::SttModelStatus>> {
        self.call::<_, Vec<wire::SttModelStatus>>(routes::STT_MODELS_STATUS, &())
            .await
            .map_err(rpc_to_string)
    }

    // ---------------- STT (streaming) ----------------

    pub async fn streaming_stt(&self) -> ClientResult<StreamingSttSession> {
        self.ensure_up().await.map_err(rpc_to_string)?;
        self.mark_activity();
        let BidiStream { tx, rx } = self
            .transport
            .open_bidi::<wire::SttClientFrame, wire::SttServerFrame>(routes::STT_STREAM)
            .await
            .map_err(rpc_to_string)?;
        Ok(StreamingSttSession { tx, rx })
    }

    // ---------------- Model download progress ----------------

    pub async fn ensure_category_with_progress(
        &self,
        category: Option<wire::ModelCategory>,
    ) -> ClientResult<mpsc::Receiver<RpcResult<wire::ProgressFrame>>> {
        self.ensure_up().await.map_err(rpc_to_string)?;
        self.mark_activity();
        self.transport
            .call_stream::<_, wire::ProgressFrame>(routes::ENSURE_CATEGORY, &wire::EnsureCategoryRequest { category })
            .await
            .map_err(rpc_to_string)
    }

    pub async fn download_stt_model(
        &self,
        model_id: String,
    ) -> ClientResult<mpsc::Receiver<RpcResult<wire::ProgressFrame>>> {
        self.ensure_up().await.map_err(rpc_to_string)?;
        self.mark_activity();
        self.transport
            .call_stream::<_, wire::ProgressFrame>(routes::DOWNLOAD_STT, &wire::DownloadSttRequest { model_id })
            .await
            .map_err(rpc_to_string)
    }

    // ---------------- Catalog (unified media intelligence API) ----------------

    pub async fn get_catalog(&self, languages: Vec<String>) -> ClientResult<wire::ModelCatalog> {
        self.ensure_up().await.map_err(rpc_to_string)?;
        self.mark_activity();
        let res: RpcResult<wire::ModelCatalog> = self
            .transport
            .call(routes::CATALOG, &wire::CatalogRequest { languages })
            .await;
        res.map_err(rpc_to_string)
    }

    pub async fn download_model(
        &self,
        model_id: String,
    ) -> ClientResult<mpsc::Receiver<RpcResult<wire::ProgressFrame>>> {
        self.ensure_up().await.map_err(rpc_to_string)?;
        self.mark_activity();
        self.transport
            .call_stream::<_, wire::ProgressFrame>(routes::MODEL_DOWNLOAD, &wire::ModelActionRequest { model_id })
            .await
            .map_err(rpc_to_string)
    }

    pub async fn unload_model(&self, model_id: String) -> ClientResult<()> {
        self.ensure_up().await.map_err(rpc_to_string)?;
        self.mark_activity();
        let res: RpcResult<wire::ShutdownResponse> = self
            .transport
            .call(routes::MODEL_UNLOAD, &wire::ModelActionRequest { model_id })
            .await;
        res.map(|_| ()).map_err(rpc_to_string)
    }

    // ---------------- Lifecycle ----------------

    /// Ask the worker to shut down gracefully. Safe to call even if not up.
    pub async fn shutdown(&self) -> ClientResult<()> {
        let res: RpcResult<wire::ShutdownResponse> = self.transport.call(routes::SHUTDOWN, &()).await;
        match res {
            Ok(_) | Err(RpcError::Transport(_)) => Ok(()),
            Err(e) => Err(e.to_string()),
        }
    }
}

fn empty_info() -> wire::WorkerInfo {
    wire::WorkerInfo {
        accel_provider: wire::AccelProvider::Cpu,
        acceleration_profile: wire::AccelerationProfile::Balanced,
        models_dir: String::new(),
        ocr_det_max_side: None,
        ocr_enabled: false,
        clip_enabled: false,
        face_enabled: false,
        stt_enabled: false,
        models_ready: false,
        ocr_models_ready: false,
        ocr_server_models_ready: false,
        ocr_mobile_models_ready: false,
        clip_models_ready: false,
        face_models_ready: false,
        stt_model_ready: false,
        streaming_stt_model_ready: false,
        ocr_loaded: false,
        clip_loaded: false,
        face_loaded: false,
        stt_loaded: false,
    }
}

/// Resolve the `tokimo-media-intelligence-worker` binary path.
/// Prefers the explicit override, then checks sibling of current exe, falls
/// back to PATH lookup.
fn resolve_worker_binary(override_path: Option<&str>) -> String {
    if let Some(path) = override_path {
        return path.to_string();
    }
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join("tokimo-media-intelligence-worker")))
        .and_then(|p| if p.exists() { Some(p) } else { None })
        .map_or_else(
            || "tokimo-media-intelligence-worker".to_string(),
            |p| p.to_string_lossy().into_owned(),
        )
}

/// Resolve `TOKIMO_MEDIA_INTELLIGENCE_PYTHON_DIR` for the dev sidecar if unset.
fn resolve_media_intelligence_python_dir() -> Vec<(String, String)> {
    let mut extra_env: Vec<(String, String)> = Vec::new();
    if std::env::var("TOKIMO_MEDIA_INTELLIGENCE_PYTHON_DIR").is_err() {
        let python_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .map(|root| root.join("packages/tokimo-media-intelligence/python"));
        if let Some(dir) = python_dir
            && dir.exists()
        {
            extra_env.push((
                "TOKIMO_MEDIA_INTELLIGENCE_PYTHON_DIR".to_string(),
                dir.to_string_lossy().into_owned(),
            ));
        }
    }
    extra_env
}
