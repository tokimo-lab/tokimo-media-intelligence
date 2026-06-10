//! Manages the embedded Python OCR sidecar (tokimo-python-ocr / FastAPI).
//!
//! The sidecar is a separate Python process started on demand via `uv run uvicorn`.
//! The first spawn may take a long time (uv syncs Python deps including torch).
//! Subsequent spawns in a pre-synced environment are fast (~2s).
//!
//! The child process inherits our stderr and is killed when the worker exits.

use std::collections::HashSet;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{Mutex, RwLock};
use tokio::time::{sleep, timeout};

/// A running Python sidecar instance.
struct Running {
    child: Child,
    url: String,
}

/// Manages the Python OCR sidecar process lifecycle.
///
/// `ensure_running` is idempotent: if a live instance already exists it is reused;
/// otherwise a fresh `uv run uvicorn` is spawned.
pub struct SidecarManager {
    python_dir: PathBuf,
    models_dir: String,
    state: Mutex<Option<Running>>,
    /// Slugs (Python sidecar side, e.g. "got-ocr-2") of models confirmed ready
    /// during this worker session. Populated by `run_sidecar_download` on
    /// success so `build_catalog` can reflect the ready state without having
    /// to async-probe the Python sidecar on every request.
    ready_models: RwLock<HashSet<String>>,
}

impl SidecarManager {
    pub fn new(python_dir: PathBuf, models_dir: String) -> Arc<Self> {
        Arc::new(Self {
            python_dir,
            models_dir,
            state: Mutex::new(None),
            ready_models: RwLock::new(HashSet::new()),
        })
    }

    /// Returns the base URL of a running sidecar, spawning one if necessary.
    ///
    /// First invocation in a fresh env may block several minutes while `uv sync`
    /// installs Python dependencies. Subsequent calls return immediately if the
    /// previous child is still alive.
    pub async fn ensure_running(&self) -> Result<String, String> {
        let mut guard = self.state.lock().await;

        // Fast path: existing healthy child.
        if let Some(running) = guard.as_mut() {
            match running.child.try_wait() {
                Ok(None) => return Ok(running.url.clone()),
                Ok(Some(status)) => {
                    tracing::warn!("python sidecar exited ({status}); respawning");
                }
                Err(e) => {
                    tracing::warn!("python sidecar wait failed: {e}; respawning");
                }
            }
            *guard = None;
        }

        if !self.python_dir.exists() {
            return Err(format!(
                "python sidecar directory missing: {}",
                self.python_dir.display()
            ));
        }

        let running = spawn(&self.python_dir, &self.models_dir).await?;
        let url = running.url.clone();
        *guard = Some(running);
        Ok(url)
    }

    /// Gracefully stop the sidecar (best-effort, non-blocking kill).
    pub async fn shutdown(&self) {
        let mut guard = self.state.lock().await;
        if let Some(mut running) = guard.take()
            && let Err(e) = running.child.start_kill()
        {
            tracing::warn!("python sidecar kill failed: {e}");
        }
        self.ready_models.write().await.clear();
    }

    /// Mark a sidecar model slug as ready (called by download driver on success).
    pub async fn mark_ready(&self, slug: &str) {
        self.ready_models.write().await.insert(slug.to_string());
    }

    /// Check whether a sidecar model slug is currently known-ready.
    pub async fn is_ready(&self, slug: &str) -> bool {
        self.ready_models.read().await.contains(slug)
    }

    /// Mark a sidecar model slug as unloaded.
    pub async fn mark_unloaded(&self, slug: &str) {
        self.ready_models.write().await.remove(slug);
    }
}

impl Drop for SidecarManager {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.state.try_lock()
            && let Some(mut running) = guard.take()
        {
            let _ = running.child.start_kill();
        }
    }
}

/// Max time for the `uvicorn` process to bind its listening port and print the banner.
/// First run may do `uv sync` which downloads GB-scale wheels — hence generous.
const STARTUP_TIMEOUT: Duration = Duration::from_mins(10);
/// Max time for `/health` to respond after startup banner.
const HEALTH_TIMEOUT: Duration = Duration::from_mins(1);

async fn spawn(python_dir: &PathBuf, models_dir: &str) -> Result<Running, String> {
    tracing::info!(
        "starting python OCR sidecar: uv run uvicorn (cwd={})",
        python_dir.display()
    );

    let mut child = Command::new("uv")
        .current_dir(python_dir)
        .args([
            "run",
            "--",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--log-level",
            "info",
        ])
        .env("DATA_LOCAL_PATH", derive_data_local_path(models_dir))
        .env("AI_MODELS_DIR", models_dir)
        // Python sidecar's `app.config.Settings` reads the `MODELS_DIR` env var
        // (pydantic-settings, case-insensitive). Without it the default
        // `/data/models` is used, which fails with "Permission denied: /data"
        // outside of Docker.
        .env("MODELS_DIR", models_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| format!("failed to spawn `uv run uvicorn`: {e} (is uv installed?)"))?;

    let stdout = child.stdout.take().ok_or("child stdout missing")?;
    let stderr = child.stderr.take().ok_or("child stderr missing")?;

    // Drain stdout in the background (uvicorn logs to stderr; stdout should be
    // empty but must still be drained so the pipe buffer doesn't fill).
    tokio::spawn(async move {
        let mut lines = BufReader::new(stdout).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            tracing::info!(target: "python-ocr", "{line}");
        }
    });

    // Parse the bind address from uvicorn's stderr (that's where uvicorn logs).
    // uvicorn prints: "INFO:     Uvicorn running on http://127.0.0.1:54321 (Press CTRL+C to quit)"
    let url = timeout(STARTUP_TIMEOUT, read_uvicorn_url(stderr))
        .await
        .map_err(|_| "timed out waiting for uvicorn startup banner".to_string())??;

    // Now health-probe to make sure FastAPI is ready.
    probe_health(&url).await?;

    tracing::info!("python sidecar ready at {url}");
    Ok(Running { child, url })
}

/// `DATA_LOCAL_PATH` is consumed by `app/config.py` to derive the models directory.
/// `models_dir` is usually `<DATA_LOCAL_PATH>/perception` (or legacy `ai-models`),
/// so stripping the suffix gives the root.
fn derive_data_local_path(models_dir: &str) -> String {
    let p = std::path::Path::new(models_dir);
    if p.ends_with("perception") || p.ends_with("ai-models") {
        p.parent()
            .map_or_else(|| models_dir.to_string(), |p| p.display().to_string())
    } else {
        models_dir.to_string()
    }
}

async fn read_uvicorn_url(stderr: tokio::process::ChildStderr) -> Result<String, String> {
    let mut lines = BufReader::new(stderr).lines();
    loop {
        match lines.next_line().await {
            Ok(Some(line)) => {
                tracing::info!(target: "python-ocr", "{line}");
                if let Some(url) = extract_url(&line) {
                    // Continue draining stdout in background so the pipe buffer doesn't fill up.
                    tokio::spawn(async move {
                        while let Ok(Some(l)) = lines.next_line().await {
                            tracing::info!(target: "python-ocr", "{l}");
                        }
                    });
                    return Ok(url);
                }
            }
            Ok(None) => return Err("uvicorn exited before printing bind URL".into()),
            Err(e) => return Err(format!("reading uvicorn stderr failed: {e}")),
        }
    }
}

fn extract_url(line: &str) -> Option<String> {
    // Matches "Uvicorn running on http://127.0.0.1:12345"
    let idx = line.find("Uvicorn running on ")?;
    let rest = &line[idx + "Uvicorn running on ".len()..];
    let url_end = rest.find(|c: char| c.is_whitespace()).unwrap_or(rest.len());
    let url = rest[..url_end].trim().trim_end_matches('/');
    if url.starts_with("http") {
        Some(url.to_string())
    } else {
        None
    }
}

async fn probe_health(url: &str) -> Result<(), String> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .map_err(|e| format!("reqwest build: {e}"))?;
    let health_url = format!("{url}/health");
    let deadline = std::time::Instant::now() + HEALTH_TIMEOUT;
    loop {
        if let Ok(resp) = client.get(&health_url).send().await
            && resp.status().is_success()
        {
            return Ok(());
        }
        if std::time::Instant::now() > deadline {
            return Err(format!("python sidecar health probe timed out: {health_url}"));
        }
        sleep(Duration::from_millis(500)).await;
    }
}
