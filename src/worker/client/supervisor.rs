//! Worker process supervisor.
//!
//! Responsible for:
//! - **Lazy spawn**: the first RPC call (via `ensure_up`) launches the worker
//!   binary. Subsequent calls are no-ops while the process is healthy.
//! - **Health check**: pings the worker over the transport after spawn.
//! - **Idle exit**: a background task sends `/v1/shutdown` when no activity
//!   has been recorded for `idle_secs`.
//! - **Auto-respawn**: if the child exits (voluntarily or crash), the next
//!   `ensure_up` call starts a fresh process.
//!
//! In `Remote` mode no child is spawned and `ensure_up` is a no-op — the
//! remote worker's lifecycle is managed externally (k8s, systemd, …).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use tokio::process::{Child, Command};
use tokio::sync::Mutex as AsyncMutex;

use crate::worker::protocol::routes;
use crate::worker::protocol::transport::AnyTransport;
use crate::worker::protocol::types as wire;
use crate::worker::protocol::{RpcError, RpcResult};

#[derive(Debug, Clone)]
pub struct SupervisorConfig {
    /// Path to the `tokimo-perception-worker` binary.
    pub worker_binary: PathBuf,
    /// UDS path the worker listens on.
    pub socket_path: PathBuf,
    /// Optional HTTP bind (empty string disables HTTP in the worker).
    pub http_addr: Option<String>,
    /// Models directory to pass through to the worker.
    pub models_dir: Option<PathBuf>,
    /// Idle shutdown threshold. 0 = disabled (keepalive forever).
    pub idle_secs: u32,
    /// Extra environment variables for the spawned process.
    pub extra_env: Vec<(String, String)>,
    /// If true, `ensure_up` is a no-op — worker is managed externally.
    pub remote: bool,
}

impl SupervisorConfig {
    pub fn remote_only() -> Self {
        Self {
            worker_binary: PathBuf::new(),
            socket_path: PathBuf::new(),
            http_addr: None,
            models_dir: None,
            idle_secs: 0,
            extra_env: Vec::new(),
            remote: true,
        }
    }
}

pub struct Supervisor {
    cfg: SupervisorConfig,
    transport: Arc<AnyTransport>,
    state: AsyncMutex<SpawnState>,
    last_activity: Mutex<Instant>,
    /// Debounce: tracks the last time kill_and_respawn successfully killed the child.
    last_kill_at: Mutex<Option<Instant>>,
}

struct SpawnState {
    child: Option<Child>,
    /// monotonic generation counter; incremented on each (re)spawn.
    generation: u64,
}

impl Supervisor {
    pub fn new(cfg: SupervisorConfig, transport: Arc<AnyTransport>) -> Arc<Self> {
        let sup = Arc::new(Self {
            cfg,
            transport,
            state: AsyncMutex::new(SpawnState {
                child: None,
                generation: 0,
            }),
            last_activity: Mutex::new(Instant::now()),
            last_kill_at: Mutex::new(None),
        });
        sup.spawn_idle_watchdog();
        sup
    }

    pub fn mark_activity(&self) {
        *self.last_activity.lock() = Instant::now();
    }

    /// Idempotent: make sure the worker process is alive and responsive.
    pub async fn ensure_up(self: &Arc<Self>) -> RpcResult<()> {
        if self.cfg.remote {
            return Ok(());
        }

        let mut st = self.state.lock().await;

        // If a child handle exists, see whether it's still alive.
        if let Some(child) = st.child.as_mut() {
            match child.try_wait() {
                Ok(Some(_status)) => {
                    tracing::warn!("ai-worker exited; will respawn");
                    st.child = None;
                }
                Ok(None) => {
                    // Still running — optimistically trust it. `call` will
                    // surface a transport error if the socket is gone.
                    return Ok(());
                }
                Err(e) => {
                    tracing::warn!("ai-worker try_wait failed: {e}");
                    st.child = None;
                }
            }
        }

        // Spawn.
        let mut cmd = Command::new(&self.cfg.worker_binary);
        cmd.arg("--socket").arg(&self.cfg.socket_path);
        if let Some(http) = &self.cfg.http_addr
            && !http.is_empty()
        {
            cmd.arg("--http").arg(http);
        }
        if let Some(models) = &self.cfg.models_dir {
            cmd.arg("--models-dir").arg(models);
        }
        cmd.arg("--idle-exit-secs").arg(self.cfg.idle_secs.to_string());

        for (k, v) in &self.cfg.extra_env {
            cmd.env(k, v);
        }

        // Best-effort: make sure the old socket is gone so bind() doesn't fail.
        let _ = std::fs::remove_file(&self.cfg.socket_path);

        let mut child = cmd
            .kill_on_drop(true)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| RpcError::Transport(format!("spawn ai-worker: {e}")))?;

        // Forward worker stdout/stderr into the parent's tracing subscriber so
        // `/api/dev/logs` picks them up. Without this, worker logs only land on
        // the parent's inherited stderr and are invisible from the HTTP API.
        if let Some(stdout) = child.stdout.take() {
            tokio::spawn(async move {
                use tokio::io::{AsyncBufReadExt, BufReader};
                let mut lines = BufReader::new(stdout).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    tracing::info!(target: "ai_worker", "{line}");
                }
            });
        }
        if let Some(stderr) = child.stderr.take() {
            tokio::spawn(async move {
                use tokio::io::{AsyncBufReadExt, BufReader};
                let mut lines = BufReader::new(stderr).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    tracing::info!(target: "ai_worker", "{line}");
                }
            });
        }

        st.child = Some(child);
        st.generation = st.generation.wrapping_add(1);

        // Wait for the socket + /v1/ping to come up. MUST happen while still
        // holding `st` — otherwise concurrent callers who acquire the lock
        // between spawn and ready would see `child = Some(...)`, optimistically
        // return Ok, and then try to connect to a socket that the worker
        // hasn't bound yet (ENOENT). This race is how a queue burst of OCR
        // jobs can lose ~all but the first when the worker is cold-starting.
        self.wait_ready().await?;
        drop(st);

        *self.last_activity.lock() = Instant::now();
        Ok(())
    }

    async fn wait_ready(&self) -> RpcResult<()> {
        let deadline = Instant::now() + Duration::from_secs(30);
        let mut delay = Duration::from_millis(50);
        loop {
            if let Ok(_pong) = self.transport.call::<_, wire::Pong>(routes::PING, &()).await {
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(RpcError::Transport("ai-worker did not become ready in 30s".into()));
            }
            tokio::time::sleep(delay).await;
            delay = (delay * 2).min(Duration::from_secs(1));
        }
    }

    /// Hard-kill the current worker child and let the next RPC respawn it.
    /// Used as a last-resort escalation when cooperative cancel didn't stop
    /// the in-flight inference in time. Debounced to 2 s — rapid successive
    /// calls collapse into one kill.
    pub async fn kill_and_respawn(self: &Arc<Self>) -> RpcResult<()> {
        if self.cfg.remote {
            return Ok(());
        }
        // Debounce: skip if we already killed within the last 2 s.
        {
            let last = self.last_kill_at.lock();
            if last.is_some_and(|t: Instant| t.elapsed() < Duration::from_secs(2)) {
                return Ok(());
            }
        }
        *self.last_kill_at.lock() = Some(Instant::now());

        let mut st = self.state.lock().await;
        if let Some(mut child) = st.child.take() {
            tracing::warn!("ai-worker: hard-killing child (cancel escalation)");
            let _ = child.start_kill();
            let _ = child.wait().await;
        }
        st.generation = st.generation.wrapping_add(1);
        drop(st);
        // The next RPC will trigger ensure_up() → respawn.
        Ok(())
    }

    fn spawn_idle_watchdog(self: &Arc<Self>) {
        if self.cfg.remote || self.cfg.idle_secs == 0 {
            return;
        }
        let weak = Arc::downgrade(self);
        let idle = Duration::from_secs(u64::from(self.cfg.idle_secs));
        tokio::spawn(async move {
            let tick = Duration::from_secs(10).min(idle / 3).max(Duration::from_secs(1));
            loop {
                tokio::time::sleep(tick).await;
                let Some(this) = weak.upgrade() else { break };
                let idle_for = Instant::now().saturating_duration_since(*this.last_activity.lock());
                if idle_for < idle {
                    continue;
                }
                // Try graceful shutdown.
                let mut st = this.state.lock().await;
                if st.child.is_none() {
                    continue;
                }
                drop(st);
                tracing::info!("ai-worker idle for {}s, sending shutdown", idle_for.as_secs());
                let _: RpcResult<wire::ShutdownResponse> = this.transport.call(routes::SHUTDOWN, &()).await;
                // Reap the child.
                st = this.state.lock().await;
                if let Some(mut child) = st.child.take() {
                    // Give it up to 5s to exit, then kill.
                    let wait = tokio::time::timeout(Duration::from_secs(5), child.wait()).await;
                    if wait.is_err() {
                        tracing::warn!("ai-worker did not exit in 5s; killing");
                        let _ = child.start_kill();
                        let _ = child.wait().await;
                    }
                }
            }
        });
    }
}
