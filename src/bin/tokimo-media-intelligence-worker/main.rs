//! `tokimo-media-intelligence-worker` — sidecar process hosting tokimo-media-intelligence out of the main
//! server's address space, so AI model memory can be reclaimed physically by
//! exiting the worker on idle.

#![allow(
    clippy::match_same_arms,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::needless_pass_by_value,
    clippy::too_many_lines,
    clippy::manual_filter
)]

mod catalog;
mod convert;
mod dispatch;
mod http;
#[cfg(unix)]
mod stt_stream;
mod supervisor;
#[cfg(unix)]
mod uds;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::Parser;
use supervisor::WorkerSignal;
use tokimo_media_intelligence::{MediaIntelligenceService, config::MediaIntelligenceConfig};
use tokio::net::TcpListener;
use tokio::sync::mpsc;

#[derive(Parser, Debug)]
#[command(name = "tokimo-media-intelligence-worker", version)]
struct Args {
    /// UDS path to listen on. If omitted, UDS listener is disabled.
    #[arg(long)]
    socket: Option<PathBuf>,

    /// Optional HTTP listener (e.g. `0.0.0.0:5679`).
    #[arg(long)]
    http: Option<String>,

    /// Model directory (overrides TOKIMO_DATA_LOCAL_PATH-derived default).
    #[arg(long)]
    models_dir: Option<PathBuf>,

    /// Idle seconds before graceful self-exit. `0` disables self-exit
    /// (the parent supervisor will manage lifecycle).
    #[arg(long, default_value_t = 0)]
    idle_exit_secs: u64,

    #[arg(long, default_value = "false")]
    disable_ocr: bool,
    #[arg(long, default_value = "false")]
    disable_clip: bool,
    #[arg(long, default_value = "false")]
    disable_face: bool,
    #[arg(long, default_value = "false")]
    disable_stt: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    if args.socket.is_none() && args.http.is_none() {
        anyhow::bail!("must specify --socket or --http (or both)");
    }

    let mut config = MediaIntelligenceConfig::default();
    if let Some(d) = &args.models_dir {
        config.models_dir = d.display().to_string();
    }
    config.enable_ocr = !args.disable_ocr;
    config.enable_clip = !args.disable_clip;
    config.enable_face = !args.disable_face;
    config.enable_stt = !args.disable_stt;

    let ai = MediaIntelligenceService::new(config);
    ai.start_idle_eviction();

    let (sig_tx, mut sig_rx) = mpsc::channel::<WorkerSignal>(256);

    if let Some(sock) = args.socket.clone() {
        #[cfg(unix)]
        {
            let ai = Arc::clone(&ai);
            let tx = sig_tx.clone();
            tokio::spawn(async move {
                if let Err(e) = uds::serve(sock, ai, tx).await {
                    tracing::error!("UDS listener failed: {e}");
                }
            });
        }
        #[cfg(not(unix))]
        {
            let _ = sock;
            anyhow::bail!("--socket (UDS) is not supported on this platform; use --http instead");
        }
    }

    if let Some(addr) = args.http.clone() {
        let ai = Arc::clone(&ai);
        let tx = sig_tx.clone();
        tokio::spawn(async move {
            match TcpListener::bind(&addr).await {
                Ok(listener) => {
                    tracing::info!("ai-worker HTTP listening on {addr}");
                    let app = http::router(ai, tx);
                    if let Err(e) = axum::serve(listener, app).await {
                        tracing::error!("HTTP server error: {e}");
                    }
                }
                Err(e) => tracing::error!("HTTP bind {addr} failed: {e}"),
            }
        });
    }

    // Main idle/shutdown loop.
    let mut last_activity = Instant::now();
    let idle_exit = if args.idle_exit_secs == 0 {
        None
    } else {
        Some(Duration::from_secs(args.idle_exit_secs))
    };

    loop {
        let deadline = idle_exit.map(|d| last_activity + d);
        let sleep_for = deadline.map_or(Duration::from_mins(1), |dl| {
            dl.saturating_duration_since(Instant::now())
        });
        tokio::select! {
            Some(sig) = sig_rx.recv() => match sig {
                WorkerSignal::Activity => last_activity = Instant::now(),
                WorkerSignal::Shutdown => {
                    tracing::info!("ai-worker received shutdown RPC, exiting");
                    break;
                }
            },
            () = tokio::time::sleep(sleep_for) => {
                if let Some(d) = idle_exit
                    && last_activity.elapsed() >= d
                {
                    tracing::info!(
                        "ai-worker idle for {}s, exiting",
                        last_activity.elapsed().as_secs()
                    );
                    break;
                }
            }
        }
    }

    // Clean up socket file if any.
    if let Some(sock) = args.socket {
        let _ = tokio::fs::remove_file(&sock).await;
    }
    Ok(())
}
