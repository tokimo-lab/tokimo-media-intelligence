//! HTTP server for ai-worker (used in cross-host deployments).
//!
//! Routes mirror the UDS protocol: `POST /v1/<route>` with an
//! `application/msgpack` body returns a msgpack-encoded `Result<T, RpcError>`.
//! Server-streamed routes return a length-prefixed frame stream in the body.
//! Bidirectional streams (STT) go over `/v1/stt/stream` WebSocket.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::DefaultBodyLimit;
use axum::extract::ws::WebSocketUpgrade;
use axum::extract::{Path as AxPath, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use tokimo_media_intelligence::MediaIntelligenceService;
use tokimo_media_intelligence::worker::client::Supervisor;
use tokimo_media_intelligence::worker::protocol::RpcError;
use tokimo_media_intelligence::worker::protocol::frame::MAX_FRAME_BYTES;
use tokimo_media_intelligence::worker::protocol::routes;
use tokimo_media_intelligence::worker::protocol::transport::write_header;
use tokimo_media_intelligence::worker::protocol::types as wire;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::sync::mpsc;
use tokio_util::io::ReaderStream;

use crate::dispatch;
use crate::supervisor::WorkerSignal;

#[derive(Clone)]
struct HttpState {
    ai: Arc<MediaIntelligenceService>,
    sig: mpsc::Sender<WorkerSignal>,
}

#[derive(Clone)]
struct ProxyState {
    supervisor: Arc<Supervisor>,
    socket_path: Arc<std::path::PathBuf>,
    active_requests: Arc<AtomicUsize>,
    idle_generation: Arc<AtomicU64>,
    last_completed_at: Arc<Mutex<Instant>>,
    idle_reap_secs: u64,
}

pub fn router(ai: Arc<MediaIntelligenceService>, sig: mpsc::Sender<WorkerSignal>) -> Router {
    let st = HttpState { ai, sig };
    Router::new()
        .route("/v1/{*route}", post(handle_unary_or_stream))
        .route("/v1/stt/stream", get(ws_stt_stream))
        .layer(DefaultBodyLimit::max(MAX_FRAME_BYTES as usize))
        .with_state(st)
}

pub fn proxy_router(supervisor: Arc<Supervisor>, socket_path: std::path::PathBuf, idle_reap_secs: u64) -> Router {
    let st = ProxyState {
        supervisor,
        socket_path: Arc::new(socket_path),
        active_requests: Arc::new(AtomicUsize::new(0)),
        idle_generation: Arc::new(AtomicU64::new(0)),
        last_completed_at: Arc::new(Mutex::new(Instant::now())),
        idle_reap_secs,
    };
    Router::new()
        .route("/v1/{*route}", post(handle_proxy_unary_or_stream))
        .route("/v1/stt/stream", get(ws_stt_stream_proxy))
        .layer(DefaultBodyLimit::max(MAX_FRAME_BYTES as usize))
        .with_state(st)
}

fn is_stream_route(route: &str) -> bool {
    matches!(
        route,
        routes::ENSURE_CATEGORY | routes::DOWNLOAD_STT | routes::MODEL_DOWNLOAD
    )
}

async fn handle_unary_or_stream(State(st): State<HttpState>, AxPath(route): AxPath<String>, body: Bytes) -> Response {
    let full_route = format!("/v1/{route}");
    let _ = st.sig.send(WorkerSignal::Activity).await;

    if is_stream_route(&full_route) {
        let (tx, rx) = mpsc::channel::<tokimo_media_intelligence::worker::protocol::RpcResult<wire::ProgressFrame>>(32);
        dispatch::dispatch_server_stream(Arc::clone(&st.ai), &full_route, &body, tx);
        // Convert mpsc<frame> into a length-prefixed byte stream.
        let stream = async_stream::stream! {
            let mut rx = rx;
            while let Some(item) = rx.recv().await {
                if let Ok(bytes) = rmp_serde::to_vec_named(&item)
                    && let Ok(len) = u32::try_from(bytes.len())
                {
                    let mut out = Vec::with_capacity(4 + bytes.len());
                    out.extend_from_slice(&len.to_be_bytes());
                    out.extend_from_slice(&bytes);
                    yield Ok::<Bytes, std::io::Error>(Bytes::from(out));
                }
            }
        };
        let body = Body::from_stream(stream);
        return Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "application/octet-stream")
            .body(body)
            .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response());
    }

    let resp_bytes = dispatch::dispatch_unary(&st.ai, &full_route, &body).await;
    if full_route == routes::SHUTDOWN {
        let _ = st.sig.send(WorkerSignal::Shutdown).await;
    }
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/msgpack")
        .body(Body::from(resp_bytes))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

async fn handle_proxy_unary_or_stream(
    State(st): State<ProxyState>,
    AxPath(route): AxPath<String>,
    body: Bytes,
) -> Response {
    let full_route = format!("/v1/{route}");
    let guard = ProxyActivityGuard::new(st.clone());
    if let Err(e) = st.supervisor.ensure_up().await {
        return msgpack_error_response(e);
    }
    st.supervisor.mark_activity();

    if is_stream_route(&full_route) {
        let release_guard_on_response = full_route == routes::ENSURE_CATEGORY;
        let stream_guard = if release_guard_on_response {
            drop(guard);
            None
        } else {
            Some(guard)
        };
        let stream = async_stream::stream! {
            let _guard = stream_guard;
            match proxy_stream_raw(&st.socket_path, &full_route, &body).await {
                Ok(mut rx) => {
                    while let Some(item) = rx.recv().await {
                        match item {
                            Ok(bytes) => yield Ok::<Bytes, std::io::Error>(Bytes::from(bytes)),
                            Err(e) => {
                                if let Ok(bytes) = encode_error_frame(e) {
                                    yield Ok::<Bytes, std::io::Error>(Bytes::from(bytes));
                                }
                                break;
                            }
                        }
                    }
                }
                Err(e) => {
                    if let Ok(bytes) = encode_error_frame(e) {
                        yield Ok::<Bytes, std::io::Error>(Bytes::from(bytes));
                    }
                }
            }
        };
        return Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "application/octet-stream")
            .body(Body::from_stream(stream))
            .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response());
    }

    let response = match proxy_unary_raw(&st.socket_path, &full_route, &body).await {
        Ok(resp_bytes) => Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "application/msgpack")
            .body(Body::from(resp_bytes))
            .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response()),
        Err(e) => msgpack_error_response(e),
    };
    drop(guard);
    response
}

struct ProxyActivityGuard {
    state: ProxyState,
}

impl ProxyActivityGuard {
    fn new(state: ProxyState) -> Self {
        state.active_requests.fetch_add(1, Ordering::SeqCst);
        state.idle_generation.fetch_add(1, Ordering::SeqCst);
        state.supervisor.mark_activity();
        Self { state }
    }
}

impl Drop for ProxyActivityGuard {
    fn drop(&mut self) {
        self.state.supervisor.mark_activity();
        if let Ok(mut last) = self.state.last_completed_at.lock() {
            *last = Instant::now();
        }

        self.state.active_requests.fetch_sub(1, Ordering::SeqCst);
        if self.state.idle_reap_secs == 0 {
            return;
        }

        let state = self.state.clone();
        let generation = state.idle_generation.fetch_add(1, Ordering::SeqCst) + 1;
        tokio::spawn(async move {
            let idle = Duration::from_secs(state.idle_reap_secs);
            tokio::time::sleep(idle).await;
            if state.idle_generation.load(Ordering::SeqCst) != generation {
                return;
            }
            let idle_for = state
                .last_completed_at
                .lock()
                .map(|last| last.elapsed())
                .unwrap_or_default();
            if idle_for < idle {
                return;
            }
            let active = state.active_requests.load(Ordering::SeqCst);
            tracing::info!(
                "ai-worker proxy idle for {}s, hard-reaping child (active_requests={active})",
                idle_for.as_secs()
            );
            let _ = state.supervisor.kill_and_respawn().await;
        });
    }
}

async fn ws_stt_stream(ws: WebSocketUpgrade, State(_st): State<HttpState>) -> Response {
    ws.on_upgrade(move |_socket| async move {
        // TODO: bridge WebSocket binary frames to the streaming STT driver.
        // The UDS path is used in the default single-host deployment; HTTP/WS
        // bidirectional STT is required only for split deployments and is
        // deferred to a follow-up change.
    })
}

async fn ws_stt_stream_proxy(ws: WebSocketUpgrade, State(_st): State<ProxyState>) -> Response {
    ws.on_upgrade(move |_socket| async move {
        // HTTP bidirectional STT is still not implemented; proxy mode keeps the
        // existing split-deployment behaviour.
    })
}

fn msgpack_error_response(e: RpcError) -> Response {
    let body = rmp_serde::to_vec_named::<tokimo_media_intelligence::worker::protocol::RpcResult<()>>(&Err(e))
        .unwrap_or_default();
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/msgpack")
        .body(Body::from(body))
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

fn encode_error_frame(e: RpcError) -> Result<Vec<u8>, RpcError> {
    let payload = rmp_serde::to_vec_named::<tokimo_media_intelligence::worker::protocol::RpcResult<wire::ProgressFrame>>(
        &Err(e),
    )?;
    let len = u32::try_from(payload.len()).map_err(|_| RpcError::BadRequest("frame too large".into()))?;
    let mut out = Vec::with_capacity(4 + payload.len());
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(&payload);
    Ok(out)
}

#[cfg(unix)]
async fn proxy_unary_raw(socket_path: &std::path::Path, route: &str, body: &[u8]) -> Result<Vec<u8>, RpcError> {
    let mut s = tokio::net::UnixStream::connect(socket_path).await?;
    write_header(&mut s, "CALL", route).await?;
    write_raw_frame(&mut s, body).await?;
    read_raw_frame(&mut s).await
}

#[cfg(not(unix))]
async fn proxy_unary_raw(_socket_path: &std::path::Path, _route: &str, _body: &[u8]) -> Result<Vec<u8>, RpcError> {
    Err(RpcError::Transport(
        "supervised HTTP proxy requires Unix sockets".into(),
    ))
}

#[cfg(unix)]
async fn proxy_stream_raw(
    socket_path: &std::path::Path,
    route: &str,
    body: &[u8],
) -> Result<mpsc::Receiver<Result<Vec<u8>, RpcError>>, RpcError> {
    let mut s = tokio::net::UnixStream::connect(socket_path).await?;
    write_header(&mut s, "SSTREAM", route).await?;
    write_raw_frame(&mut s, body).await?;
    let (tx, rx) = mpsc::channel(32);
    tokio::spawn(async move {
        loop {
            match read_raw_frame_opt(&mut s).await {
                Ok(Some(payload)) => {
                    let mut out = Vec::with_capacity(4 + payload.len());
                    out.extend_from_slice(&(payload.len() as u32).to_be_bytes());
                    out.extend_from_slice(&payload);
                    if tx.send(Ok(out)).await.is_err() {
                        break;
                    }
                }
                Ok(None) => break,
                Err(e) => {
                    let _ = tx.send(Err(e)).await;
                    break;
                }
            }
        }
    });
    Ok(rx)
}

#[cfg(not(unix))]
async fn proxy_stream_raw(
    _socket_path: &std::path::Path,
    _route: &str,
    _body: &[u8],
) -> Result<mpsc::Receiver<Result<Vec<u8>, RpcError>>, RpcError> {
    Err(RpcError::Transport(
        "supervised HTTP proxy requires Unix sockets".into(),
    ))
}

async fn write_raw_frame<W: AsyncWrite + Unpin>(w: &mut W, bytes: &[u8]) -> Result<(), RpcError> {
    let len = u32::try_from(bytes.len()).map_err(|_| RpcError::BadRequest("frame too large".into()))?;
    if len > MAX_FRAME_BYTES {
        return Err(RpcError::BadRequest(format!("frame too large: {len}")));
    }
    w.write_all(&len.to_be_bytes()).await?;
    w.write_all(bytes).await?;
    w.flush().await?;
    Ok(())
}

async fn read_raw_frame<R: AsyncRead + Unpin>(r: &mut R) -> Result<Vec<u8>, RpcError> {
    let mut len_buf = [0u8; 4];
    r.read_exact(&mut len_buf).await?;
    let len = u32::from_be_bytes(len_buf);
    if len > MAX_FRAME_BYTES {
        return Err(RpcError::BadRequest(format!("frame too large: {len}")));
    }
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf).await?;
    Ok(buf)
}

async fn read_raw_frame_opt<R: AsyncRead + Unpin>(r: &mut R) -> Result<Option<Vec<u8>>, RpcError> {
    let mut len_buf = [0u8; 4];
    match r.read_exact(&mut len_buf).await {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e.into()),
    }
    let len = u32::from_be_bytes(len_buf);
    if len > MAX_FRAME_BYTES {
        return Err(RpcError::BadRequest(format!("frame too large: {len}")));
    }
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf).await?;
    Ok(Some(buf))
}

// Unused helper reference — keeps `ReaderStream` in scope for future streaming bodies.
#[allow(dead_code)]
fn _keep_reader_stream<R: tokio::io::AsyncRead + Unpin>(r: R) -> ReaderStream<R> {
    ReaderStream::new(r)
}
