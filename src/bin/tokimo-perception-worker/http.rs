//! HTTP server for ai-worker (used in cross-host deployments).
//!
//! Routes mirror the UDS protocol: `POST /v1/<route>` with an
//! `application/msgpack` body returns a msgpack-encoded `Result<T, RpcError>`.
//! Server-streamed routes return a length-prefixed frame stream in the body.
//! Bidirectional streams (STT) go over `/v1/stt/stream` WebSocket.

use std::sync::Arc;

use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::ws::WebSocketUpgrade;
use axum::extract::{Path as AxPath, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use tokimo_perception::AiService;
use tokimo_perception::worker::protocol::routes;
use tokimo_perception::worker::protocol::types as wire;
use tokio::sync::mpsc;
use tokio_util::io::ReaderStream;

use crate::dispatch;
use crate::supervisor::WorkerSignal;

#[derive(Clone)]
struct HttpState {
    ai: Arc<AiService>,
    sig: mpsc::Sender<WorkerSignal>,
}

pub fn router(ai: Arc<AiService>, sig: mpsc::Sender<WorkerSignal>) -> Router {
    let st = HttpState { ai, sig };
    Router::new()
        .route("/v1/{*route}", post(handle_unary_or_stream))
        .route("/v1/stt/stream", get(ws_stt_stream))
        .with_state(st)
}

async fn handle_unary_or_stream(State(st): State<HttpState>, AxPath(route): AxPath<String>, body: Bytes) -> Response {
    let full_route = format!("/v1/{route}");
    let _ = st.sig.send(WorkerSignal::Activity).await;

    if full_route == routes::ENSURE_CATEGORY {
        let (tx, rx) = mpsc::channel::<tokimo_perception::worker::protocol::RpcResult<wire::ProgressFrame>>(32);
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

async fn ws_stt_stream(ws: WebSocketUpgrade, State(_st): State<HttpState>) -> Response {
    ws.on_upgrade(move |_socket| async move {
        // TODO: bridge WebSocket binary frames to the streaming STT driver.
        // The UDS path is used in the default single-host deployment; HTTP/WS
        // bidirectional STT is required only for split deployments and is
        // deferred to a follow-up change.
    })
}

// Unused helper reference — keeps `ReaderStream` in scope for future streaming bodies.
#[allow(dead_code)]
fn _keep_reader_stream<R: tokio::io::AsyncRead + Unpin>(r: R) -> ReaderStream<R> {
    ReaderStream::new(r)
}
