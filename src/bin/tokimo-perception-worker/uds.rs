//! UDS listener: accepts connections, reads header line, dispatches to handler.
#![cfg(unix)]

use std::sync::Arc;

use tokimo_perception::AiService;
use tokimo_perception::worker::protocol::error::RpcError;
use tokimo_perception::worker::protocol::transport::read_header;
use tokimo_perception::worker::protocol::{frame, types as wire};
use tokio::io::AsyncWriteExt;
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::mpsc;

use crate::dispatch;
use crate::stt_stream;
use crate::supervisor::WorkerSignal;

pub async fn serve(
    socket_path: std::path::PathBuf,
    ai: Arc<AiService>,
    sig: mpsc::Sender<WorkerSignal>,
) -> std::io::Result<()> {
    // Remove stale socket if present.
    let _ = tokio::fs::remove_file(&socket_path).await;
    if let Some(parent) = socket_path.parent() {
        let _ = tokio::fs::create_dir_all(parent).await;
    }
    let listener = UnixListener::bind(&socket_path)?;
    tracing::info!("ai-worker listening on UDS {}", socket_path.display());

    loop {
        let (stream, _) = listener.accept().await?;
        let ai = Arc::clone(&ai);
        let sig = sig.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_conn(stream, ai, sig).await {
                tracing::debug!("uds conn closed: {e}");
            }
        });
    }
}

async fn handle_conn(stream: UnixStream, ai: Arc<AiService>, sig: mpsc::Sender<WorkerSignal>) -> Result<(), RpcError> {
    let (mut r, mut w) = stream.into_split();
    let (kind, route) = read_header(&mut r).await?;
    match kind.as_str() {
        "CALL" => {
            // Read request frame, dispatch, write response.
            let req_bytes: Vec<u8> = read_frame_raw(&mut r).await?;
            let resp_bytes = dispatch::dispatch_unary(&ai, &route, &req_bytes).await;
            write_frame_raw(&mut w, &resp_bytes).await?;
            // Record activity — any RPC counts as a signal to keep the worker alive.
            let _ = sig.send(WorkerSignal::Activity).await;
            // If this was a shutdown call, trigger process exit AFTER ack.
            if route == tokimo_perception::worker::protocol::routes::SHUTDOWN {
                let _ = sig.send(WorkerSignal::Shutdown).await;
            }
        }
        "SSTREAM" => {
            let req_bytes: Vec<u8> = read_frame_raw(&mut r).await?;
            let (tx, mut rx) = mpsc::channel::<tokimo_perception::worker::protocol::RpcResult<wire::ProgressFrame>>(32);
            dispatch::dispatch_server_stream(Arc::clone(&ai), &route, &req_bytes, tx);
            while let Some(item) = rx.recv().await {
                frame::write_frame(&mut w, &item).await?;
            }
            let _ = sig.send(WorkerSignal::Activity).await;
        }
        "BIDI" => {
            if route == tokimo_perception::worker::protocol::routes::STT_STREAM {
                stt_stream::handle(Arc::clone(&ai), r, w, sig.clone()).await?;
            } else {
                return Err(RpcError::NotFound(format!("bidi route: {route}")));
            }
        }
        other => return Err(RpcError::BadRequest(format!("unknown op: {other}"))),
    }
    Ok(())
}

/// Read a length-prefixed raw frame (no rmp decode — we pass bytes through).
async fn read_frame_raw<R: tokio::io::AsyncRead + Unpin>(r: &mut R) -> Result<Vec<u8>, RpcError> {
    use tokio::io::AsyncReadExt;
    let mut len_buf = [0u8; 4];
    r.read_exact(&mut len_buf).await?;
    let len = u32::from_be_bytes(len_buf);
    if len > frame::MAX_FRAME_BYTES {
        return Err(RpcError::BadRequest(format!("frame too large: {len}")));
    }
    let mut buf = vec![0u8; len as usize];
    r.read_exact(&mut buf).await?;
    Ok(buf)
}

async fn write_frame_raw<W: tokio::io::AsyncWrite + Unpin>(w: &mut W, bytes: &[u8]) -> Result<(), RpcError> {
    let len = u32::try_from(bytes.len()).map_err(|_| RpcError::BadRequest("frame too large".into()))?;
    w.write_all(&len.to_be_bytes()).await?;
    w.write_all(bytes).await?;
    w.flush().await?;
    Ok(())
}
