//! Transport abstraction: UDS (local) or HTTP (cross-machine).
//!
//! Exposes three operation shapes that cover all RPC patterns:
//!
//! 1. `call<Req, Res>` — unary request/response (OCR, CLIP embed, face detect, …)
//! 2. `call_stream<Req, Item>` — unary request, streaming response (model download progress)
//! 3. `open_bidi<C, S>` — full-duplex stream (streaming STT)
//!
//! Both transports speak rmp-serde on the wire:
//! - **UDS**: one `UnixStream` per call; header "OP $route\n" followed by framed payload(s).
//!   Simple, direct; no persistent connection pooling needed at this scale.
//! - **HTTP**: `POST {route}` with `application/msgpack` body; streaming uses
//!   length-prefixed frames concatenated in the body (same codec as UDS).

use std::sync::Arc;

use futures_util::{Stream, StreamExt};
use serde::{Serialize, de::DeserializeOwned};
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::sync::mpsc;

use super::error::{RpcError, RpcResult};
use super::frame;

/// Sender/receiver pair for a bidirectional stream.
pub struct BidiStream<C, S>
where
    C: Serialize + Send + Sync + 'static,
    S: DeserializeOwned + Send + 'static,
{
    pub tx: mpsc::Sender<C>,
    pub rx: mpsc::Receiver<RpcResult<S>>,
}

#[async_trait::async_trait]
pub trait Transport: Send + Sync + 'static {
    /// Unary call.
    async fn call<Req, Res>(&self, route: &str, req: &Req) -> RpcResult<Res>
    where
        Req: Serialize + Sync,
        Res: DeserializeOwned + Send;

    /// Unary call with streaming response (server-sent frames).
    /// Returns a channel receiver of `Result<Item, RpcError>`.
    async fn call_stream<Req, Item>(&self, route: &str, req: &Req) -> RpcResult<mpsc::Receiver<RpcResult<Item>>>
    where
        Req: Serialize + Sync,
        Item: DeserializeOwned + Send + 'static;

    /// Full-duplex stream.
    async fn open_bidi<C, S>(&self, route: &str) -> RpcResult<BidiStream<C, S>>
    where
        C: Serialize + Send + Sync + 'static,
        S: DeserializeOwned + Send + 'static;
}

// ---------------- UDS transport (Unix-only) ----------------

#[cfg(unix)]
#[derive(Clone)]
pub struct UdsTransport {
    socket_path: Arc<std::path::PathBuf>,
}

#[cfg(unix)]
impl UdsTransport {
    pub fn new(socket_path: impl Into<std::path::PathBuf>) -> Self {
        Self {
            socket_path: Arc::new(socket_path.into()),
        }
    }

    pub fn socket_path(&self) -> &std::path::Path {
        &self.socket_path
    }

    async fn connect(&self) -> RpcResult<tokio::net::UnixStream> {
        tokio::net::UnixStream::connect(self.socket_path.as_ref())
            .await
            .map_err(|e| RpcError::Transport(format!("uds connect {}: {e}", self.socket_path.display())))
    }
}

/// Windows stub: UDS is not supported on this platform. Calls to any method
/// return `RpcError::Transport`. Construction is allowed so that consumers can
/// still compile and choose an alternate transport (HTTP) at runtime.
#[cfg(not(unix))]
#[derive(Clone)]
pub struct UdsTransport {
    socket_path: Arc<std::path::PathBuf>,
}

#[cfg(not(unix))]
impl UdsTransport {
    pub fn new(socket_path: impl Into<std::path::PathBuf>) -> Self {
        Self {
            socket_path: Arc::new(socket_path.into()),
        }
    }

    pub fn socket_path(&self) -> &std::path::Path {
        &self.socket_path
    }
}

/// On-wire line that announces the operation. One of:
///   "CALL /v1/ocr\n"
///   "SSTREAM /v1/models/ensure_category\n"  (server-streamed response)
///   "BIDI /v1/stt/stream\n"
pub async fn write_header<W: AsyncWrite + Unpin>(w: &mut W, kind: &str, route: &str) -> RpcResult<()> {
    use tokio::io::AsyncWriteExt;
    let line = format!("{kind} {route}\n");
    w.write_all(line.as_bytes()).await?;
    Ok(())
}

pub async fn read_header<R: AsyncRead + Unpin>(r: &mut R) -> RpcResult<(String, String)> {
    // NOTE: must NOT use BufReader here — its read-ahead buffer would swallow
    // the bytes of the following length-prefixed frame and they'd be lost when
    // the BufReader is dropped. Read one byte at a time up to '\n'. The header
    // is short (a few dozen bytes) and only read once per connection, so the
    // extra syscalls are irrelevant.
    use tokio::io::AsyncReadExt;
    let mut line: Vec<u8> = Vec::with_capacity(64);
    let mut byte = [0u8; 1];
    loop {
        match r.read_exact(&mut byte).await {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof && line.is_empty() => {
                return Err(RpcError::Transport("eof before header".into()));
            }
            Err(e) => return Err(e.into()),
        }
        if byte[0] == b'\n' {
            break;
        }
        line.push(byte[0]);
        if line.len() > 1024 {
            return Err(RpcError::Transport("header too long".into()));
        }
    }
    let line = std::str::from_utf8(&line).map_err(|e| RpcError::Transport(format!("header not utf8: {e}")))?;
    let mut parts = line.splitn(2, ' ');
    let kind = parts.next().unwrap_or("").to_string();
    let route = parts.next().unwrap_or("").to_string();
    if kind.is_empty() || route.is_empty() {
        return Err(RpcError::Transport(format!("bad header: {line:?}")));
    }
    Ok((kind, route))
}

#[cfg(unix)]
#[async_trait::async_trait]
impl Transport for UdsTransport {
    async fn call<Req, Res>(&self, route: &str, req: &Req) -> RpcResult<Res>
    where
        Req: Serialize + Sync,
        Res: DeserializeOwned + Send,
    {
        let mut s = self.connect().await?;
        write_header(&mut s, "CALL", route).await?;
        frame::write_frame(&mut s, req).await?;
        let res: RpcResult<Res> = frame::read_frame::<_, Result<Res, RpcError>>(&mut s).await?;
        res
    }

    async fn call_stream<Req, Item>(&self, route: &str, req: &Req) -> RpcResult<mpsc::Receiver<RpcResult<Item>>>
    where
        Req: Serialize + Sync,
        Item: DeserializeOwned + Send + 'static,
    {
        let mut s = self.connect().await?;
        write_header(&mut s, "SSTREAM", route).await?;
        frame::write_frame(&mut s, req).await?;
        let (tx, rx) = mpsc::channel(32);
        tokio::spawn(async move {
            loop {
                match frame::read_frame_opt::<_, Result<Item, RpcError>>(&mut s).await {
                    Ok(Some(Ok(item))) => {
                        if tx.send(Ok(item)).await.is_err() {
                            break;
                        }
                    }
                    Ok(Some(Err(e))) => {
                        let _ = tx.send(Err(e)).await;
                        break;
                    }
                    Ok(None) => break, // clean EOF
                    Err(e) => {
                        let _ = tx.send(Err(e)).await;
                        break;
                    }
                }
            }
        });
        Ok(rx)
    }

    async fn open_bidi<C, S>(&self, route: &str) -> RpcResult<BidiStream<C, S>>
    where
        C: Serialize + Send + Sync + 'static,
        S: DeserializeOwned + Send + 'static,
    {
        let stream = self.connect().await?;
        let (mut r, mut w) = stream.into_split();
        write_header(&mut w, "BIDI", route).await?;

        let (client_tx, mut client_rx) = mpsc::channel::<C>(32);
        let (server_tx, server_rx) = mpsc::channel::<RpcResult<S>>(32);

        // client → socket
        tokio::spawn(async move {
            while let Some(msg) = client_rx.recv().await {
                if let Err(e) = frame::write_frame(&mut w, &msg).await {
                    tracing::debug!("bidi send error: {e}");
                    break;
                }
            }
        });
        // socket → server_tx
        tokio::spawn(async move {
            loop {
                match frame::read_frame_opt::<_, Result<S, RpcError>>(&mut r).await {
                    Ok(Some(Ok(item))) => {
                        if server_tx.send(Ok(item)).await.is_err() {
                            break;
                        }
                    }
                    Ok(Some(Err(e))) => {
                        let _ = server_tx.send(Err(e)).await;
                        break;
                    }
                    Ok(None) => break,
                    Err(e) => {
                        let _ = server_tx.send(Err(e)).await;
                        break;
                    }
                }
            }
        });

        Ok(BidiStream {
            tx: client_tx,
            rx: server_rx,
        })
    }
}

#[cfg(not(unix))]
#[async_trait::async_trait]
impl Transport for UdsTransport {
    async fn call<Req, Res>(&self, _route: &str, _req: &Req) -> RpcResult<Res>
    where
        Req: Serialize + Sync,
        Res: DeserializeOwned + Send,
    {
        Err(RpcError::Transport("UDS transport not supported on Windows".into()))
    }

    async fn call_stream<Req, Item>(&self, _route: &str, _req: &Req) -> RpcResult<mpsc::Receiver<RpcResult<Item>>>
    where
        Req: Serialize + Sync,
        Item: DeserializeOwned + Send + 'static,
    {
        Err(RpcError::Transport("UDS transport not supported on Windows".into()))
    }

    async fn open_bidi<C, S>(&self, _route: &str) -> RpcResult<BidiStream<C, S>>
    where
        C: Serialize + Send + Sync + 'static,
        S: DeserializeOwned + Send + 'static,
    {
        Err(RpcError::Transport("UDS transport not supported on Windows".into()))
    }
}

// ---------------- HTTP transport ----------------
//
// HTTP is used for cross-machine mode. Unary calls are straightforward POSTs with
// rmp bodies. Streaming uses the same frame codec (length-prefixed rmp) concatenated
// into the response body for server-streamed responses. Bidirectional is deferred
// (would need HTTP/2 or WebSocket); phase 1 returns Unimplemented.

#[derive(Clone)]
pub struct HttpTransport {
    base_url: Arc<String>,
    client: reqwest::Client,
}

impl HttpTransport {
    pub fn new(base_url: impl Into<String>) -> RpcResult<Self> {
        let client = reqwest::Client::builder()
            .http1_only()
            .pool_max_idle_per_host(4)
            .build()
            .map_err(|e| RpcError::Transport(format!("http client: {e}")))?;
        Ok(Self {
            base_url: Arc::new(base_url.into().trim_end_matches('/').to_string()),
            client,
        })
    }
}

#[async_trait::async_trait]
impl Transport for HttpTransport {
    async fn call<Req, Res>(&self, route: &str, req: &Req) -> RpcResult<Res>
    where
        Req: Serialize + Sync,
        Res: DeserializeOwned + Send,
    {
        let url = format!("{}{route}", self.base_url);
        let body = rmp_serde::to_vec_named(req)?;
        let resp = self
            .client
            .post(&url)
            .header("content-type", "application/msgpack")
            .body(body)
            .send()
            .await?;
        let status = resp.status();
        let bytes = resp.bytes().await?;
        if !status.is_success() {
            // Try to decode as RpcError; fall back to raw.
            if let Ok(err) = rmp_serde::from_slice::<RpcError>(&bytes) {
                return Err(err);
            }
            return Err(RpcError::Transport(format!(
                "http {status}: {}",
                String::from_utf8_lossy(&bytes)
            )));
        }
        let res: RpcResult<Res> = rmp_serde::from_slice::<Result<Res, RpcError>>(&bytes)?;
        res
    }

    async fn call_stream<Req, Item>(&self, route: &str, req: &Req) -> RpcResult<mpsc::Receiver<RpcResult<Item>>>
    where
        Req: Serialize + Sync,
        Item: DeserializeOwned + Send + 'static,
    {
        let url = format!("{}{route}", self.base_url);
        let body = rmp_serde::to_vec_named(req)?;
        let resp = self
            .client
            .post(&url)
            .header("content-type", "application/msgpack")
            .header("accept", "application/msgpack-frames")
            .body(body)
            .send()
            .await?
            .error_for_status()?;

        let (tx, rx) = mpsc::channel(32);
        let mut byte_stream = resp.bytes_stream();

        tokio::spawn(async move {
            // Accumulate bytes and parse length-prefixed rmp frames.
            let mut buf: Vec<u8> = Vec::new();
            while let Some(chunk) = byte_stream.next().await {
                match chunk {
                    Ok(c) => buf.extend_from_slice(&c),
                    Err(e) => {
                        let _ = tx.send(Err(RpcError::Transport(e.to_string()))).await;
                        return;
                    }
                }
                loop {
                    if buf.len() < 4 {
                        break;
                    }
                    let len = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
                    if buf.len() < 4 + len {
                        break;
                    }
                    let payload = buf[4..4 + len].to_vec();
                    buf.drain(..4 + len);
                    match rmp_serde::from_slice::<Result<Item, RpcError>>(&payload) {
                        Ok(Ok(item)) => {
                            if tx.send(Ok(item)).await.is_err() {
                                return;
                            }
                        }
                        Ok(Err(e)) => {
                            let _ = tx.send(Err(e)).await;
                            return;
                        }
                        Err(e) => {
                            let _ = tx.send(Err(RpcError::Transport(format!("decode: {e}")))).await;
                            return;
                        }
                    }
                }
            }
        });
        Ok(rx)
    }

    async fn open_bidi<C, S>(&self, _route: &str) -> RpcResult<BidiStream<C, S>>
    where
        C: Serialize + Send + Sync + 'static,
        S: DeserializeOwned + Send + 'static,
    {
        Err(RpcError::Transport(
            "bidirectional streaming over HTTP not implemented (use UDS or deploy with WS gateway)".into(),
        ))
    }
}

/// Helper: wrap any `Stream<Item=RpcResult<T>>` into an mpsc receiver.
pub fn stream_to_receiver<T: Send + 'static>(
    mut s: impl Stream<Item = RpcResult<T>> + Unpin + Send + 'static,
) -> mpsc::Receiver<RpcResult<T>> {
    let (tx, rx) = mpsc::channel(32);
    tokio::spawn(async move {
        while let Some(item) = s.next().await {
            if tx.send(item).await.is_err() {
                break;
            }
        }
    });
    rx
}

// ---------------- Dyn-friendly wrapper ----------------
//
// The [`Transport`] trait has generic methods, so `dyn Transport` is not
// object-safe. Clients that want to store "some transport" at runtime use
// this enum instead.

#[derive(Clone)]
pub enum AnyTransport {
    Uds(UdsTransport),
    Http(HttpTransport),
}

impl AnyTransport {
    pub async fn call<Req, Res>(&self, route: &str, req: &Req) -> RpcResult<Res>
    where
        Req: Serialize + Sync,
        Res: DeserializeOwned + Send,
    {
        match self {
            Self::Uds(t) => t.call(route, req).await,
            Self::Http(t) => t.call(route, req).await,
        }
    }

    pub async fn call_stream<Req, Item>(&self, route: &str, req: &Req) -> RpcResult<mpsc::Receiver<RpcResult<Item>>>
    where
        Req: Serialize + Sync,
        Item: DeserializeOwned + Send + 'static,
    {
        match self {
            Self::Uds(t) => t.call_stream(route, req).await,
            Self::Http(t) => t.call_stream(route, req).await,
        }
    }

    pub async fn open_bidi<C, S>(&self, route: &str) -> RpcResult<BidiStream<C, S>>
    where
        C: Serialize + Send + Sync + 'static,
        S: DeserializeOwned + Send + 'static,
    {
        match self {
            Self::Uds(t) => t.open_bidi(route).await,
            Self::Http(t) => t.open_bidi(route).await,
        }
    }
}
