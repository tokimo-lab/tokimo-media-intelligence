//! Bidirectional streaming STT handler over UDS.
//!
//! Design: `SendOnlineStream` is non-Sync, so it stays inside a single driver
//! task. A reader task forwards client frames through an mpsc channel.
#![cfg(unix)]

use std::sync::Arc;
use std::time::Duration;

use tokimo_perception::AiService;
use tokimo_perception::worker::protocol::error::{RpcError, RpcResult};
use tokimo_perception::worker::protocol::frame::{read_frame, write_frame};
use tokimo_perception::worker::protocol::types as wire;
use tokio::net::unix::{OwnedReadHalf, OwnedWriteHalf};
use tokio::sync::mpsc;
use tokio::time::timeout;

use crate::supervisor::WorkerSignal;

enum DriverMsg {
    Audio(Vec<i16>),
    Reset,
    Close,
}

pub async fn handle(
    ai: Arc<AiService>,
    mut r: OwnedReadHalf,
    mut w: OwnedWriteHalf,
    sig: mpsc::Sender<WorkerSignal>,
) -> RpcResult<()> {
    let svc = ai.streaming_stt().await.map_err(RpcError::Internal)?;
    let (drv_tx, mut drv_rx) = mpsc::channel::<DriverMsg>(64);

    let reader_tx = drv_tx.clone();
    let reader = tokio::spawn(async move {
        loop {
            let frame: wire::SttClientFrame = match read_frame(&mut r).await {
                Ok(f) => f,
                Err(_) => break,
            };
            let msg = match frame {
                wire::SttClientFrame::Audio { pcm } => DriverMsg::Audio(pcm),
                wire::SttClientFrame::Reset => DriverMsg::Reset,
                wire::SttClientFrame::Close => DriverMsg::Close,
            };
            if reader_tx.send(msg).await.is_err() {
                break;
            }
        }
        let _ = reader_tx.send(DriverMsg::Close).await;
    });

    let stream = svc.create_stream();
    let mut last_partial = String::new();

    let drive_result: RpcResult<()> = async {
        loop {
            let mut closed = false;
            // Drain any queued audio up to a short budget, then decode.
            loop {
                match timeout(Duration::from_millis(40), drv_rx.recv()).await {
                    Ok(Some(DriverMsg::Audio(pcm))) => {
                        let samples: Vec<f32> = pcm.iter().map(|&s| f32::from(s) / 32768.0).collect();
                        svc.feed_audio(&stream, &samples);
                    }
                    Ok(Some(DriverMsg::Reset)) => {
                        svc.reset(&stream);
                        last_partial.clear();
                    }
                    Ok(Some(DriverMsg::Close) | None) => {
                        closed = true;
                        break;
                    }
                    Err(_) => break,
                }
            }

            if svc.decode_if_ready(&stream) {
                if let Some(text) = svc.get_partial(&stream)
                    && text != last_partial
                {
                    last_partial.clone_from(&text);
                    write_frame(&mut w, &wire::SttServerFrame::Partial { text }).await?;
                }
                if svc.is_endpoint(&stream) {
                    let final_text = std::mem::take(&mut last_partial);
                    if !final_text.is_empty() {
                        write_frame(&mut w, &wire::SttServerFrame::Final { text: final_text }).await?;
                    }
                    write_frame(&mut w, &wire::SttServerFrame::Endpoint).await?;
                    svc.reset(&stream);
                }
            }

            if closed {
                break Ok(());
            }
        }
    }
    .await;

    reader.abort();
    let _ = sig.send(WorkerSignal::Activity).await;
    drive_result
}
