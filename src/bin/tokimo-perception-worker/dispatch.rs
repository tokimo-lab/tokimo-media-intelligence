//! Core RPC dispatch: given a route + (decoded) request bytes, invoke the
//! matching AiService method and return response bytes.
//!
//! Shared by both the UDS listener and the HTTP axum router.

use std::sync::Arc;

use serde::Serialize;
use tokimo_perception::AiService;
use tokimo_perception::worker::protocol::error::{RpcError, RpcResult};
use tokimo_perception::worker::protocol::routes;
use tokimo_perception::worker::protocol::types as wire;
use tokio::sync::mpsc;

use crate::catalog;
use crate::convert;

fn map_err(e: String) -> RpcError {
    let lower = e.to_lowercase();
    if lower.contains("disabled") {
        RpcError::Disabled(e)
    } else if lower.contains("not found") || lower.contains("missing") {
        RpcError::NotFound(e)
    } else if lower.contains("not ready") || lower.contains("not present") {
        RpcError::ModelNotReady(e)
    } else {
        RpcError::Internal(e)
    }
}

fn decode<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> RpcResult<T> {
    rmp_serde::from_slice(bytes).map_err(Into::into)
}

fn encode<T: Serialize>(v: &T) -> RpcResult<Vec<u8>> {
    rmp_serde::to_vec_named(v).map_err(Into::into)
}

/// Run a unary RPC. Returns rmp-encoded `Result<Res, RpcError>`.
pub async fn dispatch_unary(ai: &Arc<AiService>, route: &str, req_bytes: &[u8]) -> Vec<u8> {
    let result = unary_inner(ai, route, req_bytes).await;
    // Always encode a Result<Value, RpcError>; even errors go through as Ok-encoded outer frame.
    match &result {
        Ok(b) => b.clone(),
        Err(e) => encode::<RpcResult<()>>(&Err(e.clone())).unwrap_or_default(),
    }
}

async fn unary_inner(ai: &Arc<AiService>, route: &str, req_bytes: &[u8]) -> RpcResult<Vec<u8>> {
    match route {
        routes::PING => encode::<RpcResult<wire::Pong>>(&Ok(wire::Pong {
            version: env!("CARGO_PKG_VERSION").to_string(),
            accel_provider: convert::accel_to_wire(tokimo_perception::active_ep()),
        })),
        routes::INFO => {
            let cfg = ai.config();
            let st = ai.status();
            let info = wire::WorkerInfo {
                accel_provider: convert::accel_to_wire(st.accel_provider),
                models_dir: cfg.models_dir.clone(),
                ocr_det_max_side: cfg.ocr_det_max_side,
                ocr_enabled: ai.is_ocr_enabled(),
                clip_enabled: ai.is_clip_enabled(),
                face_enabled: ai.is_face_enabled(),
                stt_enabled: ai.is_stt_enabled(),
                models_ready: ai.models_ready(),
                ocr_models_ready: ai.ocr_models_ready(),
                ocr_server_models_ready: ai.ocr_server_models_ready(),
                ocr_mobile_models_ready: ai.ocr_mobile_models_ready(),
                clip_models_ready: ai.clip_models_ready(),
                face_models_ready: ai.face_models_ready(),
                stt_model_ready: ai.stt_model_ready(),
                streaming_stt_model_ready: ai.streaming_stt_model_ready(),
                ocr_loaded: st.ocr_loaded,
                clip_loaded: st.clip_loaded,
                face_loaded: st.face_loaded,
                stt_loaded: st.stt_loaded,
            };
            encode::<RpcResult<wire::WorkerInfo>>(&Ok(info))
        }
        routes::STATUS => encode::<RpcResult<wire::AiStatus>>(&Ok(convert::status_to_wire(ai.status()))),
        routes::LIST_OCR_MODELS => {
            let list: Vec<wire::OcrModelInfo> = ai
                .ocr_available_models()
                .into_iter()
                .map(convert::ocr_model_info_to_wire)
                .collect();
            encode::<RpcResult<Vec<wire::OcrModelInfo>>>(&Ok(list))
        }
        routes::STT_MODELS_STATUS => {
            let list: Vec<wire::SttModelStatus> = ai
                .stt_models_status()
                .into_iter()
                .map(convert::stt_status_to_wire)
                .collect();
            encode::<RpcResult<Vec<wire::SttModelStatus>>>(&Ok(list))
        }
        routes::OCR => {
            let req: wire::OcrRequest = decode(req_bytes)?;
            let out = ai
                .ocr(&req.image, req.model.as_deref(), req.request_id.as_deref())
                .await
                .map_err(map_err)?;
            let items: Vec<_> = out.into_iter().map(convert::ocr_item_to_wire).collect();
            encode::<RpcResult<Vec<wire::OcrItem>>>(&Ok(items))
        }
        routes::OCR_HYBRID => {
            let req: wire::OcrHybridRequest = decode(req_bytes)?;
            let det = req
                .det_model
                .as_deref()
                .ok_or_else(|| RpcError::BadRequest("det_model required".into()))?;
            let vlm = req
                .vlm_model
                .as_deref()
                .ok_or_else(|| RpcError::BadRequest("vlm_model required".into()))?;
            let (items, _debug) = ai
                .ocr_hybrid(&req.image, det, vlm, req.request_id.as_deref())
                .await
                .map_err(map_err)?;
            let items: Vec<_> = items.into_iter().map(convert::ocr_item_to_wire).collect();
            encode::<RpcResult<Vec<wire::OcrItem>>>(&Ok(items))
        }
        routes::CLIP_IMAGE => {
            let req: wire::ClipImageRequest = decode(req_bytes)?;
            let v = ai
                .clip_image(&req.image, req.request_id.as_deref())
                .await
                .map_err(map_err)?;
            encode::<RpcResult<Vec<f32>>>(&Ok(v))
        }
        routes::CLIP_TEXT => {
            let req: wire::ClipTextRequest = decode(req_bytes)?;
            let v = ai
                .clip_text(&req.text, req.request_id.as_deref())
                .await
                .map_err(map_err)?;
            encode::<RpcResult<Vec<f32>>>(&Ok(v))
        }
        routes::CLIP_CLASSIFY => {
            let req: wire::ClipClassifyRequest = decode(req_bytes)?;
            let tags = ai.clip_classify(&req.vector).await.map_err(map_err)?;
            let tags: Vec<_> = tags.into_iter().map(convert::tag_to_wire).collect();
            encode::<RpcResult<Vec<wire::TagResult>>>(&Ok(tags))
        }
        routes::FACE_DETECT => {
            let req: wire::FaceRequest = decode(req_bytes)?;
            let out = ai
                .detect_faces(&req.image, req.request_id.as_deref())
                .await
                .map_err(map_err)?;
            let faces: Vec<_> = out.into_iter().map(convert::face_to_wire).collect();
            encode::<RpcResult<Vec<wire::FaceDetection>>>(&Ok(faces))
        }
        routes::CANCEL => {
            let req: wire::CancelRequest = decode(req_bytes)?;
            let cancelled = tokimo_perception::cancel::cancel_inflight(&req.request_id);
            encode::<RpcResult<wire::CancelResponse>>(&Ok(wire::CancelResponse { cancelled }))
        }
        routes::STT_TRANSCRIBE => {
            let req: wire::SttTranscribeRequest = decode(req_bytes)?;
            let text = ai.transcribe_audio(&req.wav).await.map_err(map_err)?;
            encode::<RpcResult<wire::SttTranscribeResponse>>(&Ok(wire::SttTranscribeResponse { text }))
        }
        routes::STT_TRANSCRIBE_PCM => {
            let req: wire::SttTranscribePcmRequest = decode(req_bytes)?;
            let text = ai.transcribe_pcm(req.samples, req.sample_rate).await.map_err(map_err)?;
            encode::<RpcResult<wire::SttTranscribeResponse>>(&Ok(wire::SttTranscribeResponse { text }))
        }
        routes::SHUTDOWN => {
            // Caller handles the actual exit; we just ack.
            encode::<RpcResult<wire::ShutdownResponse>>(&Ok(wire::ShutdownResponse { ok: true }))
        }
        routes::CATALOG => {
            let req: wire::CatalogRequest = decode(req_bytes).unwrap_or(wire::CatalogRequest { languages: Vec::new() });
            let cat = catalog::build_catalog(ai, &req).await;
            encode::<RpcResult<wire::ModelCatalog>>(&Ok(cat))
        }
        routes::MODEL_UNLOAD => {
            let req: wire::ModelActionRequest = decode(req_bytes)?;
            if let catalog::ModelRoute::Sidecar(slug) = catalog::route_for(&req.model_id) {
                let url = ai.sidecar().ensure_running().await.map_err(map_err)?;
                let client = reqwest::Client::new();
                let _ = client
                    .post(format!("{}/models/{slug}/unload", url.trim_end_matches('/')))
                    .timeout(std::time::Duration::from_secs(30))
                    .send()
                    .await;
                ai.sidecar().mark_unloaded(&slug).await;
            }
            encode::<RpcResult<wire::ShutdownResponse>>(&Ok(wire::ShutdownResponse { ok: true }))
        }
        other => Err(RpcError::NotFound(format!("route not found: {other}"))),
    }
}

/// Streaming response: write progress frames to `tx`.
/// Used for model download endpoints.
pub fn dispatch_server_stream(
    ai: Arc<AiService>,
    route: &str,
    req_bytes: &[u8],
    tx: mpsc::Sender<RpcResult<wire::ProgressFrame>>,
) {
    let route = route.to_string();
    let req_bytes = req_bytes.to_vec();
    tokio::spawn(async move {
        let res = server_stream_inner(&ai, &route, &req_bytes, tx.clone()).await;
        if let Err(e) = res {
            let _ = tx.send(Err(e)).await;
        }
    });
}

async fn server_stream_inner(
    ai: &Arc<AiService>,
    route: &str,
    req_bytes: &[u8],
    tx: mpsc::Sender<RpcResult<wire::ProgressFrame>>,
) -> RpcResult<()> {
    match route {
        routes::ENSURE_CATEGORY => {
            let req: wire::EnsureCategoryRequest = decode(req_bytes)?;
            let tx_clone = tx.clone();
            // ProgressFn is a boxed async closure — see tokimo-perception::models::ProgressFn
            let progress: tokimo_perception::models::ProgressFn = Box::new(move |file, status, pct, dl, total| {
                let frame = wire::ProgressFrame::Progress {
                    file_name: file.to_string(),
                    status: status.to_string(),
                    percent: u32::from(pct),
                    downloaded_bytes: dl,
                    total_bytes: total,
                };
                // Non-blocking: drop progress updates if channel is full/closed.
                let _ = tx_clone.try_send(Ok(frame));
            });
            match req.category {
                Some(c) => {
                    let cat = convert::category_from_wire(c);
                    ai.ensure_category_with_progress(cat, progress).await.map_err(map_err)?;
                }
                None => {
                    ai.ensure_models_with_progress(progress).await.map_err(map_err)?;
                }
            }
            let _ = tx.send(Ok(wire::ProgressFrame::Done)).await;
            Ok(())
        }
        routes::DOWNLOAD_STT => {
            let req: wire::DownloadSttRequest = decode(req_bytes)?;
            let tx_clone = tx.clone();
            let model_id_clone = req.model_id.clone();
            let progress = move |status: &str, pct: u8| {
                let frame = wire::ProgressFrame::Progress {
                    file_name: model_id_clone.clone(),
                    status: status.to_string(),
                    percent: u32::from(pct),
                    downloaded_bytes: 0,
                    total_bytes: 0,
                };
                let _ = tx_clone.try_send(Ok(frame));
            };
            ai.download_stt_model(&req.model_id, progress).await.map_err(map_err)?;
            let _ = tx.send(Ok(wire::ProgressFrame::Done)).await;
            Ok(())
        }
        routes::MODEL_DOWNLOAD => {
            let req: wire::ModelActionRequest = decode(req_bytes)?;
            let model_id = req.model_id.clone();
            let route = catalog::route_for(&model_id);
            tracing::debug!(%model_id, "worker: MODEL_DOWNLOAD entered");
            let tx_cat = tx.clone();
            let progress_native: tokimo_perception::models::ProgressFn =
                Box::new(move |file, status, pct, dl, total| {
                    // Pass the real per-file name so the rust-server side can
                    // aggregate multi-file downloads (e.g. RapidOCR = det + rec)
                    // into a single monotonic progress bar. Collapsing all files
                    // into one key would make the bar reset each time a new file
                    // starts.
                    let frame = wire::ProgressFrame::Progress {
                        file_name: file.to_string(),
                        status: status.to_string(),
                        percent: u32::from(pct),
                        downloaded_bytes: dl,
                        total_bytes: total,
                    };
                    let _ = tx_cat.try_send(Ok(frame));
                });
            match route {
                catalog::ModelRoute::OcrServer => {
                    ai.ensure_category_with_progress(
                        tokimo_perception::models::ModelCategory::OcrServer,
                        progress_native,
                    )
                    .await
                    .map_err(map_err)?;
                }
                catalog::ModelRoute::OcrMobile => {
                    ai.ensure_category_with_progress(
                        tokimo_perception::models::ModelCategory::OcrMobile,
                        progress_native,
                    )
                    .await
                    .map_err(map_err)?;
                }
                catalog::ModelRoute::Clip => {
                    ai.ensure_category_with_progress(tokimo_perception::models::ModelCategory::Clip, progress_native)
                        .await
                        .map_err(map_err)?;
                }
                catalog::ModelRoute::Face => {
                    ai.ensure_category_with_progress(tokimo_perception::models::ModelCategory::Face, progress_native)
                        .await
                        .map_err(map_err)?;
                }
                catalog::ModelRoute::Stt(slug) => {
                    let tx_stt = tx.clone();
                    let mid = model_id.clone();
                    let progress_stt = move |status: &str, pct: u8| {
                        let frame = wire::ProgressFrame::Progress {
                            file_name: mid.clone(),
                            status: status.to_string(),
                            percent: u32::from(pct),
                            downloaded_bytes: 0,
                            total_bytes: 0,
                        };
                        let _ = tx_stt.try_send(Ok(frame));
                    };
                    ai.download_stt_model(&slug, progress_stt).await.map_err(map_err)?;
                }
                catalog::ModelRoute::Sidecar(slug) => {
                    tracing::debug!(%model_id, %slug, "worker: entering run_sidecar_download");
                    run_sidecar_download(ai, &slug, &model_id, &tx).await?;
                    tracing::debug!(%model_id, %slug, "worker: run_sidecar_download returned Ok");
                }
                catalog::ModelRoute::Unknown => {
                    return Err(RpcError::BadRequest(format!("unknown model id: {model_id}")));
                }
            }
            let _ = tx.send(Ok(wire::ProgressFrame::Done)).await;
            Ok(())
        }
        other => Err(RpcError::NotFound(format!("stream route not found: {other}"))),
    }
}

/// Drive a sidecar model download: ensure Python sidecar is running, POST
/// `/models/{slug}/load`, then poll `/models` emitting `ProgressFrame`s. Mirrors
/// the status → progress shape used for native category downloads so the
/// upstream aggregator treats both paths identically.
async fn run_sidecar_download(
    ai: &Arc<AiService>,
    slug: &str,
    model_id: &str,
    tx: &mpsc::Sender<RpcResult<wire::ProgressFrame>>,
) -> RpcResult<()> {
    tracing::debug!(%slug, %model_id, "run_sidecar_download: calling ensure_running");
    let base = ai.sidecar().ensure_running().await.map_err(map_err)?;
    tracing::debug!(%slug, %base, "run_sidecar_download: sidecar base url");
    let base = base.trim_end_matches('/').to_string();
    let client = reqwest::Client::new();

    // Kick off background load
    tracing::debug!(%slug, "run_sidecar_download: POSTing /models/{slug}/load");
    let resp = client
        .post(format!("{base}/models/{slug}/load"))
        .timeout(std::time::Duration::from_secs(30))
        .send()
        .await
        .map_err(|e| RpcError::Internal(format!("sidecar load request failed: {e}")))?;
    tracing::debug!(%slug, status = %resp.status(), "run_sidecar_download: POST returned");
    if !resp.status().is_success() {
        let text = resp.text().await.unwrap_or_default();
        return Err(RpcError::Internal(format!("sidecar load error: {text}")));
    }

    // Emit initial "downloading 0%" so the aggregator flips state immediately
    let _ = tx.try_send(Ok(wire::ProgressFrame::Progress {
        file_name: model_id.to_string(),
        status: "downloading".into(),
        percent: 0,
        downloaded_bytes: 0,
        total_bytes: 0,
    }));

    let deadline = std::time::Instant::now() + std::time::Duration::from_mins(30);
    loop {
        if std::time::Instant::now() > deadline {
            return Err(RpcError::Internal("sidecar download timed out after 30 minutes".into()));
        }
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        let resp = match client
            .get(format!("{base}/models"))
            .timeout(std::time::Duration::from_secs(10))
            .send()
            .await
        {
            Ok(r) if r.status().is_success() => r,
            _ => continue,
        };
        let list: Vec<serde_json::Value> = match resp.json().await {
            Ok(v) => v,
            Err(_) => continue,
        };
        let Some(model) = list
            .into_iter()
            .find(|m| m.get("id").and_then(|v| v.as_str()) == Some(slug))
        else {
            continue;
        };
        let status = model.get("status").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let (percent, downloaded, total) = model.get("progress").map_or((0, 0, 0), |p| {
            let pct = p
                .get("percent")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0)
                .clamp(0.0, 100.0) as u32;
            let dl = p
                .get("downloaded_bytes")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0);
            let tot = p.get("total_bytes").and_then(serde_json::Value::as_u64).unwrap_or(0);
            (pct, dl, tot)
        });

        match status.as_str() {
            "ready" => {
                ai.sidecar().mark_ready(slug).await;
                return Ok(());
            }
            "error" => {
                let msg = model
                    .get("error_message")
                    .and_then(|v| v.as_str())
                    .unwrap_or("sidecar model load failed")
                    .to_string();
                return Err(RpcError::Internal(msg));
            }
            _ => {
                let _ = tx.try_send(Ok(wire::ProgressFrame::Progress {
                    file_name: model_id.to_string(),
                    status: "downloading".into(),
                    percent,
                    downloaded_bytes: downloaded,
                    total_bytes: total,
                }));
            }
        }
    }
}
