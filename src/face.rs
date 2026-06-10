/// Face detection + embedding using `InsightFace` ONNX models.
/// SCRFD (`det_10g`) for detection + `ArcFace` (`w600k_r50`) for recognition.
/// Produces 512-dim face embeddings.
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;

use image::DynamicImage;
use ndarray::Array4;
use ort::{session::Session, value::Tensor};

pub struct FaceDetection {
    pub embedding: Vec<f32>,
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
    pub confidence: f32,
}

pub struct FaceService {
    det_session: Mutex<Session>,
    rec_session: Mutex<Session>,
}

impl FaceService {
    pub fn new(models_dir: &str) -> Result<Self, String> {
        let det_path = format!("{models_dir}/face/det_10g.onnx");
        let rec_path = format!("{models_dir}/face/w600k_r50.onnx");

        if !Path::new(&det_path).exists() {
            return Err(format!("Face detection model not found: {det_path}"));
        }
        if !Path::new(&rec_path).exists() {
            return Err(format!("Face recognition model not found: {rec_path}"));
        }

        tracing::info!("Loading face detection model (SCRFD)...");
        let det_session = crate::build_session(&det_path).map_err(|e| format!("Load face det model: {e}"))?;

        tracing::info!("Loading face recognition model (ArcFace)...");
        let rec_session = crate::build_session(&rec_path).map_err(|e| format!("Load face rec model: {e}"))?;

        tracing::info!("Face service ready.");
        Ok(Self {
            det_session: Mutex::new(det_session),
            rec_session: Mutex::new(rec_session),
        })
    }

    /// Detect faces and extract embeddings.
    pub async fn detect_faces(&self, img: &DynamicImage) -> Result<Vec<FaceDetection>, String> {
        let (orig_w, orig_h) = (img.width(), img.height());

        let raw_faces = self.detect(img).await?;
        if raw_faces.is_empty() {
            return Ok(vec![]);
        }

        let mut results = Vec::new();
        for face in &raw_faces {
            let embedding = self.extract_embedding(img, face).await?;
            results.push(FaceDetection {
                embedding,
                x: face.x.max(0),
                y: face.y.max(0),
                w: face.w.min(orig_w as i32 - face.x.max(0)),
                h: face.h.min(orig_h as i32 - face.y.max(0)),
                confidence: face.confidence,
            });
        }

        Ok(results)
    }

    /// Run SCRFD face detection.
    async fn detect(&self, img: &DynamicImage) -> Result<Vec<RawFace>, String> {
        let target_size: usize = 640;
        let (orig_w, orig_h) = (img.width() as f32, img.height() as f32);
        let scale = (target_size as f32 / orig_w.max(orig_h)).min(1.0);
        let new_w = (orig_w * scale) as u32;
        let new_h = (orig_h * scale) as u32;

        let resized = img.resize_exact(new_w, new_h, image::imageops::FilterType::CatmullRom);
        let rgb = resized.to_rgb8();

        // Pad to target_size × target_size
        let mut tensor = Array4::<f32>::zeros((1, 3, target_size, target_size));
        let mean = [127.5f32, 127.5, 127.5];
        let std_val = 128.0f32;
        for y in 0..new_h as usize {
            for x in 0..new_w as usize {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    tensor[[0, c, y, x]] = (f32::from(pixel[c]) - mean[c]) / std_val;
                }
            }
        }

        let input_tensor = Tensor::from_array(tensor).map_err(|e| format!("Create tensor: {e}"))?;
        let options = Arc::new(ort::session::RunOptions::new().map_err(|e| format!("RunOptions: {e}"))?);
        // Bail early if cancel arrived before this session started.
        if crate::cancel::CANCEL_ID
            .try_with(Clone::clone)
            .ok()
            .flatten()
            .as_deref()
            .is_some_and(crate::cancel::is_cancelled)
        {
            return Err("cancelled before inference".into());
        }
        let _cancel_guard = crate::cancel::register_current(&options);
        let faces = {
            let mut session = self.det_session.lock().await;
            let outputs = session
                .run_async(ort::inputs![input_tensor], &*options)
                .map_err(|e| format!("Face det run_async: {e}"))?
                .await
                .map_err(|e| format!("Face detection inference: {e}"))?;
            parse_scrfd_outputs(&outputs, scale, 0.5)?
        };
        let faces = nms(faces, 0.4);

        Ok(faces)
    }

    /// Extract 512-dim `ArcFace` embedding from a cropped face.
    async fn extract_embedding(&self, img: &DynamicImage, face: &RawFace) -> Result<Vec<f32>, String> {
        // Crop face region with some margin
        let margin = 10;
        let x = (face.x - margin).max(0) as u32;
        let y = (face.y - margin).max(0) as u32;
        let w = (face.w + margin * 2) as u32;
        let h = (face.h + margin * 2) as u32;
        let w = w.min(img.width().saturating_sub(x));
        let h = h.min(img.height().saturating_sub(y));
        if w < 2 || h < 2 {
            return Ok(vec![0.0; 512]);
        }

        let cropped = img.crop_imm(x, y, w, h);

        // Resize to 112×112 (ArcFace standard input)
        let resized = cropped.resize_exact(112, 112, image::imageops::FilterType::CatmullRom);
        let rgb = resized.to_rgb8();

        // Normalize: (pixel - 127.5) / 127.5
        let mut tensor = Array4::<f32>::zeros((1, 3, 112, 112));
        for y in 0..112usize {
            for x in 0..112usize {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    tensor[[0, c, y, x]] = (f32::from(pixel[c]) - 127.5) / 127.5;
                }
            }
        }

        let input_tensor = Tensor::from_array(tensor).map_err(|e| format!("Create tensor: {e}"))?;
        let options = Arc::new(ort::session::RunOptions::new().map_err(|e| format!("RunOptions: {e}"))?);
        // Bail early if cancel arrived before this session started.
        if crate::cancel::CANCEL_ID
            .try_with(Clone::clone)
            .ok()
            .flatten()
            .as_deref()
            .is_some_and(crate::cancel::is_cancelled)
        {
            return Err("cancelled before inference".into());
        }
        let _cancel_guard = crate::cancel::register_current(&options);
        let raw: Vec<f32> = {
            let mut session = self.rec_session.lock().await;
            let outputs = session
                .run_async(ort::inputs![input_tensor], &*options)
                .map_err(|e| format!("Face rec run_async: {e}"))?
                .await
                .map_err(|e| format!("Face rec inference: {e}"))?;
            let (_shape, data) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("Extract embedding: {e}"))?;
            data.to_vec()
        };
        let norm: f32 = raw.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            Ok(raw.iter().map(|v| v / norm).collect())
        } else {
            Ok(raw)
        }
    }
}

// ---------------------------------------------------------------------------
// SCRFD output parsing
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct RawFace {
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    confidence: f32,
}

fn parse_scrfd_outputs(
    outputs: &ort::session::SessionOutputs,
    scale: f32,
    score_threshold: f32,
) -> Result<Vec<RawFace>, String> {
    // SCRFD det_10g has 9 outputs: 3 score maps + 3 bbox maps + 3 landmark maps
    // Each group corresponds to stride 8, 16, 32
    let strides = [8usize, 16, 32];
    let feat_size: usize = 640;
    let mut faces = Vec::new();

    for (i, &stride) in strides.iter().enumerate() {
        let score_idx = i;
        let bbox_idx = i + 3;

        if score_idx >= outputs.len() || bbox_idx >= outputs.len() {
            continue;
        }

        let (score_shape, score_data) = outputs[score_idx]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Extract score {i}: {e}"))?;

        let (bbox_shape, bbox_data) = outputs[bbox_idx]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Extract bbox {i}: {e}"))?;

        if score_shape.len() < 2 || bbox_shape.len() < 2 {
            continue;
        }

        // Shape is [N, 1] for scores, [N, 4] for bboxes — num_anchors is dim 0
        let num_anchors = score_shape[0] as usize;
        let feat_h = feat_size / stride;
        let feat_w = feat_size / stride;
        let num_anchors_per_pos = num_anchors.checked_div(feat_h * feat_w).unwrap_or(2);
        let bbox_cols = bbox_shape[1] as usize;

        for idx in 0..num_anchors {
            let score = score_data.get(idx).copied().unwrap_or(0.0);
            if score < score_threshold {
                continue;
            }

            let pos = idx / num_anchors_per_pos;
            let anchor_y = (pos / feat_w) * stride;
            let anchor_x = (pos % feat_w) * stride;

            if bbox_cols < 4 {
                continue;
            }

            // bbox format: distance to left, top, right, bottom from anchor point
            let bbox_offset = idx * bbox_cols;
            let dl = bbox_data.get(bbox_offset).copied().unwrap_or(0.0) * stride as f32;
            let dt = bbox_data.get(bbox_offset + 1).copied().unwrap_or(0.0) * stride as f32;
            let dr = bbox_data.get(bbox_offset + 2).copied().unwrap_or(0.0) * stride as f32;
            let db = bbox_data.get(bbox_offset + 3).copied().unwrap_or(0.0) * stride as f32;

            // InsightFace uses grid position (top-left of cell), not center
            let ax = anchor_x as f32;
            let ay = anchor_y as f32;

            let x1 = ((ax - dl) / scale) as i32;
            let y1 = ((ay - dt) / scale) as i32;
            let x2 = ((ax + dr) / scale) as i32;
            let y2 = ((ay + db) / scale) as i32;

            faces.push(RawFace {
                x: x1.max(0),
                y: y1.max(0),
                w: (x2 - x1).max(1),
                h: (y2 - y1).max(1),
                confidence: score,
            });
        }
    }

    Ok(faces)
}

/// Non-Maximum Suppression.
fn nms(mut faces: Vec<RawFace>, iou_threshold: f32) -> Vec<RawFace> {
    faces.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

    let mut keep = Vec::new();
    let mut suppressed = vec![false; faces.len()];

    for i in 0..faces.len() {
        if suppressed[i] {
            continue;
        }
        keep.push(faces[i].clone());

        for j in (i + 1)..faces.len() {
            if suppressed[j] {
                continue;
            }
            if iou(&faces[i], &faces[j]) > iou_threshold {
                suppressed[j] = true;
            }
        }
    }

    keep
}

fn iou(a: &RawFace, b: &RawFace) -> f32 {
    let x1 = a.x.max(b.x) as f32;
    let y1 = a.y.max(b.y) as f32;
    let x2 = ((a.x + a.w).min(b.x + b.w)) as f32;
    let y2 = ((a.y + a.h).min(b.y + b.h)) as f32;

    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area_a = (a.w * a.h) as f32;
    let area_b = (b.w * b.h) as f32;
    let union = area_a + area_b - inter;

    if union > 0.0 { inter / union } else { 0.0 }
}
