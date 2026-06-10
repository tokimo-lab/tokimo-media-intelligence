/// `PaddleOCR` CTC recognition pipeline.
/// Detection and classification are delegated to `OcrDetector`.
/// Uses ONNX CTC recognition model via ort.
///
/// NOTE: Attention-based recognition (`recognize_text_attention` / `attention_to_position`)
/// is retained below but currently unused — no production-ready attention ONNX model
/// exists yet. When one becomes available, re-enable by loading `attn_rec_session`
/// and routing through `recognize_text_attention` in `recognize_text`.
use std::sync::Arc;

use tokio::sync::Mutex;

use image::DynamicImage;
use ndarray::{Array4, s};
use ort::{session::Session, value::Tensor};

use crate::ocr_backend::{OcrBackend, PaddleOcrVariant};
use crate::ocr_detector::{self, OcrDetector, TextBox};

static OCR_KEYS_V5: &str = include_str!("../config/ppocr_keys_v5.txt");

/// Detection strategy for text box extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectionMode {
    /// Connected component analysis with axis-aligned bounding boxes (original).
    Components,
    /// Contour-based detection with rotated bounding boxes (RapidOCR-style).
    Contours,
}

pub struct OcrItem {
    pub text: String,
    pub score: f32,
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    /// Rotation angle in degrees (clockwise). 0 = horizontal text.
    pub angle: f32,
    /// 4 corner points [TL, TR, BR, BL] in original image coords.
    /// Present when detected via contour mode; `None` for legacy/sidecar results.
    pub corners: Option<[(f32, f32); 4]>,
    /// Paragraph group index assigned by spatial clustering.
    pub paragraph_id: u32,
    /// Character-level positions within the bounding box (original image pixel coords).
    /// Each entry is `(x_offset, width)` relative to the block's left edge.
    /// Derived from CTC timestep alignment or attention maps.
    /// `None` when positions are unavailable (e.g. VLM-only mode).
    pub char_positions: Option<Vec<(f32, f32)>>,
}

pub struct OcrService {
    detector: OcrDetector,
    rec_session: Mutex<Session>,
    char_dict: Vec<String>,
    variant_name: String,
    detection_mode: DetectionMode,
}

impl OcrService {
    pub fn new(models_dir: &str, variant: PaddleOcrVariant) -> Result<Self, String> {
        Self::new_with_options(models_dir, variant, DetectionMode::Components, None)
    }

    pub fn new_with_mode(models_dir: &str, variant: PaddleOcrVariant, mode: DetectionMode) -> Result<Self, String> {
        Self::new_with_options(models_dir, variant, mode, None)
    }

    pub fn new_with_options(
        models_dir: &str,
        variant: PaddleOcrVariant,
        mode: DetectionMode,
        det_max_side: Option<u32>,
    ) -> Result<Self, String> {
        let (rec_name, variant_name) = match variant {
            PaddleOcrVariant::Mobile => ("PP-OCRv5_mobile_rec.onnx", "pp-ocrv5-mobile"),
            PaddleOcrVariant::Server => ("PP-OCRv5_server_rec.onnx", "pp-ocrv5-server"),
        };

        let detector = match det_max_side {
            Some(ms) => OcrDetector::with_max_side(models_dir, variant, ms)?,
            None => OcrDetector::new(models_dir, variant)?,
        };

        let rec_path = format!("{models_dir}/ocr/{rec_name}");
        tracing::info!("Loading OCR {variant_name} recognition model...");
        let rec_session = load_session(&rec_path)?;

        // Build CTC character dictionary: blank + keys + space
        let mut char_dict = vec![String::new()]; // index 0 = CTC blank
        for line in OCR_KEYS_V5.lines() {
            // Only strip line endings — do NOT use trim() which removes
            // Unicode whitespace like U+3000 (Ideographic Space) present
            // in the PP-OCRv5 dictionary, causing index shift.
            char_dict.push(line.to_string());
        }
        char_dict.push(" ".to_string());

        tracing::info!(
            "OCR service ready: {variant_name} ({} characters), det_max_side={}.",
            char_dict.len(),
            det_max_side.unwrap_or(ocr_detector::DEFAULT_DET_MAX_SIDE),
        );
        Ok(Self {
            detector,
            rec_session: Mutex::new(rec_session),
            char_dict,
            variant_name: variant_name.to_string(),
            detection_mode: mode,
        })
    }

    async fn recognize_impl(&self, img: &DynamicImage) -> Result<Vec<OcrItem>, String> {
        let (orig_w, orig_h) = (img.width(), img.height());
        if orig_w > 10000 || orig_h > 10000 {
            return Err("Image too large".into());
        }

        // Prepared crop: all data needed after detection+classification to feed the rec model.
        struct PreparedCrop {
            /// Preprocessed CHW f32 tensor for this crop (3 × 48 × w).
            tensor: Array4<f32>,
            bbox: TextBox,
            angle_deg: f32,
            corners: Option<[(f32, f32); 4]>,
        }

        let mut prepared: Vec<PreparedCrop> = Vec::new();

        match self.detection_mode {
            DetectionMode::Components => {
                let boxes = self.detector.detect(img).await?;
                for bbox in boxes {
                    let cropped = ocr_detector::crop_text_region(img, &bbox);
                    if cropped.width() < 2 || cropped.height() < 2 {
                        continue;
                    }
                    let (rotated, was_flipped) = self.detector.classify_and_rotate(&cropped).await?;
                    if let Some(tensor) = preprocess_for_rec(&rotated) {
                        prepared.push(PreparedCrop {
                            tensor,
                            bbox,
                            angle_deg: if was_flipped { 180.0 } else { 0.0 },
                            corners: None,
                        });
                    }
                }
            }
            DetectionMode::Contours => {
                let rboxes = self.detector.detect_with_contours(img).await?;
                for rbox in rboxes {
                    let cropped = ocr_detector::crop_rotated_text_region(img, &rbox);
                    if cropped.width() < 2 || cropped.height() < 2 {
                        continue;
                    }
                    let (rotated, was_flipped) = self.detector.classify_and_rotate(&cropped).await?;
                    let mut angle_deg = rbox.angle.to_degrees();
                    if was_flipped {
                        angle_deg += 180.0;
                    }
                    let bbox = TextBox {
                        x: rbox.center.0 - rbox.size.0 / 2.0,
                        y: rbox.center.1 - rbox.size.1 / 2.0,
                        w: rbox.size.0,
                        h: rbox.size.1,
                    };
                    if let Some(tensor) = preprocess_for_rec(&rotated) {
                        prepared.push(PreparedCrop {
                            tensor,
                            bbox,
                            angle_deg,
                            corners: Some(rbox.corners),
                        });
                    }
                }
            }
        }

        if prepared.is_empty() {
            return Ok(vec![]);
        }

        // --- Single batched inference call ---
        // All crops are padded to max_w with zeros (same as PaddleOCR's official batch impl).
        // ORT's per-call overhead outweighs the padding cost; one call is faster than N buckets.
        let n = prepared.len();
        let max_w = prepared.iter().map(|p| p.tensor.shape()[3]).max().unwrap_or(1);

        let mut batch = Array4::<f32>::zeros((n, 3, 48, max_w));
        for (i, p) in prepared.iter().enumerate() {
            let w = p.tensor.shape()[3];
            batch
                .slice_mut(s![i, .., .., ..w])
                .assign(&p.tensor.slice(s![0, .., .., ..]));
        }

        let batch_tensor = Tensor::from_array(batch).map_err(|e| format!("Batch tensor: {e}"))?;
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
        let rec_array: ndarray::ArrayD<f32> = {
            let mut session = self.rec_session.lock().await;
            let outputs = session
                .run_async(ort::inputs![batch_tensor], &*options)
                .map_err(|e| format!("Recognition run_async: {e}"))?
                .await
                .map_err(|e| format!("Recognition inference: {e}"))?;
            outputs[0]
                .try_extract_array::<f32>()
                .map_err(|e| format!("Extract rec output: {e}"))?
                .to_owned()
        };

        let mut results = Vec::new();
        for (i, prep) in prepared.into_iter().enumerate() {
            let item_view = rec_array.slice(s![i, .., ..]);
            let shape = item_view.shape();
            if shape.len() < 2 {
                continue;
            }
            if let Some(mut item) = ctc_decode(item_view.view(), shape[0], shape[1], &self.char_dict, &prep.bbox) {
                item.angle = prep.angle_deg;
                item.corners = prep.corners;
                results.push(item);
            }
        }

        ocr_detector::assign_paragraph_ids(&mut results);
        Ok(results)
    }

    /// Kept for possible future use (single-crop path).
    #[allow(dead_code)]
    async fn recognize_text(&self, img: &DynamicImage, bbox: &TextBox) -> Result<Option<OcrItem>, String> {
        self.recognize_text_ctc(img, bbox).await
    }

    #[allow(dead_code)]
    async fn recognize_text_ctc(&self, img: &DynamicImage, bbox: &TextBox) -> Result<Option<OcrItem>, String> {
        let Some(tensor) = preprocess_for_rec(img) else {
            return Ok(None);
        };

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
        let rec_array: ndarray::ArrayD<f32> = {
            let mut session = self.rec_session.lock().await;
            let outputs = session
                .run_async(ort::inputs![input_tensor], &*options)
                .map_err(|e| format!("Recognition run_async: {e}"))?
                .await
                .map_err(|e| format!("Recognition inference: {e}"))?;
            outputs[0]
                .try_extract_array::<f32>()
                .map_err(|e| format!("Extract rec output: {e}"))?
                .to_owned()
        };

        let view = rec_array.view();
        let shape = view.shape();
        if shape.len() < 3 {
            return Ok(None);
        }
        let seq_len = shape[1];
        let vocab_size = shape[2];
        Ok(ctc_decode(
            view.slice(s![0, .., ..]),
            seq_len,
            vocab_size,
            &self.char_dict,
            bbox,
        ))
    }
}

#[async_trait::async_trait]
impl OcrBackend for OcrService {
    fn name(&self) -> &str {
        &self.variant_name
    }

    async fn recognize(&self, img: &DynamicImage) -> Result<Vec<OcrItem>, String> {
        self.recognize_impl(img).await
    }
}

/// Preprocess a text-region crop for the rec model.
/// Returns `(1, 3, 48, target_w)` f32 tensor, or None if the crop is too small.
fn preprocess_for_rec(img: &DynamicImage) -> Option<Array4<f32>> {
    let (w, h) = (img.width(), img.height());
    if w < 2 || h < 2 {
        return None;
    }

    // Invert if dark background
    let img = {
        let gray = img.to_luma8();
        let (gw, gh) = gray.dimensions();
        let avg_lum: f64 = gray.pixels().map(|p| f64::from(p.0[0])).sum::<f64>() / (f64::from(gw) * f64::from(gh));
        if avg_lum < 127.0 {
            let mut g = gray;
            image::imageops::invert(&mut g);
            DynamicImage::ImageLuma8(g)
        } else {
            img.clone()
        }
    };

    let target_h = 48u32;
    let target_w = ((img.width() as f32 / img.height() as f32) * target_h as f32).max(1.0) as u32;
    let target_w = target_w.min(2048);

    let resized = img.resize_exact(target_w, target_h, image::imageops::FilterType::CatmullRom);
    let rgb = resized.to_rgb8();

    let mean = [0.5f32; 3];
    let std = [0.5f32; 3];
    let mut tensor = Array4::<f32>::zeros((1, 3, target_h as usize, target_w as usize));
    for y in 0..target_h as usize {
        for x in 0..target_w as usize {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                tensor[[0, c, y, x]] = (f32::from(pixel[c]) / 255.0 - mean[c]) / std[c];
            }
        }
    }
    Some(tensor)
}

/// CTC greedy decode for a single sequence slice `[seq_len, vocab_size]`.
fn ctc_decode(
    view: ndarray::ArrayView2<f32>,
    seq_len: usize,
    vocab_size: usize,
    char_dict: &[String],
    bbox: &TextBox,
) -> Option<OcrItem> {
    let mut text = String::new();
    let mut total_score = 0.0f32;
    let mut count = 0usize;
    let mut last_idx: i64 = 0;
    let mut char_timesteps: Vec<(usize, usize)> = Vec::new();
    let mut current_char_start: usize = 0;

    for t in 0..seq_len {
        let mut max_idx = 0i64;
        let mut max_val = f32::NEG_INFINITY;
        for v in 0..vocab_size {
            let val = view[[t, v]];
            if val > max_val {
                max_val = val;
                max_idx = v as i64;
            }
        }
        if max_idx != 0 && max_idx != last_idx {
            if let Some(ch) = char_dict.get(max_idx as usize) {
                text.push_str(ch);
                total_score += max_val.exp() / (1.0 + max_val.exp());
                count += 1;
                char_timesteps.push((current_char_start, t));
            }
            current_char_start = t;
        } else if max_idx == 0 && last_idx != 0 {
            current_char_start = t + 1;
        }
        last_idx = max_idx;
    }

    if text.is_empty() {
        return None;
    }

    let avg_score = if count > 0 { total_score / count as f32 } else { 0.0 };
    let step_w = bbox.w / seq_len as f32;
    let char_positions = char_timesteps
        .iter()
        .map(|&(start_t, end_t)| (start_t as f32 * step_w, (end_t - start_t + 1) as f32 * step_w))
        .collect();

    Some(OcrItem {
        text,
        score: avg_score,
        x: bbox.x,
        y: bbox.y,
        w: bbox.w,
        h: bbox.h,
        angle: 0.0,
        corners: None,
        paragraph_id: 0,
        char_positions: Some(char_positions),
    })
}

fn load_session(path: &str) -> Result<Session, String> {
    crate::build_session(path).map_err(|e| format!("Load model {path}: {e}"))
}
