/// Chinese-CLIP ViT-B-16 service via ONNX Runtime.
/// Image → 512-dim vector, Text → 512-dim vector.
/// Also provides zero-shot classification via `clip_categories` taxonomy.
use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use tokio::sync::Mutex;

use crate::clip_categories::{self, TagResult};

use image::DynamicImage;
use ndarray::Array4;
use ort::{session::Session, value::Tensor};

use crate::tokenizer::BertTokenizer;

const IMG_SIZE: u32 = 224;
const CONTEXT_LENGTH: usize = 52;

// ImageNet normalization (same as Chinese-CLIP)
const MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
const STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];

pub struct ClipService {
    img_session: Mutex<Session>,
    txt_session: Mutex<Session>,
    tokenizer: BertTokenizer,
    cache_dir: String,
}

impl ClipService {
    pub fn new(models_dir: &str) -> Result<Self, String> {
        let img_path = format!("{models_dir}/clip/vit-b-16.img.fp32.onnx");
        let txt_path = format!("{models_dir}/clip/vit-b-16.txt.fp32.onnx");

        if !Path::new(&img_path).exists() {
            return Err(format!("CLIP image model not found: {img_path}"));
        }
        if !Path::new(&txt_path).exists() {
            return Err(format!("CLIP text model not found: {txt_path}"));
        }

        tracing::info!("Loading CLIP image model: {img_path}");
        let img_session = crate::build_session(&img_path).map_err(|e| format!("Load CLIP img model: {e}"))?;

        tracing::info!("Loading CLIP text model: {txt_path}");
        let txt_session = crate::build_session(&txt_path).map_err(|e| format!("Load CLIP txt model: {e}"))?;

        let tokenizer = BertTokenizer::new();

        // Pre-warm the category embedding cache while still in a sync context
        // (new() is called inside spawn_blocking from AiService).
        // We use a std Mutex temporarily for the sync callback.
        let txt_session_std = StdMutex::new(txt_session);
        let cache_dir_clone = models_dir.to_string();
        let _ = clip_categories::ensure_cache(
            &|text| {
                let token_ids = tokenizer.encode(text, CONTEXT_LENGTH);
                let input_tensor = Tensor::from_array(([1i64, CONTEXT_LENGTH as i64], token_ids))
                    .map_err(|e| format!("Create tensor: {e}"))?;
                let mut session = txt_session_std.lock().map_err(|e| format!("Lock: {e}"))?;
                let outputs = session
                    .run(ort::inputs![input_tensor])
                    .map_err(|e| format!("CLIP txt inference: {e}"))?;
                let (_shape, data) = outputs[0]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| format!("Extract tensor: {e}"))?;
                Ok(data.to_vec())
            },
            Some(&cache_dir_clone),
        );
        let txt_session = txt_session_std.into_inner().map_err(|e| format!("Unwrap: {e}"))?;

        tracing::info!("CLIP service ready.");

        Ok(Self {
            img_session: Mutex::new(img_session),
            txt_session: Mutex::new(txt_session),
            tokenizer,
            cache_dir: models_dir.to_string(),
        })
    }

    /// Image → 512-dim CLIP embedding.
    pub async fn embed_image(&self, img: &DynamicImage) -> Result<Vec<f32>, String> {
        let input = preprocess_image(img);
        let input_tensor = Tensor::from_array(input).map_err(|e| format!("Create tensor: {e}"))?;
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

        let data: Vec<f32> = {
            let mut session = self.img_session.lock().await;
            let outputs = session
                .run_async(ort::inputs![input_tensor], &*options)
                .map_err(|e| format!("CLIP img run_async: {e}"))?
                .await
                .map_err(|e| format!("CLIP img inference: {e}"))?;
            let (_shape, d) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("Extract tensor: {e}"))?;
            d.to_vec()
        };

        Ok(data)
    }

    /// Text → 512-dim CLIP embedding.
    pub async fn embed_text(&self, text: &str) -> Result<Vec<f32>, String> {
        let token_ids = self.tokenizer.encode(text, CONTEXT_LENGTH);
        let input_tensor = Tensor::from_array(([1i64, CONTEXT_LENGTH as i64], token_ids))
            .map_err(|e| format!("Create tensor: {e}"))?;
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

        let data: Vec<f32> = {
            let mut session = self.txt_session.lock().await;
            let outputs = session
                .run_async(ort::inputs![input_tensor], &*options)
                .map_err(|e| format!("CLIP txt run_async: {e}"))?
                .await
                .map_err(|e| format!("CLIP txt inference: {e}"))?;
            let (_shape, d) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("Extract tensor: {e}"))?;
            d.to_vec()
        };

        Ok(data)
    }

    /// Classify image vector against the taxonomy using zero-shot CLIP.
    /// Cache is already warm from `new()`, so this is sync and fast.
    pub fn classify(&self, image_vec: &[f32]) -> Result<Vec<TagResult>, String> {
        clip_categories::classify(image_vec, &|_| Err("cache miss".into()), Some(&self.cache_dir))
    }
}

/// Resize to 224×224, normalize with `ImageNet` mean/std, output NCHW f32 tensor.
fn preprocess_image(img: &DynamicImage) -> Array4<f32> {
    let resized = img.resize_exact(IMG_SIZE, IMG_SIZE, image::imageops::FilterType::CatmullRom);
    let rgb = resized.to_rgb8();

    let mut tensor = Array4::<f32>::zeros((1, 3, IMG_SIZE as usize, IMG_SIZE as usize));
    for y in 0..IMG_SIZE as usize {
        for x in 0..IMG_SIZE as usize {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                tensor[[0, c, y, x]] = (f32::from(pixel[c]) / 255.0 - MEAN[c]) / STD[c];
            }
        }
    }
    tensor
}
