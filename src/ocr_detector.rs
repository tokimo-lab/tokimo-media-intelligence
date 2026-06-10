/// Shared text detection + classification pipeline.
///
/// Extracted from `OcrService` so that both CTC-based (`OcrService`) and
/// Attention-based (`OcrAttentionService`) recognisers can share the same
/// expensive detection + classification ONNX sessions.
use std::sync::Arc;

use tokio::sync::Mutex;

use image::{DynamicImage, GrayImage, Luma};
use ndarray::Array4;
use ort::{session::Session, value::Tensor};

use crate::ocr_backend::PaddleOcrVariant;

/// Bounding box for a detected text region in original image coordinates.
#[derive(Clone)]
pub struct TextBox {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

/// Bounding box for a detected text region with rotation info.
#[derive(Clone)]
pub struct RotatedBox {
    /// Center point in original image coords.
    pub center: (f32, f32),
    /// (width, height) of the tight rotated rect.
    pub size: (f32, f32),
    /// Rotation angle in radians.
    pub angle: f32,
    /// 4 corner points in original image coords: [TL, TR, BR, BL].
    pub corners: [(f32, f32); 4],
    /// Confidence score (average prob within the polygon).
    pub score: f32,
}

/// Internal result from `DBNet` inference, shared by both detection modes.
struct DetectionOutput {
    prob_data: Vec<f32>,
    pad_w: usize,
    pad_h: usize,
    img_w: usize,
    img_h: usize,
    orig_w: f32,
    orig_h: f32,
}

/// Default detection resolution — longest side is capped to this value.
pub const DEFAULT_DET_MAX_SIDE: u32 = 4096;

/// Loads and holds the detection + classification ONNX sessions.
/// Thread-safe via interior `Mutex` — designed to be wrapped in `Arc`.
pub struct OcrDetector {
    det_session: Mutex<Session>,
    cls_session: Mutex<Session>,
    /// Maximum pixel length for the longest side during detection.
    /// Images larger than this are downscaled proportionally.
    det_max_side: f32,
}

impl OcrDetector {
    pub fn new(models_dir: &str, variant: PaddleOcrVariant) -> Result<Self, String> {
        Self::with_max_side(models_dir, variant, DEFAULT_DET_MAX_SIDE)
    }

    pub fn with_max_side(models_dir: &str, variant: PaddleOcrVariant, det_max_side: u32) -> Result<Self, String> {
        let det_name = match variant {
            PaddleOcrVariant::Mobile => "PP-OCRv5_mobile_det.onnx",
            PaddleOcrVariant::Server => "PP-OCRv5_server_det.onnx",
        };
        let det_path = format!("{models_dir}/ocr/{det_name}");
        let cls_path = format!("{models_dir}/ocr/PP-OCRv5_cls.onnx");

        let det_session = load_session(&det_path)?;
        let cls_session = load_session(&cls_path)?;

        Ok(Self {
            det_session: Mutex::new(det_session),
            cls_session: Mutex::new(cls_session),
            det_max_side: det_max_side as f32,
        })
    }

    /// Common `DBNet` inference — shared by both detection modes.
    async fn run_dbnet(&self, img: &DynamicImage) -> Result<DetectionOutput, String> {
        let (orig_w, orig_h) = (img.width() as f32, img.height() as f32);

        let max_side = self.det_max_side;
        let scale = (max_side / orig_w.max(orig_h)).min(1.0);
        let new_w = ((orig_w * scale) as u32).max(32);
        let new_h = ((orig_h * scale) as u32).max(32);
        let pad_w = new_w.div_ceil(32) * 32;
        let pad_h = new_h.div_ceil(32) * 32;

        let resized = img.resize_exact(new_w, new_h, image::imageops::FilterType::CatmullRom);
        let rgb = resized.to_rgb8();

        let mean = [0.485f32, 0.456, 0.406];
        let std = [0.229f32, 0.224, 0.225];
        let mut tensor = Array4::<f32>::zeros((1, 3, pad_h as usize, pad_w as usize));
        for y in 0..new_h as usize {
            for x in 0..new_w as usize {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    tensor[[0, c, y, x]] = (f32::from(pixel[c]) / 255.0 - mean[c]) / std[c];
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
        let prob_data: Vec<f32> = {
            let mut session = self.det_session.lock().await;
            let outputs = session
                .run_async(ort::inputs![input_tensor], &*options)
                .map_err(|e| format!("Detection run_async: {e}"))?
                .await
                .map_err(|e| format!("Detection inference: {e}"))?;
            let (_shape, prob) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("Extract det output: {e}"))?;
            prob.to_vec()
        };

        Ok(DetectionOutput {
            prob_data,
            pad_w: pad_w as usize,
            pad_h: pad_h as usize,
            img_w: new_w as usize,
            img_h: new_h as usize,
            orig_w,
            orig_h,
        })
    }

    /// Run `DBNet` text detection, return bounding boxes in original image coords.
    pub async fn detect(&self, img: &DynamicImage) -> Result<Vec<TextBox>, String> {
        let out = self.run_dbnet(img).await?;
        Ok(extract_boxes_from_prob_map(
            &out.prob_data,
            out.pad_w,
            out.pad_h,
            out.img_w,
            out.img_h,
            out.orig_w,
            out.orig_h,
        ))
    }

    /// Run `DBNet` text detection with contour-based post-processing (RapidOCR-style).
    /// Returns rotated bounding boxes for better results on angled text.
    pub async fn detect_with_contours(&self, img: &DynamicImage) -> Result<Vec<RotatedBox>, String> {
        let out = self.run_dbnet(img).await?;
        Ok(extract_rotated_boxes(
            &out.prob_data,
            out.pad_w,
            out.pad_h,
            out.img_w,
            out.img_h,
            out.orig_w,
            out.orig_h,
        ))
    }

    /// Classify text orientation (0° or 180°) and rotate if needed.
    /// Returns `(image, was_rotated)` — `was_rotated` is true when the text
    /// was detected as upside-down and the image was flipped 180°.
    pub async fn classify_and_rotate(&self, img: &DynamicImage) -> Result<(DynamicImage, bool), String> {
        let resized = img.resize_exact(192, 48, image::imageops::FilterType::CatmullRom);
        let rgb = resized.to_rgb8();

        let mean = [0.5f32, 0.5, 0.5];
        let std = [0.5f32, 0.5, 0.5];
        let mut tensor = Array4::<f32>::zeros((1, 3, 48, 192));
        for y in 0..48 {
            for x in 0..192 {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    tensor[[0, c, y, x]] = (f32::from(pixel[c]) / 255.0 - mean[c]) / std[c];
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
        let is_flipped: bool = {
            let mut session = self.cls_session.lock().await;
            let outputs = session
                .run_async(ort::inputs![input_tensor], &*options)
                .map_err(|e| format!("Classification run_async: {e}"))?
                .await
                .map_err(|e| format!("Classification inference: {e}"))?;
            let (_shape, cls_data) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("Extract cls output: {e}"))?;
            cls_data.len() >= 2 && cls_data[1] > cls_data[0] && cls_data[1] > 0.9
        };

        if is_flipped {
            Ok((img.rotate180(), true))
        } else {
            Ok((img.clone(), false))
        }
    }
}

/// Crop a text region from the original image using bounding box.
/// Adds small padding to avoid clipping character edges.
pub fn crop_text_region(img: &DynamicImage, bbox: &TextBox) -> DynamicImage {
    let pad = 3u32;
    let x = (bbox.x as u32).saturating_sub(pad).min(img.width().saturating_sub(1));
    let y = (bbox.y as u32).saturating_sub(pad).min(img.height().saturating_sub(1));
    let w = ((bbox.w as u32) + 2 * pad).min(img.width() - x).max(1);
    let h = ((bbox.h as u32) + 2 * pad).min(img.height() - y).max(1);
    img.crop_imm(x, y, w, h)
}

/// Crop a text region from the original image using a rotated bounding box.
/// Applies perspective transform for proper deskewing of angled text.
pub fn crop_rotated_text_region(img: &DynamicImage, rbox: &RotatedBox) -> DynamicImage {
    let [tl, tr, br, bl] = rbox.corners;

    let crop_w = dist_f32(tl, tr).max(dist_f32(bl, br));
    let crop_h = dist_f32(tl, bl).max(dist_f32(tr, br));

    let out_w = crop_w.round().max(1.0) as u32;
    let out_h = crop_h.round().max(1.0) as u32;

    if out_w < 2 || out_h < 2 {
        return DynamicImage::ImageRgb8(image::RgbImage::new(1, 1));
    }

    // Perspective transform: map output pixels → source image pixels.
    // dst corners in output image → src corners in original image.
    let dst = [
        (0.0f32, 0.0f32),
        (out_w as f32, 0.0),
        (out_w as f32, out_h as f32),
        (0.0, out_h as f32),
    ];
    let src = [tl, tr, br, bl];

    let Some(coeffs) = compute_perspective_transform(&dst, &src) else {
        return DynamicImage::ImageRgb8(image::RgbImage::new(1, 1));
    };

    let src_rgb = img.to_rgb8();
    let (src_w, src_h) = src_rgb.dimensions();
    let mut out = image::RgbImage::new(out_w, out_h);

    for dy in 0..out_h {
        for dx in 0..out_w {
            let u = f64::from(dx);
            let v = f64::from(dy);
            let denom = coeffs[6] * u + coeffs[7] * v + 1.0;
            if denom.abs() < 1e-10 {
                continue;
            }
            let sx = ((coeffs[0] * u + coeffs[1] * v + coeffs[2]) / denom) as f32;
            let sy = ((coeffs[3] * u + coeffs[4] * v + coeffs[5]) / denom) as f32;

            let pixel = bilinear_sample(&src_rgb, sx, sy, src_w, src_h);
            out.put_pixel(dx, dy, image::Rgb(pixel));
        }
    }

    let result = DynamicImage::ImageRgb8(out);

    // Vertical text: rotate 90° so recogniser sees horizontal text.
    if out_h as f32 / out_w.max(1) as f32 >= 1.5 {
        result.rotate90()
    } else {
        result
    }
}

/// Assign `paragraph_id` to each `OcrItem` by clustering lines that belong
/// to the same text column/paragraph.
pub fn assign_paragraph_ids(items: &mut [crate::ocr::OcrItem]) {
    if items.is_empty() {
        return;
    }

    items.sort_by(|a, b| {
        let y_cmp = a.y.partial_cmp(&b.y).unwrap_or(std::cmp::Ordering::Equal);
        if y_cmp == std::cmp::Ordering::Equal {
            a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal)
        } else {
            y_cmp
        }
    });

    let n = items.len();
    let mut parent: Vec<usize> = (0..n).collect();

    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }

    fn union(parent: &mut [usize], a: usize, b: usize) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            parent[rb] = ra;
        }
    }

    for i in 0..n {
        for j in (i + 1)..n {
            let a = &items[i];
            let b = &items[j];

            let a_left = a.x;
            let a_right = a.x + a.w;
            let b_left = b.x;
            let b_right = b.x + b.w;

            let overlap_start = a_left.max(b_left);
            let overlap_end = a_right.min(b_right);
            let overlap = (overlap_end - overlap_start).max(0.0);

            let min_width = a.w.min(b.w);
            if min_width <= 0.0 || overlap / min_width < 0.5 {
                continue;
            }

            let a_bottom = a.y + a.h;
            let b_top = b.y;
            let gap = (b_top - a_bottom).max(0.0);
            let avg_h = f32::midpoint(a.h, b.h);

            if gap <= avg_h * 1.5 {
                union(&mut parent, i, j);
            }
        }
    }

    let mut root_to_id: std::collections::HashMap<usize, u32> = std::collections::HashMap::new();
    let mut next_id = 0u32;

    for (i, item) in items.iter_mut().enumerate() {
        let root = find(&mut parent, i);
        let id = root_to_id.entry(root).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id
        });
        item.paragraph_id = *id;
    }
}

// ── Detection post-processing ────────────────────────────────────────────

fn extract_boxes_from_prob_map(
    prob_data: &[f32],
    map_w: usize,
    map_h: usize,
    img_w: usize,
    img_h: usize,
    orig_w: f32,
    orig_h: f32,
) -> Vec<TextBox> {
    let threshold = 0.3f32;
    let box_threshold = 0.5f32;
    let min_size = 3;
    let unclip_ratio = 1.6f32;

    let mut binary = GrayImage::new(map_w as u32, map_h as u32);
    for y in 0..map_h {
        for x in 0..map_w {
            let idx = y * map_w + x;
            let val = prob_data.get(idx).copied().unwrap_or(0.0);
            binary.put_pixel(x as u32, y as u32, Luma([if val > threshold { 255 } else { 0 }]));
        }
    }

    let dilated = dilate_binary(&binary, 2);
    let components = find_connected_components(&dilated);

    let scale_x = orig_w / img_w as f32;
    let scale_y = orig_h / img_h as f32;

    let mut boxes = Vec::new();
    for comp in &components {
        if comp.pixels.len() < min_size * min_size {
            continue;
        }

        let min_x = comp.pixels.iter().map(|p| p.0).min().unwrap_or(0);
        let max_x = comp.pixels.iter().map(|p| p.0).max().unwrap_or(0);
        let min_y = comp.pixels.iter().map(|p| p.1).min().unwrap_or(0);
        let max_y = comp.pixels.iter().map(|p| p.1).max().unwrap_or(0);

        let box_w = max_x - min_x + 1;
        let box_h = max_y - min_y + 1;
        if box_w < min_size || box_h < min_size {
            continue;
        }

        // Score using only pixels above the threshold (avoids dilution from
        // dilation-expanded low-probability pixels).
        let mut total = 0.0f32;
        let mut count = 0;
        for &(px, py) in &comp.pixels {
            let idx = py * map_w + px;
            let val = prob_data.get(idx).copied().unwrap_or(0.0);
            if val > threshold {
                total += val;
                count += 1;
            }
        }
        let score = if count > 0 { total / count as f32 } else { 0.0 };
        if score < box_threshold {
            continue;
        }

        // Unclip expansion: expand bounding box proportionally to its size,
        // mimicking PaddleOCR's Vatti polygon clipping (`unclip_ratio`).
        let area = (box_w * box_h) as f32;
        let perimeter = 2.0 * (box_w + box_h) as f32;
        let expansion = if perimeter > 0.0 {
            area * unclip_ratio / perimeter
        } else {
            0.0
        };

        let exp_x = (min_x as f32 - expansion).max(0.0);
        let exp_y = (min_y as f32 - expansion).max(0.0);
        let exp_w = (box_w as f32 + 2.0 * expansion).min(img_w as f32 - exp_x);
        let exp_h = (box_h as f32 + 2.0 * expansion).min(img_h as f32 - exp_y);

        let x = exp_x * scale_x;
        let y = exp_y * scale_y;
        let w = exp_w * scale_x;
        let h = exp_h * scale_y;

        boxes.push(TextBox { x, y, w, h });
    }

    boxes
}

fn dilate_binary(img: &GrayImage, iterations: u32) -> GrayImage {
    let mut current = img.clone();
    for _ in 0..iterations {
        let mut next = current.clone();
        let (w, h) = current.dimensions();
        for y in 1..h.saturating_sub(1) {
            for x in 1..w.saturating_sub(1) {
                let mut has_white = false;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let nx = (x as i32 + dx) as u32;
                        let ny = (y as i32 + dy) as u32;
                        if current.get_pixel(nx, ny)[0] > 0 {
                            has_white = true;
                            break;
                        }
                    }
                    if has_white {
                        break;
                    }
                }
                if has_white {
                    next.put_pixel(x, y, Luma([255]));
                }
            }
        }
        current = next;
    }
    current
}

struct Component {
    pixels: Vec<(usize, usize)>,
}

fn find_connected_components(img: &GrayImage) -> Vec<Component> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut visited = vec![false; w * h];
    let mut components = Vec::new();

    for y in 0..h {
        for x in 0..w {
            if visited[y * w + x] || img.get_pixel(x as u32, y as u32)[0] == 0 {
                continue;
            }

            let mut pixels = Vec::new();
            let mut queue = std::collections::VecDeque::new();
            queue.push_back((x, y));
            visited[y * w + x] = true;

            while let Some((cx, cy)) = queue.pop_front() {
                pixels.push((cx, cy));
                for (dx, dy) in [(-1i32, 0), (1, 0), (0, -1), (0, 1)] {
                    let nx = cx as i32 + dx;
                    let ny = cy as i32 + dy;
                    if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
                        let nx = nx as usize;
                        let ny = ny as usize;
                        if !visited[ny * w + nx] && img.get_pixel(nx as u32, ny as u32)[0] > 0 {
                            visited[ny * w + nx] = true;
                            queue.push_back((nx, ny));
                        }
                    }
                }
            }

            components.push(Component { pixels });
        }
    }

    components
}

// ── Contour-based detection (RapidOCR-style) ─────────────────────────────────

fn extract_rotated_boxes(
    prob_data: &[f32],
    map_w: usize,
    map_h: usize,
    img_w: usize,
    img_h: usize,
    orig_w: f32,
    orig_h: f32,
) -> Vec<RotatedBox> {
    let threshold = 0.3f32;
    let box_thresh = 0.5f32;
    let min_size = 3usize;
    let unclip_ratio = 1.6f32;
    let max_candidates = 1000;

    let mut binary = GrayImage::new(map_w as u32, map_h as u32);
    for y in 0..map_h {
        for x in 0..map_w {
            let idx = y * map_w + x;
            let val = prob_data.get(idx).copied().unwrap_or(0.0);
            binary.put_pixel(x as u32, y as u32, Luma([if val > threshold { 255 } else { 0 }]));
        }
    }

    let dilated = dilate_binary(&binary, 2);
    let components = find_connected_components(&dilated);

    let scale_x = orig_w / img_w as f32;
    let scale_y = orig_h / img_h as f32;

    // Maximum dimensions: skip regions wider/taller than 80% of the image
    // (these are background blobs, not text).
    let max_box_w = (img_w as f32 * 0.8) as usize;
    let max_box_h = (img_h as f32 * 0.8) as usize;

    let mut boxes = Vec::new();
    for comp in components.iter().take(max_candidates) {
        if comp.pixels.len() < min_size * min_size {
            continue;
        }

        // Check bounding box dimensions — skip giant regions.
        let min_x = comp.pixels.iter().map(|p| p.0).min().unwrap_or(0);
        let max_x = comp.pixels.iter().map(|p| p.0).max().unwrap_or(0);
        let min_y = comp.pixels.iter().map(|p| p.1).min().unwrap_or(0);
        let max_y = comp.pixels.iter().map(|p| p.1).max().unwrap_or(0);
        let comp_w = max_x - min_x + 1;
        let comp_h = max_y - min_y + 1;
        if comp_w > max_box_w || comp_h > max_box_h {
            continue;
        }

        // Convex hull (with boundary-pixel optimisation for large components).
        let hull_input = get_hull_input_pixels(&comp.pixels);
        let hull = convex_hull(&hull_input);
        if hull.len() < 3 {
            continue;
        }

        // Minimum area rotated rectangle (still needed for scoring and size checks).
        let (center, (w, h), angle) = min_area_rect(&hull);
        let short_side = w.min(h);
        if short_side < min_size as f32 {
            continue;
        }

        // Use the min-area-rect corners for scoring and output.
        // NOTE: fit_quad_to_hull() exists but is NOT used here because
        // DBNet's binary mask does not preserve meaningful perspective
        // geometry — the convex hull boundary noise produces random
        // "perspective" directions.  Perspective editing is available
        // in the frontend for manual correction.
        let corners = rect_corners(center, (w, h), angle);

        // Score: average probability inside the polygon (pre-binarisation map).
        let score = box_score_fast(prob_data, map_w, map_h, &corners);
        if score < box_thresh {
            continue;
        }

        // Unclip: expand the rect outward.
        let area = w * h;
        let perimeter = 2.0 * (w + h);
        let distance = if perimeter > 0.0 {
            area * unclip_ratio / perimeter
        } else {
            0.0
        };

        let new_w = w + 2.0 * distance;
        let new_h = h + 2.0 * distance;

        if new_w.min(new_h) < (min_size + 2) as f32 {
            continue;
        }

        // Recompute corners from expanded rect, then scale to original coords.
        let exp_corners = rect_corners(center, (new_w, new_h), angle);
        let scaled: [(f32, f32); 4] = [
            (exp_corners[0].0 * scale_x, exp_corners[0].1 * scale_y),
            (exp_corners[1].0 * scale_x, exp_corners[1].1 * scale_y),
            (exp_corners[2].0 * scale_x, exp_corners[2].1 * scale_y),
            (exp_corners[3].0 * scale_x, exp_corners[3].1 * scale_y),
        ];

        let ordered = order_points(scaled);

        // Recompute center / size / angle from scaled corners.
        let sc_center = (
            f32::midpoint(ordered[0].0, ordered[2].0),
            f32::midpoint(ordered[0].1, ordered[2].1),
        );
        let sc_w = dist_f32(ordered[0], ordered[1]);
        let sc_h = dist_f32(ordered[0], ordered[3]);
        let sc_angle = (ordered[1].1 - ordered[0].1).atan2(ordered[1].0 - ordered[0].0);

        // Skip degenerate boxes: NaN/Inf in any coordinate, or too large.
        if ordered.iter().any(|c| !c.0.is_finite() || !c.1.is_finite()) {
            continue;
        }
        if !sc_w.is_finite() || !sc_h.is_finite() {
            continue;
        }
        // Skip boxes whose AABB spans more than 80% of the original image.
        let aabb_w =
            ordered.iter().map(|c| c.0).fold(f32::MIN, f32::max) - ordered.iter().map(|c| c.0).fold(f32::MAX, f32::min);
        let aabb_h =
            ordered.iter().map(|c| c.1).fold(f32::MIN, f32::max) - ordered.iter().map(|c| c.1).fold(f32::MAX, f32::min);
        if aabb_w > orig_w * 0.8 || aabb_h > orig_h * 0.8 {
            continue;
        }

        boxes.push(RotatedBox {
            center: sc_center,
            size: (sc_w, sc_h),
            angle: sc_angle,
            corners: ordered,
            score,
        });
    }

    boxes
}

/// For large components, subsample to boundary pixels only.
fn get_hull_input_pixels(pixels: &[(usize, usize)]) -> Vec<(f32, f32)> {
    if pixels.len() <= 5000 {
        return pixels.iter().map(|&(x, y)| (x as f32, y as f32)).collect();
    }
    let set: std::collections::HashSet<(usize, usize)> = pixels.iter().copied().collect();
    pixels
        .iter()
        .filter(|&&(x, y)| {
            [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)].iter().any(|&(dx, dy)| {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                nx < 0 || ny < 0 || !set.contains(&(nx as usize, ny as usize))
            })
        })
        .map(|&(x, y)| (x as f32, y as f32))
        .collect()
}

/// Convex hull via Andrew's monotone chain (returns CCW polygon).
fn convex_hull(points: &[(f32, f32)]) -> Vec<(f32, f32)> {
    let n = points.len();
    if n < 3 {
        return points.to_vec();
    }

    let mut pts = points.to_vec();
    pts.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    });

    let mut hull: Vec<(f32, f32)> = Vec::with_capacity(2 * n);

    // Lower hull.
    for &p in &pts {
        while hull.len() >= 2 && cross_2d(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }

    // Upper hull.
    let lower_len = hull.len();
    for &p in pts.iter().rev().skip(1) {
        while hull.len() > lower_len && cross_2d(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }

    hull.pop(); // remove duplicate endpoint
    hull
}

/// 2D cross product of vectors OA and OB.
fn cross_2d(o: (f32, f32), a: (f32, f32), b: (f32, f32)) -> f32 {
    (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
}

/// Minimum area rotated rectangle enclosing a convex hull (rotating calipers).
/// Returns (center, (width, height), `angle_radians`) where width >= height.
fn min_area_rect(hull: &[(f32, f32)]) -> ((f32, f32), (f32, f32), f32) {
    if hull.is_empty() {
        return ((0.0, 0.0), (0.0, 0.0), 0.0);
    }
    if hull.len() == 1 {
        return (hull[0], (0.0, 0.0), 0.0);
    }
    if hull.len() == 2 {
        let cx = f32::midpoint(hull[0].0, hull[1].0);
        let cy = f32::midpoint(hull[0].1, hull[1].1);
        let d = dist_f32(hull[0], hull[1]);
        let a = (hull[1].1 - hull[0].1).atan2(hull[1].0 - hull[0].0);
        return ((cx, cy), (d, 0.0), a);
    }

    let n = hull.len();
    let mut best_area = f32::MAX;
    let mut best_center = (0.0f32, 0.0f32);
    let mut best_size = (0.0f32, 0.0f32);
    let mut best_angle = 0.0f32;

    for i in 0..n {
        let j = (i + 1) % n;
        let dx = hull[j].0 - hull[i].0;
        let dy = hull[j].1 - hull[i].1;
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-6 {
            continue;
        }

        // Edge direction (normalised) and perpendicular.
        let ex = dx / len;
        let ey = dy / len;
        let px = -ey;
        let py = ex;

        let mut min_along = f32::MAX;
        let mut max_along = f32::MIN;
        let mut min_perp = f32::MAX;
        let mut max_perp = f32::MIN;

        for &p in hull {
            let rx = p.0 - hull[i].0;
            let ry = p.1 - hull[i].1;
            let along = rx * ex + ry * ey;
            let perp = rx * px + ry * py;
            min_along = min_along.min(along);
            max_along = max_along.max(along);
            min_perp = min_perp.min(perp);
            max_perp = max_perp.max(perp);
        }

        let width = max_along - min_along;
        let height = max_perp - min_perp;
        let area = width * height;

        if area < best_area {
            best_area = area;
            let along_mid = f32::midpoint(min_along, max_along);
            let perp_mid = f32::midpoint(min_perp, max_perp);
            best_center = (
                hull[i].0 + along_mid * ex + perp_mid * px,
                hull[i].1 + along_mid * ey + perp_mid * py,
            );
            best_size = (width, height);
            best_angle = ey.atan2(ex);
        }
    }

    // Ensure width >= height.
    let (mut w, mut h, mut angle) = (best_size.0, best_size.1, best_angle);
    if h > w {
        std::mem::swap(&mut w, &mut h);
        angle += std::f32::consts::FRAC_PI_2;
    }

    // Normalise angle to [-π/2, π/2].
    while angle > std::f32::consts::FRAC_PI_2 {
        angle -= std::f32::consts::PI;
    }
    while angle < -std::f32::consts::FRAC_PI_2 {
        angle += std::f32::consts::PI;
    }

    (best_center, (w, h), angle)
}

/// Compute 4 corner points of a rotated rectangle.
fn rect_corners(center: (f32, f32), size: (f32, f32), angle: f32) -> [(f32, f32); 4] {
    let hw = size.0 / 2.0;
    let hh = size.1 / 2.0;
    let cos_a = angle.cos();
    let sin_a = angle.sin();

    let offsets: [(f32, f32); 4] = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)];

    [
        (
            center.0 + offsets[0].0 * cos_a - offsets[0].1 * sin_a,
            center.1 + offsets[0].0 * sin_a + offsets[0].1 * cos_a,
        ),
        (
            center.0 + offsets[1].0 * cos_a - offsets[1].1 * sin_a,
            center.1 + offsets[1].0 * sin_a + offsets[1].1 * cos_a,
        ),
        (
            center.0 + offsets[2].0 * cos_a - offsets[2].1 * sin_a,
            center.1 + offsets[2].0 * sin_a + offsets[2].1 * cos_a,
        ),
        (
            center.0 + offsets[3].0 * cos_a - offsets[3].1 * sin_a,
            center.1 + offsets[3].0 * sin_a + offsets[3].1 * cos_a,
        ),
    ]
}

/// Order 4 corner points as [TL, TR, BR, BL] (`RapidOCR` convention).
fn order_points(mut corners: [(f32, f32); 4]) -> [(f32, f32); 4] {
    corners.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    });

    // Left pair (smallest x): TL has smaller y.
    let (tl, bl) = if corners[0].1 <= corners[1].1 {
        (corners[0], corners[1])
    } else {
        (corners[1], corners[0])
    };

    // Right pair (largest x): TR has smaller y.
    let (tr, br) = if corners[2].1 <= corners[3].1 {
        (corners[2], corners[3])
    } else {
        (corners[3], corners[2])
    };

    [tl, tr, br, bl]
}

/// Average probability inside a rotated quad (fast axis-aligned scan).
fn box_score_fast(prob_data: &[f32], map_w: usize, map_h: usize, corners: &[(f32, f32); 4]) -> f32 {
    let min_x = corners.iter().map(|c| c.0).fold(f32::MAX, f32::min).max(0.0) as usize;
    let max_x = corners
        .iter()
        .map(|c| c.0)
        .fold(f32::MIN, f32::max)
        .min(map_w.saturating_sub(1) as f32) as usize;
    let min_y = corners.iter().map(|c| c.1).fold(f32::MAX, f32::min).max(0.0) as usize;
    let max_y = corners
        .iter()
        .map(|c| c.1)
        .fold(f32::MIN, f32::max)
        .min(map_h.saturating_sub(1) as f32) as usize;

    if min_x > max_x || min_y > max_y {
        return 0.0;
    }

    let mut total = 0.0f32;
    let mut count = 0u32;

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            if point_in_quad((x as f32, y as f32), corners)
                && let Some(&val) = prob_data.get(y * map_w + x)
            {
                total += val;
                count += 1;
            }
        }
    }

    if count > 0 { total / count as f32 } else { 0.0 }
}

/// Test whether a point lies inside a convex quadrilateral (winding test).
fn point_in_quad(p: (f32, f32), quad: &[(f32, f32); 4]) -> bool {
    let mut pos = 0u8;
    let mut neg = 0u8;
    for i in 0..4 {
        let j = (i + 1) % 4;
        let cross = (quad[j].0 - quad[i].0) * (p.1 - quad[i].1) - (quad[j].1 - quad[i].1) * (p.0 - quad[i].0);
        if cross > 0.0 {
            pos += 1;
        }
        if cross < 0.0 {
            neg += 1;
        }
    }
    pos == 0 || neg == 0
}

/// Compute 3×3 perspective transform mapping src points → dst points.
/// Returns 8 coefficients [a,b,c,d,e,f,g,h] such that:
///   `dst_x` = (`a·src_x` + `b·src_y` + c) / (`g·src_x` + `h·src_y` + 1)
///   `dst_y` = (`d·src_x` + `e·src_y` + f) / (`g·src_x` + `h·src_y` + 1)
fn compute_perspective_transform(src: &[(f32, f32); 4], dst: &[(f32, f32); 4]) -> Option<[f64; 8]> {
    let mut mat = [[0.0f64; 9]; 8];
    for i in 0..4 {
        let u = f64::from(src[i].0);
        let v = f64::from(src[i].1);
        let x = f64::from(dst[i].0);
        let y = f64::from(dst[i].1);

        mat[2 * i] = [u, v, 1.0, 0.0, 0.0, 0.0, -u * x, -v * x, x];
        mat[2 * i + 1] = [0.0, 0.0, 0.0, u, v, 1.0, -u * y, -v * y, y];
    }

    solve_8x8(&mut mat)
}

/// Gaussian elimination with partial pivoting for an 8×8 augmented system.
#[allow(clippy::needless_range_loop)]
fn solve_8x8(mat: &mut [[f64; 9]; 8]) -> Option<[f64; 8]> {
    for col in 0..8 {
        // Find pivot.
        let mut max_row = col;
        let mut max_val = mat[col][col].abs();
        for row in (col + 1)..8 {
            let v = mat[row][col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-12 {
            return None;
        }
        mat.swap(col, max_row);

        // Eliminate below.
        let pivot = mat[col][col];
        for row in (col + 1)..8 {
            let factor = mat[row][col] / pivot;
            for j in col..9 {
                mat[row][j] -= factor * mat[col][j];
            }
        }
    }

    // Back substitution.
    let mut result = [0.0f64; 8];
    for row in (0..8).rev() {
        let mut sum = mat[row][8];
        for col in (row + 1)..8 {
            sum -= mat[row][col] * result[col];
        }
        result[row] = sum / mat[row][row];
    }
    Some(result)
}

/// Bilinear interpolation with border replication.
fn bilinear_sample(src: &image::RgbImage, sx: f32, sy: f32, w: u32, h: u32) -> [u8; 3] {
    let sx = sx.clamp(0.0, w.saturating_sub(1) as f32);
    let sy = sy.clamp(0.0, h.saturating_sub(1) as f32);

    let x0 = sx.floor() as u32;
    let y0 = sy.floor() as u32;
    let x1 = (x0 + 1).min(w.saturating_sub(1));
    let y1 = (y0 + 1).min(h.saturating_sub(1));

    let fx = sx - x0 as f32;
    let fy = sy - y0 as f32;

    let p00 = src.get_pixel(x0, y0).0;
    let p10 = src.get_pixel(x1, y0).0;
    let p01 = src.get_pixel(x0, y1).0;
    let p11 = src.get_pixel(x1, y1).0;

    let mut result = [0u8; 3];
    for c in 0..3 {
        let v = f32::from(p00[c]) * (1.0 - fx) * (1.0 - fy)
            + f32::from(p10[c]) * fx * (1.0 - fy)
            + f32::from(p01[c]) * (1.0 - fx) * fy
            + f32::from(p11[c]) * fx * fy;
        result[c] = v.round().clamp(0.0, 255.0) as u8;
    }
    result
}

fn dist_f32(a: (f32, f32), b: (f32, f32)) -> f32 {
    let dx = b.0 - a.0;
    let dy = b.1 - a.1;
    (dx * dx + dy * dy).sqrt()
}

/// Fit a quadrilateral to a convex hull by selecting the 4 vertices with the
/// largest exterior (turning) angles. These are the "sharpest corners" and
/// naturally capture perspective distortion that `minAreaRect` discards.
/// Returns [TL, TR, BR, BL] via `order_points`.
///
/// NOTE: Currently unused — `DBNet`'s binary mask does not preserve reliable
/// perspective geometry.  Kept for future use with edge-based detection.
#[allow(dead_code)]
fn fit_quad_to_hull(hull: &[(f32, f32)]) -> Option<[(f32, f32); 4]> {
    let n = hull.len();
    if n < 4 {
        return None;
    }
    if n == 4 {
        return Some(order_points([hull[0], hull[1], hull[2], hull[3]]));
    }

    // Compute exterior (turning) angle at each hull vertex.
    // The exterior angle is the angle between consecutive edge directions;
    // largest = sharpest corner, smallest = straightest edge.
    let mut angles: Vec<(f32, usize)> = Vec::with_capacity(n);
    for i in 0..n {
        let prev = hull[(i + n - 1) % n];
        let curr = hull[i];
        let next = hull[(i + 1) % n];

        let d_in = (curr.0 - prev.0, curr.1 - prev.1);
        let d_out = (next.0 - curr.0, next.1 - curr.1);

        let dot = d_in.0 * d_out.0 + d_in.1 * d_out.1;
        let len_in = (d_in.0 * d_in.0 + d_in.1 * d_in.1).sqrt();
        let len_out = (d_out.0 * d_out.0 + d_out.1 * d_out.1).sqrt();
        let denom = len_in * len_out;
        let cos_val = if denom > 1e-8 {
            (dot / denom).clamp(-1.0, 1.0)
        } else {
            1.0
        };
        // θ = acos(cos_val) is the angle between consecutive edge direction
        // vectors d_in and d_out.  For a convex polygon vertex with interior
        // angle α, θ = π − α, i.e. θ IS the exterior (turning) angle.
        // Sharpest corners have the LARGEST θ.
        let turning = cos_val.acos();
        angles.push((turning, i));
    }

    // Pick the 4 vertices with the largest turning angle.
    angles.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut indices: Vec<usize> = angles.iter().take(4).map(|&(_, idx)| idx).collect();
    // Sort by position on the hull to preserve winding order.
    indices.sort_unstable();

    Some(order_points([
        hull[indices[0]],
        hull[indices[1]],
        hull[indices[2]],
        hull[indices[3]],
    ]))
}

/// Expand a quad outward by `distance` along each edge's outward normal.
/// Uses polygon edge offset with line-line intersection for new corners.
/// Quad must be in [TL, TR, BR, BL] order (clockwise in image coords).
///
/// NOTE: Currently unused — see `fit_quad_to_hull` note.
#[allow(dead_code)]
fn unclip_quad(quad: &[(f32, f32); 4], distance: f32) -> [(f32, f32); 4] {
    let n = 4usize;
    // For each edge, compute the offset edge (shifted outward by `distance`).
    // CW winding → outward normal of edge (a, b) is (dy, -dx) / len.
    let mut offset_edges: Vec<((f32, f32), (f32, f32))> = Vec::with_capacity(n);
    for i in 0..n {
        let a = quad[i];
        let b = quad[(i + 1) % n];
        let dx = b.0 - a.0;
        let dy = b.1 - a.1;
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-8 {
            offset_edges.push((a, b));
            continue;
        }
        let nx = dy / len * distance;
        let ny = -dx / len * distance;
        offset_edges.push(((a.0 + nx, a.1 + ny), (b.0 + nx, b.1 + ny)));
    }

    // New corners = intersection of adjacent offset edges.
    let mut result = [(0.0f32, 0.0f32); 4];
    for i in 0..n {
        let prev = (i + n - 1) % n;
        if let Some(pt) = line_intersect(
            offset_edges[prev].0,
            offset_edges[prev].1,
            offset_edges[i].0,
            offset_edges[i].1,
        ) {
            result[i] = pt;
        } else {
            // Fallback: use the offset endpoint
            result[i] = offset_edges[i].0;
        }
    }
    result
}

/// Intersection of two lines (each defined by two points).
#[allow(dead_code)]
fn line_intersect(a1: (f32, f32), a2: (f32, f32), b1: (f32, f32), b2: (f32, f32)) -> Option<(f32, f32)> {
    let d1 = (a2.0 - a1.0, a2.1 - a1.1);
    let d2 = (b2.0 - b1.0, b2.1 - b1.1);
    let cross = d1.0 * d2.1 - d1.1 * d2.0;
    if cross.abs() < 1e-8 {
        return None; // parallel
    }
    let t = ((b1.0 - a1.0) * d2.1 - (b1.1 - a1.1) * d2.0) / cross;
    Some((a1.0 + t * d1.0, a1.1 + t * d1.1))
}

fn load_session(path: &str) -> Result<Session, String> {
    crate::build_session(path).map_err(|e| format!("Load model {path}: {e}"))
}
