//! Convert between tokimo-media-intelligence' native types and ai-worker-protocol wire types.

use tokimo_media_intelligence as rm;
use tokimo_media_intelligence::worker::protocol::types as wire;

pub fn accel_to_wire(p: rm::AccelProvider) -> wire::AccelProvider {
    match p {
        rm::AccelProvider::Cuda => wire::AccelProvider::Cuda,
        rm::AccelProvider::ROCm => wire::AccelProvider::Rocm,
        rm::AccelProvider::CoreML => wire::AccelProvider::CoreMl,
        rm::AccelProvider::DirectML => wire::AccelProvider::DirectMl,
        rm::AccelProvider::Cpu => wire::AccelProvider::Cpu,
    }
}

pub fn status_to_wire(s: rm::MediaIntelligenceStatus) -> wire::MediaIntelligenceStatus {
    wire::MediaIntelligenceStatus {
        accel_provider: accel_to_wire(s.accel_provider),
        ocr_loaded: s.ocr_loaded,
        clip_loaded: s.clip_loaded,
        face_loaded: s.face_loaded,
        stt_loaded: s.stt_loaded,
    }
}

pub fn ocr_item_to_wire(it: rm::ocr::OcrItem) -> wire::OcrItem {
    wire::OcrItem {
        text: it.text,
        score: it.score,
        x: it.x,
        y: it.y,
        w: it.w,
        h: it.h,
        angle: it.angle,
        corners: it.corners,
        paragraph_id: it.paragraph_id,
        char_positions: it.char_positions,
    }
}

pub fn ocr_model_info_to_wire(m: rm::ocr_manager::OcrModelInfo) -> wire::OcrModelInfo {
    wire::OcrModelInfo {
        id: m.id.to_string(),
        display_name: m.display_name.to_string(),
        loaded: m.loaded,
    }
}

pub fn face_to_wire(f: rm::face::FaceDetection) -> wire::FaceDetection {
    wire::FaceDetection {
        embedding: f.embedding,
        x: f.x,
        y: f.y,
        w: f.w,
        h: f.h,
        confidence: f.confidence,
    }
}

pub fn tag_to_wire(t: rm::clip_categories::TagResult) -> wire::TagResult {
    wire::TagResult {
        category: t.category.to_string(),
        icon: t.icon.to_string(),
        subcategory: t.subcategory.to_string(),
        score: t.score,
    }
}

pub fn stt_status_to_wire(s: rm::stt::SttModelStatus) -> wire::SttModelStatus {
    wire::SttModelStatus {
        id: s.id,
        name: s.name,
        ready: s.ready,
    }
}

pub fn category_from_wire(c: wire::ModelCategory) -> rm::models::ModelCategory {
    match c {
        wire::ModelCategory::OcrServer => rm::models::ModelCategory::OcrServer,
        wire::ModelCategory::OcrMobile => rm::models::ModelCategory::OcrMobile,
        wire::ModelCategory::Clip => rm::models::ModelCategory::Clip,
        wire::ModelCategory::Face => rm::models::ModelCategory::Face,
    }
}
