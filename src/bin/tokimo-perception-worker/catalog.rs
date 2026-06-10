//! Unified model catalog for the settings UI.
//!
//! Aggregates status from the native [`AiService`] (OCR / CLIP / Face / STT) and
//! — if configured — the Python sidecar `/models` endpoint, and returns a
//! fully localized [`ModelCatalog`] per the caller's preferred languages.
//!
//! The catalog is the single source of truth consumed by the main server's
//! `GET /api/perception/catalog` pass-through; everything user-visible (section
//! titles, model names, attribute labels, capability tags, action labels) is
//! localized *here* so downstream consumers remain pure data routers.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use serde::Deserialize;
use tokimo_perception::AiService;
use tokimo_perception::ocr_manager::MODEL_GOT_OCR_2;
use tokimo_perception::worker::protocol::types as wire;

// ---------------- Locale bundle ----------------

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct LocaleBundle {
    sections: HashMap<String, SectionText>,
    models: HashMap<String, ModelText>,
    attrs: HashMap<String, String>,
    capabilities: HashMap<String, String>,
    actions: HashMap<String, String>,
}

#[derive(Debug, Deserialize, Clone)]
struct SectionText {
    title: String,
    description: String,
}

#[derive(Debug, Deserialize, Clone)]
struct ModelText {
    name: String,
    description: String,
}

macro_rules! locale_entry {
    ($tag:literal) => {
        ($tag, include_str!(concat!("../../../locales/", $tag, ".json")))
    };
}

static LOCALES: OnceLock<HashMap<&'static str, LocaleBundle>> = OnceLock::new();

fn locales() -> &'static HashMap<&'static str, LocaleBundle> {
    LOCALES.get_or_init(|| {
        let raw: [(&str, &str); 9] = [
            locale_entry!("zh-CN"),
            locale_entry!("en-US"),
            locale_entry!("ja-JP"),
            locale_entry!("ko-KR"),
            locale_entry!("fr-FR"),
            locale_entry!("de-DE"),
            locale_entry!("es-ES"),
            locale_entry!("pt-BR"),
            locale_entry!("ru-RU"),
        ];
        let mut map = HashMap::new();
        for (tag, body) in raw {
            match serde_json::from_str::<LocaleBundle>(body) {
                Ok(b) => {
                    map.insert(tag, b);
                }
                Err(e) => tracing::error!("catalog: failed to parse locale {tag}: {e}"),
            }
        }
        map
    })
}

const DEFAULT_LANG: &str = "en-US";

fn pick_locale(prefs: &[String]) -> &'static LocaleBundle {
    let map = locales();
    for p in prefs {
        if let Some(b) = map.get(p.as_str()) {
            return b;
        }
        let lang = p.split(['-', '_']).next().unwrap_or("");
        if !lang.is_empty() {
            for (tag, b) in map {
                if tag.starts_with(lang) {
                    return b;
                }
            }
        }
    }
    map.get(DEFAULT_LANG).expect("en-US locale must be present")
}

fn t_section(l: &LocaleBundle, key: &str) -> SectionText {
    l.sections.get(key).cloned().unwrap_or(SectionText {
        title: key.to_string(),
        description: String::new(),
    })
}

fn t_model(l: &LocaleBundle, key: &str) -> ModelText {
    l.models.get(key).cloned().unwrap_or(ModelText {
        name: key.to_string(),
        description: String::new(),
    })
}

fn t_attr(l: &LocaleBundle, key: &str) -> String {
    l.attrs.get(key).cloned().unwrap_or_else(|| key.to_string())
}

fn t_cap(l: &LocaleBundle, key: &str) -> String {
    l.capabilities.get(key).cloned().unwrap_or_else(|| key.to_string())
}

// ---------------- Public entry ----------------

pub async fn build_catalog(ai: &Arc<AiService>, req: &wire::CatalogRequest) -> wire::ModelCatalog {
    let l = pick_locale(&req.languages);
    let mut sections = Vec::new();

    sections.push(build_ocr_section(ai, l).await);
    sections.push(build_clip_section(ai, l));
    sections.push(build_face_section(ai, l));
    sections.push(build_stt_section(ai, l));
    // Sidecar section is best-effort; we return an empty stub when not configured
    // so the UI still renders the section heading (can be hidden if empty).
    sections.push(build_sidecar_section(ai, l));

    wire::ModelCatalog { sections }
}

// ---------------- OCR ----------------

async fn build_ocr_section(ai: &Arc<AiService>, l: &LocaleBundle) -> wire::CatalogSection {
    let text = t_section(l, "ocr");
    let mut models = Vec::new();
    for m in ai.ocr_available_models() {
        let cat_id = format!("ocr.{}", m.id);
        let txt = t_model(l, &cat_id);
        let is_mobile = m.id == "pp-ocrv5-mobile";
        let is_sidecar = m.id == MODEL_GOT_OCR_2;
        let ready = if is_sidecar {
            ai.sidecar().is_ready(m.id).await
        } else if is_mobile {
            ai.ocr_mobile_models_ready()
        } else {
            ai.ocr_server_models_ready()
        };
        let name = if txt.name == cat_id {
            m.display_name.to_string()
        } else {
            txt.name
        };
        models.push(wire::CatalogModel {
            id: cat_id,
            name,
            description: txt.description,
            size_mb: None,
            attrs: vec![wire::CatalogAttr {
                key: "provider".into(),
                label: t_attr(l, "provider"),
                value: if is_sidecar {
                    t_attr(l, "provider_sidecar")
                } else {
                    t_attr(l, "provider_native")
                },
            }],
            capabilities: vec![t_cap(l, "text"), t_cap(l, "blocks")],
            provider: if is_sidecar {
                "python-sidecar".into()
            } else {
                "rust-native".into()
            },
            state: if ready {
                wire::ModelState::Ready
            } else {
                wire::ModelState::NotDownloaded
            },
            actions: if ready {
                vec![wire::ModelAction::Remove]
            } else {
                vec![wire::ModelAction::Download]
            },
        });
    }
    wire::CatalogSection {
        id: "ocr".into(),
        title: text.title,
        description: text.description,
        icon: "scan-text".into(),
        models,
    }
}

// ---------------- CLIP ----------------

fn build_clip_section(ai: &Arc<AiService>, l: &LocaleBundle) -> wire::CatalogSection {
    let text = t_section(l, "clip");
    let mut models = Vec::new();
    if ai.is_clip_enabled() {
        let ready = ai.clip_models_ready();
        let id = "clip.default".to_string();
        let txt = t_model(l, &id);
        models.push(wire::CatalogModel {
            id,
            name: txt.name,
            description: txt.description,
            size_mb: None,
            attrs: vec![wire::CatalogAttr {
                key: "provider".into(),
                label: t_attr(l, "provider"),
                value: t_attr(l, "provider_native"),
            }],
            capabilities: vec![t_cap(l, "multilingual")],
            provider: "rust-native".into(),
            state: if ready {
                wire::ModelState::Ready
            } else {
                wire::ModelState::NotDownloaded
            },
            actions: if ready {
                vec![wire::ModelAction::Remove]
            } else {
                vec![wire::ModelAction::Download]
            },
        });
    }
    wire::CatalogSection {
        id: "clip".into(),
        title: text.title,
        description: text.description,
        icon: "image".into(),
        models,
    }
}

// ---------------- Face ----------------

fn build_face_section(ai: &Arc<AiService>, l: &LocaleBundle) -> wire::CatalogSection {
    let text = t_section(l, "face");
    let mut models = Vec::new();
    if ai.is_face_enabled() {
        let ready = ai.face_models_ready();
        let id = "face.default".to_string();
        let txt = t_model(l, &id);
        models.push(wire::CatalogModel {
            id,
            name: txt.name,
            description: txt.description,
            size_mb: None,
            attrs: vec![wire::CatalogAttr {
                key: "provider".into(),
                label: t_attr(l, "provider"),
                value: t_attr(l, "provider_native"),
            }],
            capabilities: Vec::new(),
            provider: "rust-native".into(),
            state: if ready {
                wire::ModelState::Ready
            } else {
                wire::ModelState::NotDownloaded
            },
            actions: if ready {
                vec![wire::ModelAction::Remove]
            } else {
                vec![wire::ModelAction::Download]
            },
        });
    }
    wire::CatalogSection {
        id: "face".into(),
        title: text.title,
        description: text.description,
        icon: "user-round".into(),
        models,
    }
}

// ---------------- STT ----------------

fn build_stt_section(ai: &Arc<AiService>, l: &LocaleBundle) -> wire::CatalogSection {
    let text = t_section(l, "stt");
    let models = if ai.is_stt_enabled() {
        ai.stt_models_status()
            .into_iter()
            .map(|m| {
                let cat_id = format!("stt.{}", m.id);
                let txt = t_model(l, &cat_id);
                let caps = if m.id.contains("streaming") {
                    vec![t_cap(l, "realtime"), t_cap(l, "multilingual")]
                } else {
                    vec![t_cap(l, "multilingual")]
                };
                wire::CatalogModel {
                    id: cat_id,
                    name: if txt.name.starts_with("stt.") {
                        m.name.clone()
                    } else {
                        txt.name
                    },
                    description: txt.description,
                    size_mb: None,
                    attrs: vec![wire::CatalogAttr {
                        key: "provider".into(),
                        label: t_attr(l, "provider"),
                        value: t_attr(l, "provider_native"),
                    }],
                    capabilities: caps,
                    provider: "rust-native".into(),
                    state: if m.ready {
                        wire::ModelState::Ready
                    } else {
                        wire::ModelState::NotDownloaded
                    },
                    actions: if m.ready {
                        vec![wire::ModelAction::Remove]
                    } else {
                        vec![wire::ModelAction::Download]
                    },
                }
            })
            .collect()
    } else {
        Vec::new()
    };
    wire::CatalogSection {
        id: "stt".into(),
        title: text.title,
        description: text.description,
        icon: "mic".into(),
        models,
    }
}

// ---------------- Sidecar ----------------

fn build_sidecar_section(_ai: &Arc<AiService>, l: &LocaleBundle) -> wire::CatalogSection {
    // For v1 we return an empty sidecar section. Aggregating the Python sidecar's
    // `/models` endpoint is tracked as a follow-up — the architecture already
    // supports it (CatalogModel.provider = "python-sidecar", id prefix `sidecar.*`).
    let text = t_section(l, "sidecar");
    wire::CatalogSection {
        id: "sidecar".into(),
        title: text.title,
        description: text.description,
        icon: "boxes".into(),
        models: Vec::new(),
    }
}

// ---------------- Action dispatch ----------------

/// Parse a catalog model id (`<section>.<slug>`) into its routing hint.
pub enum ModelRoute {
    OcrServer,
    OcrMobile,
    Clip,
    Face,
    Stt(String),
    #[allow(dead_code)]
    Sidecar(String),
    Unknown,
}

pub fn route_for(model_id: &str) -> ModelRoute {
    let Some((section, slug)) = model_id.split_once('.') else {
        return ModelRoute::Unknown;
    };
    match section {
        "ocr" if slug == "pp-ocrv5-mobile" => ModelRoute::OcrMobile,
        "ocr" if slug == MODEL_GOT_OCR_2 => ModelRoute::Sidecar(slug.to_string()),
        "ocr" => ModelRoute::OcrServer,
        "clip" => ModelRoute::Clip,
        "face" => ModelRoute::Face,
        "stt" => ModelRoute::Stt(slug.to_string()),
        "sidecar" => ModelRoute::Sidecar(slug.to_string()),
        _ => ModelRoute::Unknown,
    }
}
