//! Speech-to-text service using sherpa-onnx.
//!
//! Two modes:
//! - **Offline** (SenseVoice): high-quality batch transcription from WAV audio.
//! - **Streaming** (Zipformer OnlineRecognizer): real-time partial results,
//!   with `SenseVoice` refinement on detected endpoints.

#![allow(unsafe_code)]

use std::path::Path;
use std::sync::Arc;

use sherpa_onnx::{
    OfflineRecognizer, OfflineRecognizerConfig, OfflineSenseVoiceModelConfig, OnlineRecognizer, OnlineRecognizerConfig,
    OnlineTransducerModelConfig,
};

// ── Model catalogue ──────────────────────────────────────────────────────────

/// Available STT models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SttModel {
    /// `SenseVoice` int8 — supports zh/en/ja/ko/yue, ~230 MB (offline)
    SenseVoiceInt8,
    /// Streaming Zipformer bilingual zh-en — real-time streaming
    StreamingZipformerBilingualZhEn,
}

impl SttModel {
    pub fn id(&self) -> &'static str {
        match self {
            Self::SenseVoiceInt8 => "sense-voice-int8",
            Self::StreamingZipformerBilingualZhEn => "streaming-zipformer-bilingual-zh-en",
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            Self::SenseVoiceInt8 => "SenseVoice (int8, ~230 MB)",
            Self::StreamingZipformerBilingualZhEn => "Streaming Zipformer bilingual zh-en (~70 MB)",
        }
    }

    /// Directory name inside `models_dir/stt/`.
    pub fn dir_name(&self) -> &'static str {
        match self {
            Self::SenseVoiceInt8 => "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
            Self::StreamingZipformerBilingualZhEn => "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20",
        }
    }

    pub fn download_url(&self) -> &'static str {
        match self {
            Self::SenseVoiceInt8 => {
                "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2"
            }
            Self::StreamingZipformerBilingualZhEn => {
                "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2"
            }
        }
    }

    /// Whether this model is for streaming (online) recognition.
    pub fn is_streaming(&self) -> bool {
        matches!(self, Self::StreamingZipformerBilingualZhEn)
    }

    /// Tokens filename inside the model directory.
    pub fn tokens_file(&self) -> &'static str {
        "tokens.txt"
    }
}

pub const DEFAULT_MODEL: SttModel = SttModel::SenseVoiceInt8;
pub const STREAMING_MODEL: SttModel = SttModel::StreamingZipformerBilingualZhEn;
pub const ALL_MODELS: &[SttModel] = &[SttModel::SenseVoiceInt8, SttModel::StreamingZipformerBilingualZhEn];

// ── Offline SttService (SenseVoice) ──────────────────────────────────────────

pub struct SttService {
    recognizer: Arc<SendRecognizer>,
}

/// Wrapper to assert Send + Sync for `OfflineRecognizer`.
/// sherpa-onnx's ONNX runtime is thread-safe internally; the raw pointer
/// just lacks the marker traits.
struct SendRecognizer(OfflineRecognizer);
unsafe impl Send for SendRecognizer {}
unsafe impl Sync for SendRecognizer {}

impl std::ops::Deref for SendRecognizer {
    type Target = OfflineRecognizer;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Clone for SttService {
    fn clone(&self) -> Self {
        Self {
            recognizer: Arc::clone(&self.recognizer),
        }
    }
}

impl SttService {
    /// Load an STT model from the given directory.
    pub fn new(models_dir: &str, model: SttModel) -> Result<Self, String> {
        let model_dir = format!("{}/stt/{}", models_dir, model.dir_name());
        let model_path = format!("{model_dir}/model.int8.onnx");
        let tokens_path = format!("{}/{}", model_dir, model.tokens_file());

        if !Path::new(&model_path).exists() {
            return Err(format!("STT model not found: {model_path}"));
        }
        if !Path::new(&tokens_path).exists() {
            return Err(format!("STT tokens not found: {tokens_path}"));
        }

        let mut config = OfflineRecognizerConfig::default();
        config.model_config.sense_voice = OfflineSenseVoiceModelConfig {
            model: Some(model_path),
            language: Some("auto".into()),
            use_itn: true,
        };
        config.model_config.tokens = Some(tokens_path);
        config.model_config.num_threads = 4;

        let recognizer = OfflineRecognizer::create(&config)
            .ok_or_else(|| "Failed to create STT recognizer (check model paths)".to_string())?;

        Ok(Self {
            recognizer: Arc::new(SendRecognizer(recognizer)),
        })
    }

    /// Transcribe raw WAV audio bytes to text.
    pub fn transcribe(&self, wav_bytes: &[u8]) -> Result<String, String> {
        let (samples, sample_rate) = decode_wav_to_f32(wav_bytes)?;

        let duration_secs = samples.len() as f64 / f64::from(sample_rate);
        tracing::info!(
            "STT: {} samples, {}Hz, {:.2}s",
            samples.len(),
            sample_rate,
            duration_secs,
        );

        if samples.len() < (sample_rate as usize / 2) {
            return Err(format!("Audio too short ({duration_secs:.1}s, need at least 0.5s)"));
        }

        let stream = self.recognizer.create_stream();
        stream.accept_waveform(sample_rate, &samples);
        self.recognizer.decode(&stream);

        let result = stream
            .get_result()
            .ok_or_else(|| "Failed to get recognition result".to_string())?;

        let text = result.text.trim().to_string();
        tracing::info!("STT result: {:?}", text);
        Ok(text)
    }

    /// Transcribe raw f32 PCM samples (mono, 16 kHz) directly.
    pub fn transcribe_pcm(&self, samples: &[f32], sample_rate: i32) -> Result<String, String> {
        let duration_secs = samples.len() as f64 / f64::from(sample_rate);
        if samples.len() < (sample_rate as usize / 2) {
            return Err(format!("Audio too short ({duration_secs:.1}s, need at least 0.5s)"));
        }

        let stream = self.recognizer.create_stream();
        stream.accept_waveform(sample_rate, samples);
        self.recognizer.decode(&stream);

        let result = stream
            .get_result()
            .ok_or_else(|| "Failed to get recognition result".to_string())?;

        Ok(result.text.trim().to_string())
    }
}

// ── Streaming SttService (Zipformer OnlineRecognizer) ────────────────────────

/// Thread-safe wrapper for `OnlineRecognizer`.
struct SendOnlineRecognizer(OnlineRecognizer);
unsafe impl Send for SendOnlineRecognizer {}
unsafe impl Sync for SendOnlineRecognizer {}

impl std::ops::Deref for SendOnlineRecognizer {
    type Target = OnlineRecognizer;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Thread-safe wrapper for `OnlineStream`.
/// Only accessed from a single async task at a time.
pub struct SendOnlineStream(sherpa_onnx::OnlineStream);
unsafe impl Send for SendOnlineStream {}
unsafe impl Sync for SendOnlineStream {}

impl std::ops::Deref for SendOnlineStream {
    type Target = sherpa_onnx::OnlineStream;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Streaming STT service using `OnlineRecognizer` (Zipformer transducer).
pub struct StreamingSttService {
    recognizer: Arc<SendOnlineRecognizer>,
}

impl Clone for StreamingSttService {
    fn clone(&self) -> Self {
        Self {
            recognizer: Arc::clone(&self.recognizer),
        }
    }
}

/// Find the best matching ONNX file for a transducer component.
/// Prefers int8 → epoch naming → simple naming.
fn find_onnx_file(dir: &str, component: &str) -> Option<String> {
    let candidates = [
        format!("{dir}/{component}-epoch-99-avg-1.int8.onnx"),
        format!("{dir}/{component}.int8.onnx"),
        format!("{dir}/{component}-epoch-99-avg-1.onnx"),
        format!("{dir}/{component}.onnx"),
    ];
    candidates.into_iter().find(|p| Path::new(p).exists())
}

impl StreamingSttService {
    /// Load the streaming Zipformer model.
    pub fn new(models_dir: &str) -> Result<Self, String> {
        let model = STREAMING_MODEL;
        let model_dir = format!("{}/stt/{}", models_dir, model.dir_name());
        let tokens_path = format!("{}/{}", model_dir, model.tokens_file());

        let encoder = find_onnx_file(&model_dir, "encoder")
            .ok_or_else(|| format!("Streaming encoder not found in {model_dir}"))?;
        let decoder = find_onnx_file(&model_dir, "decoder")
            .ok_or_else(|| format!("Streaming decoder not found in {model_dir}"))?;
        let joiner =
            find_onnx_file(&model_dir, "joiner").ok_or_else(|| format!("Streaming joiner not found in {model_dir}"))?;

        if !Path::new(&tokens_path).exists() {
            return Err(format!("Streaming tokens not found: {tokens_path}"));
        }

        tracing::info!("Loading streaming STT: encoder={encoder}, decoder={decoder}, joiner={joiner}");

        let mut config = OnlineRecognizerConfig::default();
        config.model_config.transducer = OnlineTransducerModelConfig {
            encoder: Some(encoder),
            decoder: Some(decoder),
            joiner: Some(joiner),
        };
        config.model_config.tokens = Some(tokens_path);
        config.model_config.num_threads = 2;
        config.decoding_method = Some("greedy_search".into());
        config.enable_endpoint = true;
        // Endpoint rules: detect end-of-utterance
        config.rule1_min_trailing_silence = 2.4; // long silence → endpoint
        config.rule2_min_trailing_silence = 0.8; // shorter silence after speech
        config.rule3_min_utterance_length = 20.0;

        let recognizer =
            OnlineRecognizer::create(&config).ok_or_else(|| "Failed to create streaming recognizer".to_string())?;

        tracing::info!("Streaming STT model loaded");
        Ok(Self {
            recognizer: Arc::new(SendOnlineRecognizer(recognizer)),
        })
    }

    /// Create a new online stream for a streaming session.
    pub fn create_stream(&self) -> SendOnlineStream {
        SendOnlineStream(self.recognizer.create_stream())
    }

    /// Feed audio samples to the stream.
    pub fn feed_audio(&self, stream: &SendOnlineStream, samples: &[f32]) {
        stream.accept_waveform(16000, samples);
    }

    /// Decode any ready frames. Returns true if decoding happened.
    pub fn decode_if_ready(&self, stream: &SendOnlineStream) -> bool {
        let mut decoded = false;
        while self.recognizer.is_ready(stream) {
            self.recognizer.decode(stream);
            decoded = true;
        }
        decoded
    }

    /// Get the current partial recognition text.
    pub fn get_partial(&self, stream: &SendOnlineStream) -> Option<String> {
        self.recognizer
            .get_result(stream)
            .map(|r| r.text.trim().to_string())
            .filter(|t| !t.is_empty())
    }

    /// Check if an endpoint (end-of-utterance) was detected.
    pub fn is_endpoint(&self, stream: &SendOnlineStream) -> bool {
        self.recognizer.is_endpoint(stream)
    }

    /// Reset the stream for a new utterance (after endpoint).
    pub fn reset(&self, stream: &SendOnlineStream) {
        self.recognizer.reset(stream);
    }

    /// Signal end-of-input.
    pub fn finish(&self, stream: &SendOnlineStream) {
        stream.input_finished();
    }
}

// ── Shared helpers ───────────────────────────────────────────────────────────

/// Decode WAV bytes to f32 mono samples, returning (samples, `sample_rate`).
fn decode_wav_to_f32(wav_bytes: &[u8]) -> Result<(Vec<f32>, i32), String> {
    let cursor = std::io::Cursor::new(wav_bytes);
    let reader = hound::WavReader::new(cursor).map_err(|e| format!("Failed to read WAV: {e}"))?;

    let spec = reader.spec();
    let sample_rate = spec.sample_rate as i32;
    let channels = spec.channels as usize;

    let raw_samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.into_samples::<f32>().filter_map(Result::ok).collect(),
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            let max_val = (1u32 << (bits - 1)) as f32;
            reader
                .into_samples::<i32>()
                .filter_map(Result::ok)
                .map(|s| s as f32 / max_val)
                .collect()
        }
    };

    // Mix to mono if multi-channel
    let mono: Vec<f32> = if channels > 1 {
        raw_samples
            .chunks(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    } else {
        raw_samples
    };

    Ok((mono, sample_rate))
}

/// Convert Int16 LE bytes to f32 samples.
pub fn int16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| {
            let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
            f32::from(sample) / 32768.0
        })
        .collect()
}

// ── Model status ─────────────────────────────────────────────────────────────

/// Check if a specific STT model exists on disk.
pub fn model_exists(models_dir: &str, model: SttModel) -> bool {
    let dir = format!("{}/stt/{}", models_dir, model.dir_name());
    let tokens = format!("{}/{}", dir, model.tokens_file());
    if !Path::new(&tokens).exists() {
        return false;
    }
    if model.is_streaming() {
        // Need encoder + decoder + joiner
        find_onnx_file(&dir, "encoder").is_some()
            && find_onnx_file(&dir, "decoder").is_some()
            && find_onnx_file(&dir, "joiner").is_some()
    } else {
        // SenseVoice: single model.int8.onnx
        Path::new(&format!("{dir}/model.int8.onnx")).exists()
    }
}

/// Return status info for all models.
pub fn models_status(models_dir: &str) -> Vec<SttModelStatus> {
    ALL_MODELS
        .iter()
        .map(|m| SttModelStatus {
            id: m.id().to_string(),
            name: m.display_name().to_string(),
            ready: model_exists(models_dir, *m),
        })
        .collect()
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SttModelStatus {
    pub id: String,
    pub name: String,
    pub ready: bool,
}
