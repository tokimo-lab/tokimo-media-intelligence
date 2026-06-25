//! Cooperative cancellation of in-flight ONNX Runtime inference.
//!
//! Problem: `Session::run_async(..)` spawns the actual ORT call on an internal
//! blocking thread. Dropping the returned future does **not** stop ORT — the
//! thread keeps running to completion, which is why cancelled AI jobs used to
//! leave the worker process pinned at 100% CPU.
//!
//! Solution: ORT exposes `RunOptions::terminate()` which sets a cooperative
//! cancellation flag; the running inference observes it at the next safe point
//! and returns early with "Exiting due to terminate flag". We combine that with
//! a task-local `CANCEL_ID` so that intermediate layers (OcrManager, backends,
//! CLIP/Face services) don't need to thread a `cancel_id` parameter through
//! their signatures — only the MediaIntelligenceService public API does.
//!
//! Flow:
//!   1. `MediaIntelligenceService::ocr(img, model, cancel_id)` wraps its body in
//!      [`with_cancel_id`], which sets the task-local.
//!   2. The actual inference site creates `Arc<RunOptions>` and calls
//!      [`register_current`] before `run_async`. If a cancel_id is set, the
//!      `Arc` is stashed in the global registry keyed by that id.
//!   3. When the server decides to cancel a job, it calls
//!      [`cancel_inflight`] (via the worker `/v1/cancel` RPC); we look up the
//!      entry and call `terminate()` on every registered RunOptions.
//!   4. The inference future returns an error; `register_current`'s guard is
//!      dropped, removing the weak ref from the registry.
//!
//! Concurrency: multiple `run_async` calls may share the same `cancel_id`
//! (e.g. OCR detect + recognize for the same photo). The registry stores a
//! `Vec<Weak<RunOptions>>` per id; `cancel_inflight` terminates all of them.

use std::sync::{Arc, LazyLock, Weak};
use std::time::Instant;

use dashmap::DashMap;
use ort::session::{NoSelectedOutputs, RunOptions};

tokio::task_local! {
    /// Current inference's cancel id (if any). Set via [`with_cancel_id`];
    /// read by [`register_current`] at the inference call site.
    pub static CANCEL_ID: Option<String>;
}

type OptRef = Weak<RunOptions<NoSelectedOutputs>>;

static REGISTRY: LazyLock<DashMap<String, Vec<OptRef>>> = LazyLock::new(DashMap::new);

/// Records cancel intent for ids that may not have a live RunOptions yet
/// (pre-session / between-session windows). Entries are GC'd inline by
/// `cancel_inflight` after 120 s.
static CANCELLED: LazyLock<DashMap<String, Instant>> = LazyLock::new(DashMap::new);

/// Returns `true` if a cancel intent was recorded for `id` (regardless of
/// whether any `RunOptions` is currently registered).
pub fn is_cancelled(id: &str) -> bool {
    CANCELLED.contains_key(id)
}

/// RAII guard: while held, the paired `RunOptions` is reachable from
/// [`cancel_inflight`]. On drop, it is removed from the registry.
pub struct InflightGuard {
    id: String,
    options: OptRef,
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        let Some(mut entry) = REGISTRY.get_mut(&self.id) else {
            return;
        };
        entry.retain(|w| !ptr_eq(w, &self.options) && w.strong_count() > 0);
        if entry.is_empty() {
            drop(entry);
            REGISTRY.remove(&self.id);
        }
    }
}

fn ptr_eq(a: &OptRef, b: &OptRef) -> bool {
    Weak::ptr_eq(a, b)
}

/// Register the given `RunOptions` under the current task-local [`CANCEL_ID`].
/// Returns a guard that un-registers on drop. Returns `None` when no cancel id
/// is bound (e.g. when MediaIntelligenceService was called with `cancel_id = None` — the fast
/// path, no registry interaction).
#[must_use]
pub fn register_current(options: &Arc<RunOptions<NoSelectedOutputs>>) -> Option<InflightGuard> {
    let id = CANCEL_ID.try_with(Clone::clone).ok().flatten()?;
    let weak = Arc::downgrade(options);
    REGISTRY.entry(id.clone()).or_default().push(weak.clone());
    Some(InflightGuard { id, options: weak })
}

/// Fire `terminate()` on every `RunOptions` registered under `id`. Returns
/// `true` if at least one live entry was terminated. This is safe to call from
/// any thread and does not block waiting for inference to exit — ORT will
/// notice the flag at its next cancellation point.
///
/// Also records the cancel intent in `CANCELLED` so that pre-session checks
/// via [`is_cancelled`] can bail out even between ORT sessions. Inline GC
/// removes `CANCELLED` entries older than 120 s to bound memory growth.
pub fn cancel_inflight(id: &str) -> bool {
    // Inline GC: remove stale CANCELLED entries (>120 s old).
    let gc_threshold = std::time::Duration::from_mins(2);
    CANCELLED.retain(|_, ts| ts.elapsed() < gc_threshold);

    // Record cancel intent before querying REGISTRY so pre-session checks
    // that run right after this call will observe it.
    CANCELLED.insert(id.to_string(), Instant::now());

    let Some(entry) = REGISTRY.get(id) else {
        return false;
    };
    let mut any = false;
    for weak in entry.iter() {
        if let Some(opts) = weak.upgrade()
            && opts.terminate().is_ok()
        {
            any = true;
        }
    }
    any
}

/// Bind `cancel_id` as the task-local while `fut` runs. Any ORT inference
/// inside that correctly calls [`register_current`] becomes cancellable via
/// [`cancel_inflight`] with the same id.
///
/// Passing `None` is a no-op wrapper (still cheap — task_local scope is
/// essentially free when the value is None).
pub async fn with_cancel_id<F, T>(cancel_id: Option<&str>, fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    CANCEL_ID.scope(cancel_id.map(str::to_string), fut).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn register_returns_none_without_scope() {
        let opts = Arc::new(RunOptions::new().unwrap());
        assert!(register_current(&opts).is_none());
    }

    #[tokio::test]
    async fn cancel_terminates_registered_options() {
        let opts = Arc::new(RunOptions::new().unwrap());
        let opts_clone = Arc::clone(&opts);
        with_cancel_id(Some("test-id"), async move {
            let _guard = register_current(&opts_clone).expect("should register");
            assert!(cancel_inflight("test-id"));
        })
        .await;
        // After scope exits and guard drops, registry is clean.
        assert!(!cancel_inflight("test-id"));
    }
}
