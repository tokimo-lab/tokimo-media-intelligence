//! Worker lifecycle signals.
//!
//! The UDS and HTTP servers emit `WorkerSignal::Activity` on every request so
//! the main task can track idleness; `Shutdown` is emitted after a `/shutdown`
//! RPC has been ack'd.

#[derive(Debug, Clone, Copy)]
pub enum WorkerSignal {
    Activity,
    Shutdown,
}
