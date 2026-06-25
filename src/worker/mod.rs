//! AI worker sidecar: out-of-process execution of tokimo-media-intelligence AI inference.
//!
//! This module will house three layers that cooperate to let the heavy
//! ONNX / sherpa-onnx models run in a separate OS process so memory can be
//! reclaimed by exiting the process:
//!
//! - `protocol` — wire types + transport abstraction (RPC over UDS / HTTP)
//! - `server`   — binary-side handlers backing those RPCs (future commit)
//! - `client`   — in-proc client + supervisor used by `tokimo-server` (future commit)
//!
//! The binary target lives at `src/bin/tokimo-media-intelligence-worker/`.

pub mod client;
pub mod protocol;
