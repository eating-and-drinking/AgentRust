//! Deprecated.
//!
//! Earlier iterations of this module shipped an HTTP client that delegated
//! every memory operation to a running Hindsight server. The current
//! implementation is a native Rust port of Hindsight's architecture (see
//! `mod.rs`, `model.rs`, `storage.rs`, `retain.rs`, `recall.rs`,
//! `consolidate.rs`). This file is retained only so external references to
//! `crate::memory::client` do not break; it intentionally exports nothing.
