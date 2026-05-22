//! Deprecated — replaced by `consolidate.rs` in the native engine port.
//!
//! Earlier iterations of this module triggered consolidation via an HTTP
//! call to a running Hindsight server. The current implementation is a
//! native LLM-driven pipeline that lives in `consolidate.rs`. This file is
//! retained only so external references to `crate::memory::consolidation`
//! do not break; it intentionally exports nothing and is not declared as a
//! submodule of `mod.rs`.
