//! MERIT — Metacognitive layer for the AgentRust agent loop.
//!
//! Four loosely-coupled, training-free layers:
//!
//! 1. [`SelfBelief`] + [`MetaController`] — Bayesian outcome monitoring and
//!    Expected-Free-Energy meta-action policy (`Act / Reflect / Decompose /
//!    Escalate / Abort`).
//! 2. [`CoTMonitor`] — sliding-window detector for low-quality reasoning
//!    and tool-parameter loops.
//! 3. [`SelfModelStore`] (+ [`SelfModelMemoryAdapter`]) — structured
//!    self-knowledge as `SelfProposition`s, persisted via the
//!    `MemoryManager`.
//! 4. [`SchemaReviser`] — periodic clustering over recent failures that
//!    proposes new dimensions or propositions.
//!
//! [`MetacognitionEngine`] is the single façade the agent loop talks to.
//!
//! This module is a faithful port of `include/agent/*.hpp` +
//! `src/agent/*.cpp` from the sibling AgentCpp project.

pub mod cot_monitor;
pub mod engine;
pub mod meta_controller;
pub mod schema_reviser;
pub mod self_belief;
pub mod self_model_memory_adapter;
pub mod self_model_store;
pub mod self_proposition;

pub use cot_monitor::{CoTConfig, CoTMonitor};
pub use engine::{EngineConfig, MetacognitionEngine, MetacognitionEvent};
pub use meta_controller::{
    IterationObservation, MetaAction, MetaConfig, MetaController, MetaDecision, StopReason,
};
pub use schema_reviser::{
    FailureEvent, ProposalKind, ReviserConfig, SchemaProposal, SchemaReviser,
};
pub use self_belief::{BetaParam, SelfBelief};
pub use self_model_memory_adapter::{
    SelfModelMemoryAdapter, DEFAULT_BANK_ID as SELF_MODEL_BANK, SELFPROP_TAG,
};
pub use self_model_store::SelfModelStore;
pub use self_proposition::{make_proposition_id, SelfProposition};
