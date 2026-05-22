//! AgentRust — a general-purpose autonomous agent runtime in Rust.
//!
//! AgentRust is **not** specific to coding. It is a domain-agnostic agent
//! platform: a CLI / GUI / Web shell wrapped around a plan-execute-reflect
//! loop, a swappable LLM client (OpenAI-compatible), a pluggable tool
//! registry organised into capability bundles, long-term SQLite memory,
//! and a metacognitive controller. Coding is one of several built-in
//! personas — research, writing, analysis, and desktop operation are
//! first-class siblings.
//!
//! Highlights:
//! - Async-first runtime (Tokio)
//! - Personas: Coder / Researcher / Writer / Analyst / Operator / General
//! - Capability bundles: Coding / Knowledge / Desktop / Web / Communication
//! - Autonomous `task` loop with plan/execute/reflect
//! - Multimodal chat messages (text + image)
//! - MCP protocol, plugins, skills, channels
//! - SQLite-backed memory with BM25 + embedding hybrid recall
//! - Metacognition: Bayesian self-belief, EFE meta-controller, CoT monitor
//! - Native TUI (ratatui), desktop GUI (egui), Web (axum), WASM bindings
//! - Fluent-based i18n

pub mod agent;
pub mod cli;
pub mod tools;
pub mod api;
pub mod config;
pub mod state;
pub mod mcp;
pub mod voice;
pub mod memory;
pub mod metacognition;
pub mod plugins;
pub mod utils;
pub mod services;
pub mod session;
pub mod terminal;
pub mod advanced;
pub mod skills;
pub mod channels;

// Feature-gated modules
#[cfg(feature = "wasm")]
pub mod wasm;
#[cfg(feature = "gui-egui")]
pub mod gui;
#[cfg(feature = "web")]
pub mod web;
#[cfg(feature = "i18n")]
pub mod i18n;

pub use agent::{AgentRunner, Goal, GoalStatus, Persona, PersonaProfile, RunOutcome, RunStep, StepKind, StopReason};
pub use cli::Cli;
pub use state::AppState;
pub use tools::ToolRegistry;
pub use api::{ApiClient, AnthropicClient, ChatMessage};
pub use config::Settings;
pub use mcp::McpManager;
pub use voice::VoiceInput;
pub use memory::MemoryManager;
pub use plugins::PluginManager;
pub use skills::{Skill, SkillRegistry, SkillExecutor, SkillContext, SkillParams, SkillResult, SkillError, SkillCategory};
pub use channels::{ChannelBus, ChannelInfo, ChannelMessage};
pub use metacognition::{
    MetaAction, MetaDecision, MetacognitionEngine, MetacognitionEvent, SelfBelief,
    SelfModelStore, SelfProposition,
};

// Feature-gated re-exports
#[cfg(feature = "wasm")]
pub use wasm::AgentRustWasm;
#[cfg(feature = "gui-egui")]
pub use gui::AgentRustApp;
#[cfg(feature = "web")]
pub use web::WebServer;
#[cfg(feature = "i18n")]
pub use i18n::Translator;
