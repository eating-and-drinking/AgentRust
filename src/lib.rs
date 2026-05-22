//! AgentRust Rust - High-performance CLI for Claude AI
//!
//! A complete Rust implementation of AgentRust, featuring:
//! - Async-first architecture with Tokio
//! - Native terminal UI with Ratatui
//! - MCP protocol support
//! - Voice input support
//! - Memory management and team sync
//! - Plugin system
//! - SSH connection support
//! - Remote execution
//! - Project initialization
//! - WebAssembly support for browser environments
//! - Native GUI with egui/eframe
//! - Plugin marketplace web interface
//! - Multi-language i18n support

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
