//! Configuration Module

pub mod api_config;
pub mod mcp_config;

pub use api_config::ApiConfig;
pub use mcp_config::{McpConfig, McpServerStatus};

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Main configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    /// API configuration
    pub api: ApiConfig,
    /// MCP server configurations
    pub mcp_servers: Vec<McpConfig>,
    /// Model selection
    pub model: String,
    /// Enable verbose logging
    pub verbose: bool,
    /// Working directory
    pub working_dir: PathBuf,
    /// Memory settings
    pub memory: MemorySettings,
    /// Voice settings
    pub voice: VoiceSettings,
    /// Plugin settings
    pub plugins: PluginSettings,
    /// MERIT metacognition layer settings.
    #[serde(default)]
    pub metacog: MetacogSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySettings {
    /// Enable memory persistence
    pub enabled: bool,
    /// Path used by the native memory engine. If it ends in `.json` the
    /// engine writes a sibling `.db` file alongside it.
    pub path: PathBuf,
    /// Auto-consolidation interval (hours)
    pub consolidation_interval: u64,
    /// Maximum memories to keep
    pub max_memories: usize,
    /// Bank id used as the tenant in the native memory engine.
    #[serde(default = "default_hindsight_bank", alias = "hindsight_bank")]
    pub bank_id: String,
    /// Reserved for future remote-engine support. Unused by the current
    /// native implementation.
    #[serde(default, alias = "hindsight_url")]
    pub remote_url: Option<String>,
    /// Reserved for future remote-engine support. Unused by the current
    /// native implementation.
    #[serde(default, alias = "hindsight_token")]
    pub remote_token: Option<String>,
}

fn default_hindsight_bank() -> String {
    "agentrust".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceSettings {
    /// Enable voice input
    pub enabled: bool,
    /// Push-to-talk mode
    pub push_to_talk: bool,
    /// Silence detection threshold
    pub silence_threshold: f32,
    /// Sample rate
    pub sample_rate: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginSettings {
    /// Enable plugin system
    pub enabled: bool,
    /// Plugin directory
    pub plugin_dir: PathBuf,
    /// Auto-update plugins
    pub auto_update: bool,
}

/// MERIT (metacognition) layer configuration.
///
/// Field defaults match the AgentCpp port; the empty `Default` impl gives
/// you a working engine without any explicit settings file entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetacogSettings {
    /// Master switch. When false the engine is still constructed but the
    /// agent loop won't call into it.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Layer 3 — persist `SelfProposition`s to the memory engine.
    #[serde(default = "default_true")]
    pub enable_layer3: bool,
    /// Layer 4 — periodic schema revision.
    #[serde(default = "default_true")]
    pub enable_layer4: bool,
    /// How many top-k self-knowledge propositions to splice into the
    /// system prompt.
    #[serde(default = "default_prompt_top_k")]
    pub prompt_top_k: usize,
    /// CoT pathology quality floor in `[0, 1]`. Below this the engine
    /// emits a process intervention.
    #[serde(default = "default_low_quality_thresh")]
    pub low_quality_thresh: f64,
    /// Trigger Layer-4 schema revision every N completed turns.
    #[serde(default = "default_review_every")]
    pub review_every_n_episodes: usize,
}

fn default_true() -> bool { true }
fn default_prompt_top_k() -> usize { 3 }
fn default_low_quality_thresh() -> f64 { 0.4 }
fn default_review_every() -> usize { 5 }

impl Default for MetacogSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            enable_layer3: true,
            enable_layer4: true,
            prompt_top_k: 3,
            low_quality_thresh: 0.4,
            review_every_n_episodes: 5,
        }
    }
}

impl MetacogSettings {
    /// Build an `EngineConfig` from these settings.
    pub fn to_engine_config(&self) -> crate::metacognition::EngineConfig {
        let mut cfg = crate::metacognition::EngineConfig::default();
        cfg.enable_layer3 = self.enable_layer3;
        cfg.enable_layer4 = self.enable_layer4;
        cfg.prompt_top_k = self.prompt_top_k;
        cfg.layer2.low_quality_thresh = self.low_quality_thresh;
        cfg.layer4.review_every_n_episodes = self.review_every_n_episodes;
        cfg
    }
}

impl Default for Settings {
    fn default() -> Self {
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        let config_dir = home.join(".agentrust");

        Self {
            api: ApiConfig::default(),
            mcp_servers: Vec::new(),
            model: "sonnet".to_string(),
            verbose: false,
            working_dir: PathBuf::from("."),
            memory: MemorySettings {
                enabled: true,
                path: config_dir.join("memory.db"),
                consolidation_interval: 24,
                max_memories: 1000,
                bank_id: default_hindsight_bank(),
                remote_url: None,
                remote_token: None,
            },
            voice: VoiceSettings {
                enabled: false,
                push_to_talk: false,
                silence_threshold: 0.01,
                sample_rate: 16000,
            },
            plugins: PluginSettings {
                enabled: true,
                plugin_dir: config_dir.join("plugins"),
                auto_update: true,
            },
            metacog: MetacogSettings::default(),
        }
    }
}

impl Settings {
    /// Load settings from file
    pub fn load() -> anyhow::Result<Self> {
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        let config_path = home.join(".agentrust").join("settings.json");

        if config_path.exists() {
            let content = std::fs::read_to_string(&config_path)?;
            let settings: Settings = serde_json::from_str(&content)?;
            Ok(settings)
        } else {
            let settings = Settings::default();
            settings.save()?;
            Ok(settings)
        }
    }

    /// Save settings to file
    pub fn save(&self) -> anyhow::Result<()> {
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        let config_dir = home.join(".agentrust");
        std::fs::create_dir_all(&config_dir)?;
        
        let config_path = config_dir.join("settings.json");
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(&config_path, content)?;
        
        Ok(())
    }

    /// Set a configuration value
    pub fn set(key: &str, value: &str) -> anyhow::Result<()> {
        let mut settings = Self::load()?;
        
        match key {
            "model" => settings.model = value.to_string(),
            "verbose" => settings.verbose = value.parse().unwrap_or(false),
            "api_key" => settings.api.api_key = Some(value.to_string()),
            "base_url" => settings.api.base_url = value.to_string(),
            "max_tokens" => settings.api.max_tokens = value.parse().unwrap_or(4096),
            "timeout" => settings.api.timeout = value.parse().unwrap_or(120),
            "streaming" => settings.api.streaming = value.parse().unwrap_or(true),
            "memory.enabled" => settings.memory.enabled = value.parse().unwrap_or(true),
            "memory.bank_id" | "memory.hindsight_bank" => {
                settings.memory.bank_id = value.to_string();
            }
            "memory.remote_url" | "memory.hindsight_url" => {
                settings.memory.remote_url = if value.is_empty() {
                    None
                } else {
                    Some(value.to_string())
                };
            }
            "memory.remote_token" | "memory.hindsight_token" => {
                settings.memory.remote_token = if value.is_empty() {
                    None
                } else {
                    Some(value.to_string())
                };
            }
            "voice.enabled" => settings.voice.enabled = value.parse().unwrap_or(false),
            _ => return Err(anyhow::anyhow!("Unknown setting: {}", key)),
        }
        
        settings.save()?;
        Ok(())
    }

    /// Reset settings to defaults
    pub fn reset() -> anyhow::Result<()> {
        let settings = Settings::default();
        settings.save()?;
        Ok(())
    }
}