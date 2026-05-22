//! Channel tools — expose the in-process pub/sub bus to the model.
//!
//! Three tools, mirroring the AgentCpp set:
//! - `ChannelPublish(channel, text)` — post a message
//! - `ChannelRead(channel, since_id)` — read messages newer than `since_id`
//! - `ChannelList()` — enumerate known channels
//!
//! All three share the global `ChannelBus` singleton.

use super::{Tool, ToolError, ToolOutput};
use crate::channels::ChannelBus;
use async_trait::async_trait;
use chrono::TimeZone;
use std::collections::HashMap;
use std::fmt::Write as _;

fn local_hms(epoch_ms: i64) -> String {
    chrono::Local
        .timestamp_millis_opt(epoch_ms)
        .single()
        .map(|t| t.format("%H:%M:%S").to_string())
        .unwrap_or_else(|| "??:??:??".to_string())
}

// ──────────────────────────── ChannelPublish ────────────────────────────────

pub struct ChannelPublishTool;

impl Default for ChannelPublishTool {
    fn default() -> Self {
        Self::new()
    }
}

impl ChannelPublishTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for ChannelPublishTool {
    fn name(&self) -> &str {
        "ChannelPublish"
    }

    fn description(&self) -> &str {
        "Publish a message to a named in-process channel. Channels are auto-created on first use and persist for the duration of this CLI run. Useful for coordinating with sub-agents or leaving notes for later turns to read with ChannelRead."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "channel": { "type": "string", "description": "Channel name." },
                "text":    { "type": "string", "description": "Message body." },
                "sender":  { "type": "string", "description": "Optional sender label (defaults to 'main')." }
            },
            "required": ["channel", "text"]
        })
    }

    async fn execute(&self, input: serde_json::Value) -> Result<ToolOutput, ToolError> {
        let channel = input["channel"].as_str().ok_or_else(|| ToolError {
            message: "'channel' is required".to_string(),
            code: Some("missing_parameter".to_string()),
        })?;
        if channel.is_empty() {
            return Err(ToolError {
                message: "'channel' is required".to_string(),
                code: Some("missing_parameter".to_string()),
            });
        }
        let text = input["text"].as_str().unwrap_or("");
        let sender = input["sender"].as_str().unwrap_or("main");

        let id = ChannelBus::instance().publish(channel, sender, text);

        Ok(ToolOutput {
            output_type: "text".to_string(),
            content: format!("Published message #{} to '{}'", id, channel),
            metadata: HashMap::new(),
        })
    }
}

// ──────────────────────────── ChannelRead ───────────────────────────────────

pub struct ChannelReadTool;

impl Default for ChannelReadTool {
    fn default() -> Self {
        Self::new()
    }
}

impl ChannelReadTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for ChannelReadTool {
    fn name(&self) -> &str {
        "ChannelRead"
    }

    fn description(&self) -> &str {
        "Read messages from a named channel. Pass `since_id` to return only messages newer than a given id (use the last id you saw)."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "channel":  { "type": "string",  "description": "Channel name." },
                "since_id": { "type": "integer", "description": "Return messages with id > since_id.", "default": 0 }
            },
            "required": ["channel"]
        })
    }

    async fn execute(&self, input: serde_json::Value) -> Result<ToolOutput, ToolError> {
        let channel = input["channel"].as_str().ok_or_else(|| ToolError {
            message: "'channel' is required".to_string(),
            code: Some("missing_parameter".to_string()),
        })?;
        if channel.is_empty() {
            return Err(ToolError {
                message: "'channel' is required".to_string(),
                code: Some("missing_parameter".to_string()),
            });
        }
        let since: u64 = input["since_id"].as_u64().unwrap_or(0);

        let msgs = ChannelBus::instance().read(channel, since);

        let mut out = String::new();
        let suffix = if msgs.len() == 1 { "" } else { "s" };
        let _ = write!(
            out,
            "Channel '{}' ({} msg{} since #{})\n",
            channel,
            msgs.len(),
            suffix,
            since
        );
        if msgs.is_empty() {
            out.push_str("(none)");
        } else {
            for m in &msgs {
                let _ = write!(
                    out,
                    "\n#{} [{} {}] {}",
                    m.id,
                    local_hms(m.epoch_ms),
                    m.sender,
                    m.text
                );
            }
        }

        Ok(ToolOutput {
            output_type: "text".to_string(),
            content: out,
            metadata: HashMap::new(),
        })
    }
}

// ──────────────────────────── ChannelList ───────────────────────────────────

pub struct ChannelListTool;

impl Default for ChannelListTool {
    fn default() -> Self {
        Self::new()
    }
}

impl ChannelListTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for ChannelListTool {
    fn name(&self) -> &str {
        "ChannelList"
    }

    fn description(&self) -> &str {
        "List every channel that currently has at least one message, with the message count and the latest message id."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {}
        })
    }

    async fn execute(&self, _input: serde_json::Value) -> Result<ToolOutput, ToolError> {
        let infos = ChannelBus::instance().list();
        let mut out = format!("Channels: {}\n", infos.len());
        if infos.is_empty() {
            out.push_str("(none)");
        } else {
            for i in &infos {
                let _ = write!(
                    out,
                    "  {}  msgs={}  latest=#{}\n",
                    i.name, i.message_count, i.latest_id
                );
            }
        }
        Ok(ToolOutput {
            output_type: "text".to_string(),
            content: out,
            metadata: HashMap::new(),
        })
    }
}
