//! Task tool — spawn a sub-agent for a self-contained task.
//!
//! Ported from the AgentCpp `TaskTool`. The sub-agent has its own
//! independent conversation, runs to completion, and returns its final
//! text response.
//!
//! This implementation uses the existing `ApiClient` for a single-prompt
//! completion (no nested tool use). Recursion is bounded by an atomic
//! depth counter shared across the process.

use super::{Tool, ToolError, ToolOutput};
use crate::api::{ApiClient, ChatMessage};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Process-wide sub-agent recursion depth. Bumped on entry, decremented on
/// exit. Defaults to 0 (top-level agent).
static SUBAGENT_DEPTH: AtomicUsize = AtomicUsize::new(0);

/// Maximum sub-agent depth. Mirrors AgentCpp's `ToolContext::max_subagent_depth`.
pub const DEFAULT_MAX_SUBAGENT_DEPTH: usize = 2;

/// RAII guard that increments the depth counter on creation and decrements
/// it on drop. Used to ensure the counter is reset even on early returns.
struct DepthGuard;

impl DepthGuard {
    fn new() -> Self {
        SUBAGENT_DEPTH.fetch_add(1, Ordering::SeqCst);
        Self
    }
}

impl Drop for DepthGuard {
    fn drop(&mut self) {
        SUBAGENT_DEPTH.fetch_sub(1, Ordering::SeqCst);
    }
}

pub struct TaskTool {
    client: Option<Arc<ApiClient>>,
    max_depth: usize,
    system_prompt: String,
}

impl TaskTool {
    /// Construct a fully-wired TaskTool with a live API client.
    pub fn new(client: Arc<ApiClient>) -> Self {
        Self {
            client: Some(client),
            max_depth: DEFAULT_MAX_SUBAGENT_DEPTH,
            system_prompt: default_subagent_system_prompt(),
        }
    }

    /// Construct a stub TaskTool. Calling it will return an error.
    /// Useful when the registry is built before the API client is configured.
    pub fn unbound() -> Self {
        Self {
            client: None,
            max_depth: DEFAULT_MAX_SUBAGENT_DEPTH,
            system_prompt: default_subagent_system_prompt(),
        }
    }

    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }
}

fn default_subagent_system_prompt() -> String {
    "You are a sub-agent spawned to handle a self-contained task. \
     Respond with your final answer as a single concise text reply. \
     You do not have access to tools; reason from the prompt alone."
        .to_string()
}

#[async_trait]
impl Tool for TaskTool {
    fn name(&self) -> &str {
        "Task"
    }

    fn description(&self) -> &str {
        "Spawn a sub-agent to handle a self-contained task. The sub-agent has its own independent conversation and returns its final text response. Use this to delegate work that would otherwise pollute the main conversation, such as broad research or parallel investigations. Recursion is bounded."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Short 3-5 word description of the task (used for logging)."
                },
                "prompt": {
                    "type": "string",
                    "description": "The full instruction the sub-agent should receive. Should be self-contained — the sub-agent has no memory of the parent's context."
                }
            },
            "required": ["prompt"]
        })
    }

    async fn execute(&self, input: serde_json::Value) -> Result<ToolOutput, ToolError> {
        let prompt = input["prompt"].as_str().unwrap_or("");
        if prompt.is_empty() {
            return Err(ToolError {
                message: "'prompt' is required".to_string(),
                code: Some("missing_parameter".to_string()),
            });
        }

        let cur_depth = SUBAGENT_DEPTH.load(Ordering::SeqCst);
        if cur_depth >= self.max_depth {
            return Err(ToolError {
                message: format!(
                    "Sub-agent recursion limit reached (depth={}/{}). The current agent must complete this task directly.",
                    cur_depth, self.max_depth
                ),
                code: Some("recursion_limit".to_string()),
            });
        }

        let client = match &self.client {
            Some(c) => c.clone(),
            None => {
                return Err(ToolError {
                    message: "Task tool not wired to an ApiClient (use TaskTool::new(...) instead of TaskTool::unbound())".to_string(),
                    code: Some("unbound".to_string()),
                });
            }
        };

        let _guard = DepthGuard::new();

        let messages = vec![
            ChatMessage::system(self.system_prompt.clone()),
            ChatMessage::user(prompt.to_string()),
        ];

        let resp = client
            .chat(messages, None)
            .await
            .map_err(|e| ToolError {
                message: format!("sub-agent API call failed: {}", e),
                code: Some("api_error".to_string()),
            })?;

        let text = resp
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .unwrap_or_default();

        if text.is_empty() {
            return Err(ToolError {
                message: "Sub-agent returned no text".to_string(),
                code: Some("empty_response".to_string()),
            });
        }

        let mut metadata = HashMap::new();
        if let Some(desc) = input["description"].as_str() {
            metadata.insert(
                "task_description".to_string(),
                serde_json::Value::String(desc.to_string()),
            );
        }
        metadata.insert(
            "subagent_depth".to_string(),
            serde_json::Value::Number((cur_depth + 1).into()),
        );

        Ok(ToolOutput {
            output_type: "text".to_string(),
            content: text,
            metadata,
        })
    }
}
