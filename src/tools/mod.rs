//! Tools Module - File operations, commands, search, etc.

pub mod bundle;
pub mod file_read;
pub mod http_fetch;
pub mod file_edit;
pub mod file_write;
pub mod execute_command;
pub mod search;
pub mod list_files;
pub mod git_operations;
pub mod task_management;
pub mod note_edit;
// Ports from AgentCpp:
pub mod channel;
pub mod computer;
pub mod glob_tool;
pub mod file_memory;
pub mod skill_md_tool;
pub mod sub_agent;

pub use file_read::FileReadTool;
pub use file_edit::FileEditTool;
pub use file_write::FileWriteTool;
pub use execute_command::ExecuteCommandTool;
pub use search::SearchTool;
pub use list_files::ListFilesTool;
pub use git_operations::GitOperationsTool;
pub use task_management::TaskManagementTool;
pub use note_edit::NoteEditTool;
pub use channel::{ChannelListTool, ChannelPublishTool, ChannelReadTool};
pub use computer::ComputerTool;
pub use glob_tool::GlobTool;
pub use file_memory::{FileMemoryStore, MemoryListTool, MemoryReadTool, MemoryWriteTool};
pub use skill_md_tool::{MarkdownSkill, MarkdownSkillRegistry, SkillMdTool};
pub use sub_agent::TaskTool;
pub use http_fetch::{HttpFetchTool, WebSearchTool};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Tool trait for all tools
#[async_trait]
pub trait Tool: Send + Sync {
    /// Tool name
    fn name(&self) -> &str;

    /// Tool description
    fn description(&self) -> &str;

    /// Tool input schema
    fn input_schema(&self) -> serde_json::Value;

    /// Execute the tool
    async fn execute(&self, input: serde_json::Value) -> Result<ToolOutput, ToolError>;

    /// Convert to OpenAI-compatible function definition
    fn tool_definition(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "function",
            "function": {
                "name": self.name(),
                "description": self.description(),
                "parameters": self.input_schema()
            }
        })
    }
}

/// Tool output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolOutput {
    /// Output type
    pub output_type: String,
    /// Output content
    pub content: String,
    /// Metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Tool error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolError {
    /// Error message
    pub message: String,
    /// Error code
    pub code: Option<String>,
}

/// Tool registry
pub struct ToolRegistry {
    /// Registered tools
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    /// Create a new tool registry
    pub fn new() -> Self {
        let mut registry = Self {
            tools: HashMap::new(),
        };
        
        // Register built-in tools
        registry.register(Box::new(file_read::FileReadTool::new()));
        registry.register(Box::new(file_edit::FileEditTool::new()));
        registry.register(Box::new(file_write::FileWriteTool::new()));
        registry.register(Box::new(execute_command::ExecuteCommandTool::new()));
        registry.register(Box::new(search::SearchTool::new()));
        registry.register(Box::new(list_files::ListFilesTool::new()));
        registry.register(Box::new(git_operations::GitOperationsTool::new()));
        registry.register(Box::new(task_management::TaskManagementTool::new()));
        registry.register(Box::new(note_edit::NoteEditTool::new()));

        // ── Tools ported from AgentCpp ─────────────────────────────
        registry.register(Box::new(glob_tool::GlobTool::new()));
        registry.register(Box::new(channel::ChannelPublishTool::new()));
        registry.register(Box::new(channel::ChannelReadTool::new()));
        registry.register(Box::new(channel::ChannelListTool::new()));
        registry.register(Box::new(computer::ComputerTool::new()));
        // TaskTool is registered unbound — call with_api_client() later to wire it.
        registry.register(Box::new(sub_agent::TaskTool::unbound()));

        // Native web tools — give the `web` bundle real teeth.
        registry.register(Box::new(http_fetch::HttpFetchTool::new()));
        registry.register(Box::new(http_fetch::WebSearchTool::new()));

        // Memory tools share a single FileMemoryStore so list/read/write see
        // each other's writes. Wire up here using the default root.
        let mem_store = Arc::new(file_memory::FileMemoryStore::default());
        registry.register(Box::new(file_memory::MemoryListTool::new(mem_store.clone())));
        registry.register(Box::new(file_memory::MemoryReadTool::new(mem_store.clone())));
        registry.register(Box::new(file_memory::MemoryWriteTool::new(mem_store)));

        // SkillMdTool starts with an empty registry. Callers that want
        // markdown-skill support should load roots and re-register the tool.
        let skill_registry = Arc::new(skill_md_tool::MarkdownSkillRegistry::new());
        registry.register(Box::new(skill_md_tool::SkillMdTool::new(skill_registry)));

        registry
    }

    /// Wire a live `ApiClient` into the `Task` sub-agent tool. The default
    /// registry registers an *unbound* `TaskTool` that returns an error if
    /// called; applications should call this after constructing the registry
    /// (and before exposing it to the model) to enable sub-agent spawning.
    pub fn with_api_client(&mut self, client: Arc<crate::api::ApiClient>) {
        self.register(Box::new(sub_agent::TaskTool::new(client)));
    }

    /// Replace the markdown-skill registry attached to the `Skill` tool.
    /// Use this after loading SKILL.md files from disk.
    pub fn with_markdown_skills(&mut self, skills: Arc<skill_md_tool::MarkdownSkillRegistry>) {
        self.register(Box::new(skill_md_tool::SkillMdTool::new(skills)));
    }

    /// Replace the file-backed memory store used by `MemoryList/Read/Write`.
    /// Useful for `--read-only` mode or for relocating the memory root.
    pub fn with_memory_store(&mut self, store: Arc<file_memory::FileMemoryStore>, read_only: bool) {
        self.register(Box::new(file_memory::MemoryListTool::new(store.clone())));
        self.register(Box::new(file_memory::MemoryReadTool::new(store.clone())));
        self.register(Box::new(file_memory::MemoryWriteTool::with_read_only(store, read_only)));
    }
    
    /// Register a tool
    pub fn register(&mut self, tool: Box<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    /// Prune the registry down to the tools exposed by the given
    /// capability bundles. Tools not in any of the requested bundles
    /// are dropped. Pass an empty slice to keep everything.
    ///
    /// Bundle slugs that don't exist are ignored. Tool names referenced
    /// by a bundle but not actually registered are also ignored.
    pub fn restrict_to_bundles(&mut self, bundles: &[impl AsRef<str>]) {
        if bundles.is_empty() {
            return;
        }
        let allowed = bundle::resolve(bundles);
        if allowed.is_empty() {
            return;
        }
        self.tools.retain(|name, _| allowed.contains(name));
    }

    /// Return the list of bundle slugs whose every tool is currently
    /// registered. Useful for introspection / status output.
    pub fn active_bundles(&self) -> Vec<&'static str> {
        bundle::all()
            .iter()
            .filter(|b| b.tools.iter().all(|t| self.tools.contains_key(*t)))
            .map(|b| b.name)
            .collect()
    }
    
    /// Get a tool by name
    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(|b| b.as_ref())
    }
    
    /// List all tools
    pub fn list(&self) -> Vec<&dyn Tool> {
        self.tools.values().map(|b| b.as_ref()).collect()
    }
    
    /// Execute a tool
    pub async fn execute(&self, name: &str, input: serde_json::Value) -> Result<ToolOutput, ToolError> {
        match self.tools.get(name) {
            Some(tool) => tool.execute(input).await,
            None => Err(ToolError {
                message: format!("Tool not found: {}", name),
                code: Some("tool_not_found".to_string()),
            }),
        }
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}