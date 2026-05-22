//! Tests for Tools Module

use agentrust::tools::{ToolRegistry, ToolOutput};

#[tokio::test]
async fn test_tool_registry_creation() {
    let registry = ToolRegistry::new();
    let tools = registry.list();

    // Should have 9 tools now (6 original + 3 new)
    assert!(tools.len() >= 6);
}

#[tokio::test]
async fn test_file_read_tool() {
    let registry = ToolRegistry::new();
    let tool = registry.get("file_read").expect("file_read tool should exist");

    assert_eq!(tool.name(), "file_read");
    assert!(!tool.description().is_empty());
}

#[tokio::test]
async fn test_git_operations_tool() {
    let registry = ToolRegistry::new();
    let tool = registry.get("git_operations").expect("git_operations tool should exist");

    assert_eq!(tool.name(), "git_operations");
    assert!(tool.description().contains("Git"));
}

#[tokio::test]
async fn test_task_management_tool() {
    let registry = ToolRegistry::new();
    let tool = registry.get("task_management").expect("task_management tool should exist");

    assert_eq!(tool.name(), "task_management");
    assert!(tool.description().contains("task"));
}

#[tokio::test]
async fn test_note_edit_tool() {
    let registry = ToolRegistry::new();
    let tool = registry.get("note_edit").expect("note_edit tool should exist");

    assert_eq!(tool.name(), "note_edit");
    assert!(tool.description().contains("note"));
}

#[tokio::test]
async fn test_task_create_and_list() {
    use serde_json::json;

    let registry = ToolRegistry::new();

    // Create a task
    let create_result = registry.execute("task_management", json!({
        "operation": "create",
        "subject": "Test Task",
        "description": "This is a test task"
    })).await;

    assert!(create_result.is_ok());

    // List tasks
    let list_result = registry.execute("task_management", json!({
        "operation": "list"
    })).await;

    assert!(list_result.is_ok());
}

#[tokio::test]
async fn test_note_create_and_search() {
    use serde_json::json;

    let registry = ToolRegistry::new();

    // Create a note
    let create_result = registry.execute("note_edit", json!({
        "operation": "create",
        "title": "Test Note",
        "content": "This is a test note content",
        "tags": ["test", "example"]
    })).await;

    assert!(create_result.is_ok());

    // Search notes
    let search_result = registry.execute("note_edit", json!({
        "operation": "search",
        "search_query": "test"
    })).await;

    assert!(search_result.is_ok());
}

// ─────────────────────────────────────────────────────────────────────
// Tests for tools ported from AgentCpp
// ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_glob_tool_registered() {
    let registry = ToolRegistry::new();
    let tool = registry.get("Glob").expect("Glob tool should exist");
    assert_eq!(tool.name(), "Glob");
    assert!(tool.description().to_lowercase().contains("glob"));
}

#[tokio::test]
async fn test_channel_publish_and_read() {
    use serde_json::json;
    let registry = ToolRegistry::new();

    let chan = format!("test-chan-{}", uuid::Uuid::new_v4());

    let publish = registry
        .execute(
            "ChannelPublish",
            json!({ "channel": chan, "text": "hello" }),
        )
        .await
        .expect("publish should succeed");
    assert!(publish.content.contains("Published"));

    let read = registry
        .execute("ChannelRead", json!({ "channel": chan, "since_id": 0 }))
        .await
        .expect("read should succeed");
    assert!(read.content.contains("hello"));

    let list = registry
        .execute("ChannelList", json!({}))
        .await
        .expect("list should succeed");
    assert!(list.content.contains(&chan));
}

#[tokio::test]
async fn test_memory_round_trip() {
    use serde_json::json;
    use std::path::PathBuf;
    use std::sync::Arc;

    // Use an isolated tempdir so the test doesn't touch the user's real
    // memory root.
    let dir = tempfile::tempdir().expect("tempdir");
    let store = Arc::new(agentrust::tools::FileMemoryStore::new(
        Some(PathBuf::from(dir.path())),
        true,
    ));

    let mut registry = ToolRegistry::new();
    registry.with_memory_store(store, false);

    let write = registry
        .execute(
            "MemoryWrite",
            json!({ "name": "notes/example.md", "content": "first line\nsecond line" }),
        )
        .await
        .expect("write should succeed");
    assert!(write.content.contains("Wrote"));

    let read = registry
        .execute("MemoryRead", json!({ "name": "notes/example.md" }))
        .await
        .expect("read should succeed");
    assert!(read.content.contains("first line"));

    let list = registry
        .execute("MemoryList", json!({}))
        .await
        .expect("list should succeed");
    assert!(list.content.contains("notes/example.md"));
}

#[tokio::test]
async fn test_memory_rejects_path_traversal() {
    use serde_json::json;
    let registry = ToolRegistry::new();

    let result = registry
        .execute(
            "MemoryWrite",
            json!({ "name": "../escape.txt", "content": "x" }),
        )
        .await;
    assert!(result.is_err(), "path traversal must be rejected");
}

#[tokio::test]
async fn test_task_tool_unbound_returns_error() {
    use serde_json::json;
    let registry = ToolRegistry::new();
    // Default registry has TaskTool unbound (no ApiClient wired).
    let result = registry
        .execute(
            "Task",
            json!({ "description": "test", "prompt": "say hi" }),
        )
        .await;
    assert!(result.is_err(), "unbound TaskTool must error");
}

#[tokio::test]
async fn test_skill_tool_registered_with_empty_registry() {
    use serde_json::json;
    let registry = ToolRegistry::new();
    let tool = registry.get("Skill").expect("Skill tool should exist");
    assert_eq!(tool.name(), "Skill");

    // Calling with an unknown skill should error.
    let result = registry
        .execute("Skill", json!({ "name": "nonexistent" }))
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_computer_tool_registered() {
    let registry = ToolRegistry::new();
    let tool = registry.get("Computer").expect("Computer tool should exist");
    assert_eq!(tool.name(), "Computer");
    assert!(tool.description().to_lowercase().contains("screen"));
}