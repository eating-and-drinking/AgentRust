//! Integration tests for the `agent` module — Goal / Persona / Bundle
//! resolution, multimodal `ChatMessage` serialisation, and registry
//! restriction. None of these tests touch the network: they exercise
//! the pure data layer of the runtime.

use agentrust::agent::{Goal, GoalStatus, Persona};
use agentrust::api::{ChatMessage, ImageRef};
use agentrust::tools::{bundle, ToolRegistry};

// ── Goal ────────────────────────────────────────────────────────────

#[test]
fn goal_new_has_pending_status_and_uuid() {
    let g = Goal::new("write a haiku about rust");
    assert_eq!(g.status, GoalStatus::Pending);
    assert!(!g.id.is_empty());
    assert!(g.success_criteria.is_empty());
    assert!(g.context.is_empty());
    assert!(g.deadline.is_none());
}

#[test]
fn goal_builders_chain() {
    let g = Goal::new("ship feature X")
        .with_criterion("tests pass")
        .with_criterion("docs updated")
        .with_context("affects auth subsystem");

    assert_eq!(g.success_criteria.len(), 2);
    assert_eq!(g.context.len(), 1);
}

#[test]
fn goal_prompt_block_contains_all_sections() {
    let g = Goal::new("ship feature X")
        .with_criterion("tests pass")
        .with_context("see Linear AUTH-42");
    let block = g.to_prompt_block();

    assert!(block.contains("# Goal"));
    assert!(block.contains("ship feature X"));
    assert!(block.contains("## Success criteria"));
    assert!(block.contains("- tests pass"));
    assert!(block.contains("## Context"));
    assert!(block.contains("- see Linear AUTH-42"));
}

// ── Persona ─────────────────────────────────────────────────────────

#[test]
fn persona_from_slug_known() {
    assert_eq!(Persona::from_slug("coder"), Persona::Coder);
    assert_eq!(Persona::from_slug("RESEARCHER"), Persona::Researcher);
    assert_eq!(Persona::from_slug("writing"), Persona::Writer);
    assert_eq!(Persona::from_slug("data"), Persona::Analyst);
    assert_eq!(Persona::from_slug("computer"), Persona::Operator);
    assert_eq!(Persona::from_slug("default"), Persona::General);
}

#[test]
fn persona_from_slug_unknown_becomes_custom() {
    match Persona::from_slug("marketing-copywriter") {
        Persona::Custom(s) => assert_eq!(s, "marketing-copywriter"),
        other => panic!("expected Custom, got {:?}", other),
    }
}

#[test]
fn persona_profile_has_nonempty_prompt_and_bundles() {
    for persona in [
        Persona::General,
        Persona::Coder,
        Persona::Researcher,
        Persona::Writer,
        Persona::Analyst,
        Persona::Operator,
    ] {
        let p = persona.profile();
        assert!(!p.name.is_empty(), "{:?} has empty name", persona);
        assert!(!p.system_prompt.is_empty(), "{:?} has empty prompt", persona);
        assert!(!p.bundles.is_empty(), "{:?} has no bundles", persona);
        assert!(p.temperature >= 0.0 && p.temperature <= 1.5);
    }
}

#[test]
fn persona_default_is_general() {
    assert_eq!(Persona::default(), Persona::General);
}

// ── Bundles ─────────────────────────────────────────────────────────

#[test]
fn bundle_lookup_returns_known_bundles() {
    for slug in ["coding", "knowledge", "desktop", "web", "communication"] {
        assert!(bundle::lookup(slug).is_some(), "missing bundle: {}", slug);
    }
    assert!(bundle::lookup("nonexistent").is_none());
}

#[test]
fn bundle_resolve_unions_tools() {
    let union = bundle::resolve(&["coding", "knowledge"]);
    // Coding-only tools
    assert!(union.contains("git_operations"));
    assert!(union.contains("execute_command"));
    // Knowledge-only tools
    assert!(union.contains("MemoryRead"));
    assert!(union.contains("Skill"));
    // Shared tool (file_read appears in both)
    assert!(union.contains("file_read"));
}

#[test]
fn bundle_resolve_unknown_slug_is_silent() {
    let result = bundle::resolve(&["coding", "made_up_slug"]);
    // Should contain coding's tools, no panic.
    assert!(result.contains("file_edit"));
}

#[test]
fn web_bundle_exposes_native_http_tools() {
    let web = bundle::lookup("web").expect("web bundle");
    assert!(web.tools.contains(&"http_fetch"));
    assert!(web.tools.contains(&"web_search"));
}

// ── ToolRegistry bundle restriction ─────────────────────────────────

#[test]
fn restrict_to_bundles_drops_non_listed_tools() {
    let mut reg = ToolRegistry::new();
    let before = reg.list().len();
    reg.restrict_to_bundles(&["desktop"]);
    let after = reg.list().len();

    // Desktop only exposes the `Computer` tool.
    assert!(after < before, "restriction did not shrink the registry");
    assert!(reg.get("Computer").is_some());
    assert!(reg.get("git_operations").is_none());
}

#[test]
fn restrict_to_bundles_empty_input_is_noop() {
    let mut reg = ToolRegistry::new();
    let before = reg.list().len();
    let no_slugs: Vec<String> = Vec::new();
    reg.restrict_to_bundles(&no_slugs);
    assert_eq!(reg.list().len(), before);
}

// ── ChatMessage multimodal serialisation ────────────────────────────

#[test]
fn chat_message_plain_text_serialises_as_string() {
    let msg = ChatMessage::user("hello world");
    let json = serde_json::to_value(&msg).expect("serialize");
    assert_eq!(json["content"], serde_json::json!("hello world"));
    assert_eq!(json["role"], serde_json::json!("user"));
}

#[test]
fn chat_message_with_image_serialises_as_parts_array() {
    let msg = ChatMessage::user_with_images(
        "what's in this image?",
        vec![ImageRef::from_url("https://example.com/cat.png")],
    );
    let json = serde_json::to_value(&msg).expect("serialize");
    let parts = json["content"].as_array().expect("content should be array");
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0]["type"], serde_json::json!("text"));
    assert_eq!(parts[0]["text"], serde_json::json!("what's in this image?"));
    assert_eq!(parts[1]["type"], serde_json::json!("image_url"));
    assert_eq!(
        parts[1]["image_url"]["url"],
        serde_json::json!("https://example.com/cat.png")
    );
}

#[test]
fn chat_message_with_image_builder_appends() {
    let msg = ChatMessage::user("inspect")
        .with_image(ImageRef::from_url("https://x.test/1.png"))
        .with_image(ImageRef::from_url("https://x.test/2.png"));
    let json = serde_json::to_value(&msg).expect("serialize");
    let parts = json["content"].as_array().unwrap();
    // text + 2 images
    assert_eq!(parts.len(), 3);
}
