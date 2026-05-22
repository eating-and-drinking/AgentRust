//! Capability bundles — group built-in tools by domain so a persona /
//! task can opt into just the surface it needs.
//!
//! AgentRust's [`ToolRegistry`](super::ToolRegistry) registers every
//! built-in tool in `new()`. That's convenient for the REPL, but
//! wasteful for a focused autonomous run — a `Researcher` persona has
//! no business with `git_operations`, and a `Writer` doesn't need
//! `computer`-driven screenshots.
//!
//! A `Bundle` is a named subset of tool names. The registry can be
//! pruned to a chosen list via [`ToolRegistry::restrict_to_bundles`].

use std::collections::HashSet;

/// A named tool subset.
#[derive(Debug, Clone)]
pub struct Bundle {
    pub name: &'static str,
    pub description: &'static str,
    pub tools: &'static [&'static str],
}

/// Software engineering: edit code, run commands, manage git, search.
///
/// Tool-name strings match what each `Tool::name()` returns at runtime
/// — note the mix of snake_case and PascalCase in the existing
/// codebase.
pub const CODING: Bundle = Bundle {
    name: "coding",
    description: "Read/edit code, run commands, manage git, search source.",
    tools: &[
        "file_read",
        "file_write",
        "file_edit",
        "list_files",
        "search",
        "Glob",
        "execute_command",
        "git_operations",
    ],
};

/// Knowledge work: notes, durable memory, agent-to-agent channels,
/// task lists, markdown skills, sub-agent dispatch.
pub const KNOWLEDGE: Bundle = Bundle {
    name: "knowledge",
    description: "Notes, durable memory, channels, task lists, sub-agents, markdown skills.",
    tools: &[
        "file_read",
        "note_edit",
        "task_management",
        "MemoryList",
        "MemoryRead",
        "MemoryWrite",
        "ChannelPublish",
        "ChannelRead",
        "ChannelList",
        "Skill",
        "Task",
    ],
};

/// Desktop / GUI operation via the `Computer` tool family.
pub const DESKTOP: Bundle = Bundle {
    name: "desktop",
    description: "Drive the user's desktop: screenshot, click, type, keys.",
    tools: &["Computer"],
};

/// Open-web research: native HTTP fetch + DuckDuckGo-backed search.
/// Falls back to `file_read` / `file_write` for caching pages locally.
pub const WEB: Bundle = Bundle {
    name: "web",
    description: "Native HTTP fetch + web search; file_read/write for local caching.",
    tools: &["http_fetch", "web_search", "file_read", "file_write"],
};

/// Inter-agent communication: pub/sub channel bus and sub-agent task
/// dispatch.
pub const COMMUNICATION: Bundle = Bundle {
    name: "communication",
    description: "Channel bus + sub-agent dispatch.",
    tools: &["ChannelPublish", "ChannelRead", "ChannelList", "Task"],
};

/// All registered bundles, in iteration order.
pub fn all() -> &'static [Bundle] {
    &[CODING, KNOWLEDGE, DESKTOP, WEB, COMMUNICATION]
}

/// Resolve a list of bundle slugs to the union of tool names they
/// expose. Unknown slugs are silently skipped (the caller may want to
/// warn separately).
pub fn resolve(slugs: &[impl AsRef<str>]) -> HashSet<String> {
    let mut out = HashSet::new();
    for slug in slugs {
        let slug = slug.as_ref();
        if let Some(b) = all().iter().find(|b| b.name == slug) {
            for t in b.tools {
                out.insert((*t).to_string());
            }
        }
    }
    out
}

/// Look up a bundle by slug.
pub fn lookup(slug: &str) -> Option<&'static Bundle> {
    all().iter().find(|b| b.name == slug)
}
