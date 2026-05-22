//! Personas — packaged (system prompt + suggested tool bundles + temperature).
//!
//! A [`Persona`] is the agent's "role" for a given run. Picking a
//! persona is how a single AgentRust binary serves wildly different
//! workloads: research, writing, coding, data analysis, desktop
//! automation. Personas are *suggestions*, not hard wiring — the caller
//! can override the tool bundle list, the model, and any prompt
//! fragment.

use serde::{Deserialize, Serialize};

/// Built-in persona identities. `Custom` is for user-defined personas
/// loaded from disk (future work).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Persona {
    /// Domain-neutral problem solver — the default when nothing else
    /// fits.
    General,
    /// Software engineering: read/edit code, run commands, manage git.
    Coder,
    /// Open-web research, citation, summarisation.
    Researcher,
    /// Long-form writing, editing, style adaptation.
    Writer,
    /// Data exploration, computation, chart-friendly outputs.
    Analyst,
    /// Desktop / browser operation via the `computer` tool family.
    Operator,
    /// User-defined persona — supply your own profile.
    Custom(String),
}

impl Persona {
    /// Canonical slug (used by `--persona` and config files).
    pub fn slug(&self) -> &str {
        match self {
            Persona::General => "general",
            Persona::Coder => "coder",
            Persona::Researcher => "researcher",
            Persona::Writer => "writer",
            Persona::Analyst => "analyst",
            Persona::Operator => "operator",
            Persona::Custom(s) => s.as_str(),
        }
    }

    /// Parse a CLI / config slug. Unknown slugs become `Custom`.
    pub fn from_slug(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "general" | "default" => Persona::General,
            "coder" | "coding" | "code" => Persona::Coder,
            "researcher" | "research" => Persona::Researcher,
            "writer" | "writing" => Persona::Writer,
            "analyst" | "analysis" | "data" => Persona::Analyst,
            "operator" | "computer" | "desktop" => Persona::Operator,
            other => Persona::Custom(other.to_string()),
        }
    }

    /// Resolve a persona to its profile.
    pub fn profile(&self) -> PersonaProfile {
        match self {
            Persona::General => PersonaProfile {
                name: "General".into(),
                system_prompt: indoc(
                    "You are AgentRust — a careful, domain-agnostic agent. \
                     Decompose the user's goal, choose the smallest useful tool \
                     for each step, verify intermediate results before \
                     trusting them, and report progress in plain language. \
                     If the goal is ambiguous, ask one focused clarifying \
                     question rather than guessing.",
                ),
                bundles: vec!["knowledge".into(), "coding".into()],
                temperature: 0.5,
            },
            Persona::Coder => PersonaProfile {
                name: "Coder".into(),
                system_prompt: indoc(
                    "You are a senior software engineer. Read code before \
                     editing it. Prefer surgical edits to rewrites. After \
                     non-trivial changes, run the relevant tests or a \
                     compile check. Cite file paths and line numbers when \
                     explaining changes.",
                ),
                bundles: vec!["coding".into(), "knowledge".into()],
                temperature: 0.3,
            },
            Persona::Researcher => PersonaProfile {
                name: "Researcher".into(),
                system_prompt: indoc(
                    "You are a research assistant. For factual claims cite \
                     a source (URL, file path, or document title). \
                     Distinguish between what your sources say and what \
                     you infer. When sources disagree, say so explicitly. \
                     Prefer primary sources.",
                ),
                bundles: vec!["knowledge".into(), "web".into()],
                temperature: 0.4,
            },
            Persona::Writer => PersonaProfile {
                name: "Writer".into(),
                system_prompt: indoc(
                    "You are a writing partner. Match the requested \
                     register, audience, and length. Prefer concrete \
                     nouns and verbs. Avoid filler. Offer one revision \
                     pass after the first draft is complete.",
                ),
                bundles: vec!["knowledge".into()],
                temperature: 0.7,
            },
            Persona::Analyst => PersonaProfile {
                name: "Analyst".into(),
                system_prompt: indoc(
                    "You are a data analyst. State assumptions before \
                     computing. Show your work in small reproducible \
                     steps. Caveat conclusions with sample size, \
                     selection effects, or missing data when relevant.",
                ),
                bundles: vec!["coding".into(), "knowledge".into()],
                temperature: 0.3,
            },
            Persona::Operator => PersonaProfile {
                name: "Operator".into(),
                system_prompt: indoc(
                    "You are an operator agent driving the user's desktop \
                     / browser. Take a screenshot before acting. Confirm \
                     the action took effect before chaining the next step. \
                     Never type credentials or submit irreversible forms \
                     without explicit user approval.",
                ),
                bundles: vec!["desktop".into(), "web".into()],
                temperature: 0.2,
            },
            Persona::Custom(name) => {
                // Prefer a user-supplied profile on disk; fall back to a
                // generic stub if none is found.
                PersonaProfile::load_from_disk(name).unwrap_or_else(|| PersonaProfile {
                    name: name.clone(),
                    system_prompt: format!(
                        "You are the `{name}` agent. The user will provide \
                         specifics in the goal block."
                    ),
                    bundles: vec!["knowledge".into()],
                    temperature: 0.5,
                })
            }
        }
    }
}

impl Default for Persona {
    fn default() -> Self {
        Persona::General
    }
}

impl std::fmt::Display for Persona {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.slug())
    }
}

/// Resolved persona — what the runner actually uses.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct PersonaProfile {
    /// Human-friendly name.
    pub name: String,
    /// System prompt fragment (gets composed with the goal block).
    pub system_prompt: String,
    /// Suggested tool bundles (see [`crate::tools::bundle`]).
    pub bundles: Vec<String>,
    /// Default sampling temperature.
    pub temperature: f32,
}

impl PersonaProfile {
    /// Try to load a custom persona from `~/.agentrust/personas/<slug>.toml`.
    ///
    /// Format:
    /// ```toml
    /// name = "Marketing Copywriter"
    /// system_prompt = "You write punchy B2B SaaS marketing copy ..."
    /// bundles = ["knowledge", "web"]
    /// temperature = 0.8
    /// ```
    pub fn load_from_disk(slug: &str) -> Option<Self> {
        let home = dirs::home_dir()?;
        let path = home.join(".agentrust").join("personas").join(format!("{}.toml", slug));
        let text = std::fs::read_to_string(&path).ok()?;
        toml::from_str::<PersonaProfile>(&text).ok()
    }
}

/// Trim leading whitespace from every line — keeps the source readable
/// without bloating the prompt.
fn indoc(s: &str) -> String {
    s.lines()
        .map(|l| l.trim_start())
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string()
}
