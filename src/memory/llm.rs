//! LLM abstraction for fact extraction and consolidation.
//!
//! Hindsight's `engine/llm_interface.py:LLMInterface` defines `call`,
//! `call_with_tools`, retry/backoff, and a sanitiser. We port the same
//! shape onto an async trait. Two shipped implementations:
//!
//! * [`NoopLlm`] — the offline fallback. Fact extraction degrades to
//!   sentence-splitting; consolidation produces no creates/updates. This
//!   keeps the engine functional when no LLM is configured.
//! * [`AgentRustLlm`] — wraps AgentRust's existing `ApiClient` so the
//!   memory engine reuses whatever Anthropic / OpenAI-compatible provider
//!   the user has already configured.

use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// One extracted fact returned by the LLM extractor. Mirrors Hindsight's
/// `ExtractedFact` from `engine/retain/fact_extraction.py`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedFact {
    pub text: String,
    /// `world` | `experience`
    pub fact_type: String,
    #[serde(default)]
    pub entities: Vec<ExtractedEntity>,
    #[serde(default)]
    pub when: Option<String>,
    #[serde(default)]
    pub where_: Option<String>,
    #[serde(default)]
    pub who: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEntity {
    pub text: String,
    #[serde(default)]
    pub entity_type: Option<String>,
}

/// A consolidator decision returned by the LLM. Mirrors Hindsight's
/// `_ConsolidationBatchResponse` from `engine/consolidation/consolidator.py`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConsolidationDecision {
    #[serde(default)]
    pub creates: Vec<ConsolidatedObservation>,
    #[serde(default)]
    pub updates: Vec<ConsolidatedObservation>,
    /// Observation ids to delete outright.
    #[serde(default)]
    pub deletes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidatedObservation {
    /// If `Some`, this is an update to an existing observation row.
    #[serde(default)]
    pub id: Option<String>,
    pub text: String,
    /// Source MemoryUnit ids that support this observation.
    #[serde(default)]
    pub source_memory_ids: Vec<String>,
    #[serde(default)]
    pub tags: Vec<String>,
}

#[async_trait]
pub trait MemoryLlm: Send + Sync {
    fn provider_name(&self) -> &'static str;

    /// Extract zero or more facts from a chunk of input text. Hindsight's
    /// real extractor is a single LLM call that takes the chunk and a small
    /// schema description; we keep that shape so providers can implement it
    /// however they like.
    async fn extract_facts(&self, text: &str) -> anyhow::Result<Vec<ExtractedFact>>;

    /// Decide how to consolidate a batch of source memories against the
    /// observations already in the bank.
    async fn consolidate(
        &self,
        sources: &[(String, String)], // (id, text)
        existing_observations: &[(String, String)],
        mission: Option<&str>,
    ) -> anyhow::Result<ConsolidationDecision>;
}

/// No-op LLM. Sentence-splits for extraction, returns an empty decision for
/// consolidation. Used when the user has not configured an LLM provider.
#[derive(Debug, Clone, Default)]
pub struct NoopLlm;

#[async_trait]
impl MemoryLlm for NoopLlm {
    fn provider_name(&self) -> &'static str {
        "noop"
    }

    async fn extract_facts(&self, text: &str) -> anyhow::Result<Vec<ExtractedFact>> {
        // Cheap fall-back: split on sentence terminators, treat every
        // non-trivial sentence as a `world` fact. This keeps the retain
        // pipeline functional without an LLM.
        let mut facts = Vec::new();
        for raw in text
            .split(|c: char| c == '.' || c == '!' || c == '?' || c == '\n')
            .map(str::trim)
            .filter(|s| s.len() >= 4)
        {
            facts.push(ExtractedFact {
                text: raw.to_string(),
                fact_type: "world".to_string(),
                entities: Vec::new(),
                when: None,
                where_: None,
                who: None,
            });
        }
        Ok(facts)
    }

    async fn consolidate(
        &self,
        _sources: &[(String, String)],
        _existing_observations: &[(String, String)],
        _mission: Option<&str>,
    ) -> anyhow::Result<ConsolidationDecision> {
        Ok(ConsolidationDecision::default())
    }
}

/// LLM-backed implementation that reuses AgentRust's existing
/// [`crate::api::ApiClient`]. The actual chat completion is performed via
/// `ApiClient::chat`; we ask the model to return strict JSON.
pub struct AgentRustLlm {
    client: Arc<crate::api::ApiClient>,
}

impl AgentRustLlm {
    pub fn new(client: Arc<crate::api::ApiClient>) -> Self {
        Self { client }
    }

    async fn json_call<T: for<'de> Deserialize<'de>>(
        &self,
        system: &str,
        user: &str,
    ) -> anyhow::Result<T> {
        let messages = vec![
            crate::api::ChatMessage {
                role: "system".to_string(),
                content: Some(system.to_string()),
                tool_calls: None,
                tool_call_id: None,
                images: Vec::new(),
            },
            crate::api::ChatMessage {
                role: "user".to_string(),
                content: Some(user.to_string()),
                tool_calls: None,
                tool_call_id: None,
                images: Vec::new(),
            },
        ];
        let resp = self.client.chat(messages, None).await?;
        let content = extract_text(&resp).unwrap_or_default();
        // Pull the first JSON object/array out of the response; LLMs often
        // wrap output in code fences.
        let trimmed = strip_code_fence(&content);
        Ok(serde_json::from_str(&trimmed)?)
    }
}

fn strip_code_fence(s: &str) -> String {
    let s = s.trim();
    if let Some(rest) = s.strip_prefix("```json") {
        return rest.trim_end_matches("```").trim().to_string();
    }
    if let Some(rest) = s.strip_prefix("```") {
        return rest.trim_end_matches("```").trim().to_string();
    }
    s.to_string()
}

fn extract_text(resp: &crate::api::ChatResponse) -> Option<String> {
    resp.choices
        .first()
        .and_then(|c| c.message.content.clone())
}

const EXTRACT_SYSTEM: &str = "You extract atomic facts from input text for an agent memory system.\n\
Return ONLY JSON in the form: {\"facts\": [{\"text\": str, \"fact_type\": \"world\" | \"experience\", \"entities\": [{\"text\": str, \"entity_type\": str?}], \"when\": str?, \"where_\": str?, \"who\": str?}]}.\n\
Rules:\n\
- One fact per object. Self-contained, single-clause sentences.\n\
- \"world\" = durable third-person facts. \"experience\" = first-person actions / interactions.\n\
- Skip vague or empty input. If the input has no facts, return {\"facts\": []}.\n\
- Do NOT include explanations, markdown, or any prose outside the JSON.";

const CONSOLIDATE_SYSTEM: &str = "You consolidate a batch of source memories into long-lived observations for an agent memory system.\n\
You will receive: (1) a list of source memories with ids and text, (2) the set of existing observations with ids and text, (3) an optional mission string for the agent.\n\
Return ONLY JSON in the form: {\"creates\": [{\"text\": str, \"source_memory_ids\": [str], \"tags\": [str]}], \"updates\": [{\"id\": str, \"text\": str, \"source_memory_ids\": [str], \"tags\": [str]}], \"deletes\": [str]}.\n\
Rules:\n\
- One observation per facet. Never restate the same idea in two observations.\n\
- An update merges new evidence into an existing observation by id; preserve its meaning.\n\
- A delete removes an existing observation that has been refuted or made redundant.\n\
- Cite source_memory_ids drawn from the source list.\n\
- Never invent ids; never include source ids that were not supplied.";

#[derive(Debug, Deserialize)]
struct ExtractResponse {
    #[serde(default)]
    facts: Vec<ExtractedFact>,
}

#[async_trait]
impl MemoryLlm for AgentRustLlm {
    fn provider_name(&self) -> &'static str {
        "agentrust"
    }

    async fn extract_facts(&self, text: &str) -> anyhow::Result<Vec<ExtractedFact>> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }
        let parsed: ExtractResponse = self.json_call(EXTRACT_SYSTEM, text).await?;
        Ok(parsed.facts)
    }

    async fn consolidate(
        &self,
        sources: &[(String, String)],
        existing_observations: &[(String, String)],
        mission: Option<&str>,
    ) -> anyhow::Result<ConsolidationDecision> {
        if sources.is_empty() {
            return Ok(ConsolidationDecision::default());
        }
        let payload = serde_json::json!({
            "mission": mission,
            "sources": sources.iter().map(|(id, t)| serde_json::json!({"id": id, "text": t})).collect::<Vec<_>>(),
            "existing_observations": existing_observations.iter().map(|(id, t)| serde_json::json!({"id": id, "text": t})).collect::<Vec<_>>(),
        });
        let user = serde_json::to_string(&payload)?;
        let parsed: ConsolidationDecision = self.json_call(CONSOLIDATE_SYSTEM, &user).await?;
        Ok(parsed)
    }
}

pub type SharedLlm = Arc<dyn MemoryLlm>;

pub fn default_llm() -> SharedLlm {
    Arc::new(NoopLlm::default())
}
