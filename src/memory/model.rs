//! Core data model.
//!
//! Mirrors Hindsight's SQLAlchemy models (`hindsight_api/models.py`) onto
//! Rust structs. The central invariants are:
//!
//! * Three `FactType` values on one [`MemoryUnit`] table:
//!   `world` (third-person durable facts), `experience` (first-person
//!   episodic), and `observation` (LLM-consolidated synthesis).
//! * `Bank → Document → MemoryUnit → (Entity, MemoryLink)` graph.
//! * Observations carry `proof_count` and `source_memory_ids` so
//!   consolidation can update them in place.
//!
//! Tenancy is row-level by `bank_id`; AgentRust runs as a single tenant.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Hindsight's three first-class memory categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FactType {
    /// Third-person durable facts about the external world.
    World,
    /// First-person episodic events — actions, interactions, observations
    /// of the agent itself.
    Experience,
    /// LLM-consolidated synthesis over a set of source memories.
    Observation,
}

impl FactType {
    pub fn as_str(&self) -> &'static str {
        match self {
            FactType::World => "world",
            FactType::Experience => "experience",
            FactType::Observation => "observation",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        Some(match s {
            "world" => FactType::World,
            "experience" => FactType::Experience,
            "observation" => FactType::Observation,
            _ => return None,
        })
    }
}

/// Inter-memory edge type. Mirrors Hindsight's `memory_links.link_type`
/// CHECK. We ship the three retrieval-critical kinds and reserve room for
/// the causal trio.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LinkType {
    Temporal,
    Semantic,
    Entity,
}

impl LinkType {
    pub fn as_str(&self) -> &'static str {
        match self {
            LinkType::Temporal => "temporal",
            LinkType::Semantic => "semantic",
            LinkType::Entity => "entity",
        }
    }
    pub fn from_str(s: &str) -> Option<Self> {
        Some(match s {
            "temporal" => LinkType::Temporal,
            "semantic" => LinkType::Semantic,
            "entity" => LinkType::Entity,
            _ => return None,
        })
    }
}

/// Tag scoping mode at recall time. Same semantics as
/// Hindsight's `engine/search/tags.py`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TagMatch {
    /// `tags && q OR untagged` — default permissive intersection.
    #[default]
    Any,
    /// `tags @> q OR untagged` — every requested tag present.
    All,
    /// Same as `Any` but excludes untagged rows.
    AnyStrict,
    /// Same as `All` but excludes untagged rows.
    AllStrict,
}

/// Recall token budget tier. Maps to per-arm fetch limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Budget {
    Low,
    #[default]
    Mid,
    High,
}

impl Budget {
    /// How many candidates each retrieval arm pulls before fusion.
    pub fn per_arm(&self) -> usize {
        match self {
            Budget::Low => 100,
            Budget::Mid => 300,
            Budget::High => 1000,
        }
    }
}

/// A bank is Hindsight's tenant/agent isolation unit. AgentRust uses a
/// single bank by default but the column is kept so multi-agent setups work.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bank {
    pub bank_id: String,
    pub background: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Bank {
    pub fn new(bank_id: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            bank_id: bank_id.into(),
            background: None,
            created_at: now,
            updated_at: now,
        }
    }
}

/// A document groups memory units that came from the same conversation /
/// transcript / source. Composite identity `(id, bank_id)`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub bank_id: String,
    pub original_text: Option<String>,
    pub content_hash: Option<String>,
    pub metadata: HashMap<String, String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Document {
    pub fn new(id: impl Into<String>, bank_id: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: id.into(),
            bank_id: bank_id.into(),
            original_text: None,
            content_hash: None,
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
        }
    }
}

/// The central record. One row per atomic fact extracted from input text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUnit {
    pub id: String,
    pub bank_id: String,
    pub document_id: Option<String>,
    pub text: String,
    pub fact_type: FactType,
    pub context: Option<String>,
    pub event_date: Option<DateTime<Utc>>,
    pub occurred_start: Option<DateTime<Utc>>,
    pub occurred_end: Option<DateTime<Utc>>,
    pub mentioned_at: DateTime<Utc>,
    /// Visibility scoping tags.
    pub tags: Vec<String>,
    /// Arbitrary string metadata carried along with the unit.
    pub metadata: HashMap<String, String>,
    /// Vector embedding of `text`. Length must equal the embedder's
    /// configured dimension (default 64 for the hash embedder).
    pub embedding: Vec<f32>,
    /// Source memory ids for an observation (empty for world / experience).
    pub source_memory_ids: Vec<String>,
    /// Number of supporting source memories for an observation.
    pub proof_count: u32,
    /// Append-only history of revisions on observations.
    pub history: Vec<ObservationHistoryEntry>,
    pub created_at: DateTime<Utc>,
}

impl MemoryUnit {
    pub fn new(bank_id: impl Into<String>, text: impl Into<String>, fact_type: FactType) -> Self {
        let now = Utc::now();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            bank_id: bank_id.into(),
            document_id: None,
            text: text.into(),
            fact_type,
            context: None,
            event_date: None,
            occurred_start: None,
            occurred_end: None,
            mentioned_at: now,
            tags: Vec::new(),
            metadata: HashMap::new(),
            embedding: Vec::new(),
            source_memory_ids: Vec::new(),
            proof_count: if fact_type == FactType::Observation { 1 } else { 0 },
            history: Vec::new(),
            created_at: now,
        }
    }
}

/// One revision entry kept on an observation's `history`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationHistoryEntry {
    pub at: DateTime<Utc>,
    pub action: String, // "create" | "update" | "merge"
    pub note: Option<String>,
}

/// A named entity that appears in one or more memory units.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub id: String,
    pub bank_id: String,
    pub canonical_name: String,
    pub entity_type: Option<String>,
    pub first_seen: DateTime<Utc>,
    pub last_seen: DateTime<Utc>,
    pub mention_count: u32,
}

impl Entity {
    pub fn new(
        bank_id: impl Into<String>,
        canonical_name: impl Into<String>,
        entity_type: Option<String>,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            bank_id: bank_id.into(),
            canonical_name: canonical_name.into(),
            entity_type,
            first_seen: now,
            last_seen: now,
            mention_count: 0,
        }
    }
}

/// Edge between two memory units. Direction matters only for causal-class
/// links; the retrieval-time graph is treated as undirected.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLink {
    pub from_unit_id: String,
    pub to_unit_id: String,
    pub link_type: LinkType,
    /// Optional entity that mediated this edge (used by `Entity` links).
    pub entity_id: Option<String>,
    pub weight: f32,
}

// ---------------------------------------------------------------------------
// Backwards-compatible public types — kept so callers outside `src/memory/`
// (notably `cli/args.rs` and the `pub use memory::MemoryManager` in
// `lib.rs`) continue to compile.

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum MemoryType {
    Session,
    Conversation,
    Knowledge,
    Preference,
    Task,
    Error,
    Insight,
}

impl MemoryType {
    pub fn fact_type(&self) -> FactType {
        match self {
            MemoryType::Knowledge | MemoryType::Preference => FactType::World,
            MemoryType::Insight => FactType::Observation,
            _ => FactType::Experience,
        }
    }

    pub fn as_tag(&self) -> &'static str {
        match self {
            MemoryType::Session => "session",
            MemoryType::Conversation => "conversation",
            MemoryType::Knowledge => "knowledge",
            MemoryType::Preference => "preference",
            MemoryType::Task => "task",
            MemoryType::Error => "error",
            MemoryType::Insight => "insight",
        }
    }

    pub fn from_tag(tag: &str) -> Option<Self> {
        Some(match tag {
            "session" => MemoryType::Session,
            "conversation" => MemoryType::Conversation,
            "knowledge" => MemoryType::Knowledge,
            "preference" => MemoryType::Preference,
            "task" => MemoryType::Task,
            "error" => MemoryType::Error,
            "insight" => MemoryType::Insight,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id: String,
    pub memory_type: MemoryType,
    pub content: String,
    pub timestamp: DateTime<Utc>,
    pub importance: f32,
    pub tags: Vec<String>,
    pub metadata: HashMap<String, serde_json::Value>,
    pub embedding: Option<Vec<f32>>,
}

impl MemoryEntry {
    pub fn new(memory_type: MemoryType, content: &str) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            memory_type,
            content: content.to_string(),
            timestamp: Utc::now(),
            importance: 0.5,
            tags: Vec::new(),
            metadata: HashMap::new(),
            embedding: None,
        }
    }

    pub fn with_importance(mut self, importance: f32) -> Self {
        self.importance = importance.clamp(0.0, 1.0);
        self
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    pub fn with_metadata(mut self, key: &str, value: serde_json::Value) -> Self {
        self.metadata.insert(key.to_string(), value);
        self
    }

    /// Hoist a `MemoryEntry` (older API) into a freshly-built `MemoryUnit`
    /// (new internal model). The unit gets no embedding here; the retain
    /// pipeline fills it in.
    pub fn into_unit(self, bank_id: &str) -> MemoryUnit {
        let mut tags = self.tags;
        let type_tag = format!("type:{}", self.memory_type.as_tag());
        if !tags.iter().any(|t| t == &type_tag) {
            tags.push(type_tag);
        }
        let mut metadata: HashMap<String, String> = HashMap::new();
        metadata.insert("importance".to_string(), self.importance.to_string());
        metadata.insert(
            "memory_type".to_string(),
            self.memory_type.as_tag().to_string(),
        );
        for (k, v) in self.metadata {
            metadata.insert(k, v.to_string());
        }
        MemoryUnit {
            id: self.id,
            bank_id: bank_id.to_string(),
            document_id: None,
            text: self.content,
            fact_type: self.memory_type.fact_type(),
            context: None,
            event_date: Some(self.timestamp),
            occurred_start: Some(self.timestamp),
            occurred_end: None,
            mentioned_at: self.timestamp,
            tags,
            metadata,
            embedding: self.embedding.unwrap_or_default(),
            source_memory_ids: Vec::new(),
            proof_count: 0,
            history: Vec::new(),
            created_at: self.timestamp,
        }
    }

    /// Inverse of `into_unit`: surface a unit as the older flat shape so
    /// existing callers (export/import, `cli/args.rs`) keep working.
    pub fn from_unit(unit: &MemoryUnit) -> Self {
        let memory_type = unit
            .tags
            .iter()
            .find_map(|t| t.strip_prefix("type:").and_then(MemoryType::from_tag))
            .unwrap_or_else(|| match unit.fact_type {
                FactType::World => MemoryType::Knowledge,
                FactType::Experience => MemoryType::Conversation,
                FactType::Observation => MemoryType::Insight,
            });
        let importance = unit
            .metadata
            .get("importance")
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.5);
        let metadata: HashMap<String, serde_json::Value> = unit
            .metadata
            .iter()
            .filter(|(k, _)| k.as_str() != "importance" && k.as_str() != "memory_type")
            .map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone())))
            .collect();
        Self {
            id: unit.id.clone(),
            memory_type,
            content: unit.text.clone(),
            timestamp: unit.event_date.unwrap_or(unit.mentioned_at),
            importance,
            tags: unit
                .tags
                .iter()
                .filter(|t| !t.starts_with("type:"))
                .cloned()
                .collect(),
            metadata,
            embedding: if unit.embedding.is_empty() {
                None
            } else {
                Some(unit.embedding.clone())
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStatus {
    pub total_memories: usize,
    pub session_count: usize,
    pub conversation_count: usize,
    pub knowledge_count: usize,
    pub last_consolidation: Option<DateTime<Utc>>,
    pub storage_size_bytes: u64,
}
