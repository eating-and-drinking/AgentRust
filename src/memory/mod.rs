//! Memory Module — native Hindsight-style memory engine.
//!
//! Architecture mirrored from `C:\python\hindsight`:
//!
//! * Three [`FactType`]s on one [`MemoryUnit`] table (`world`,
//!   `experience`, `observation`) — see [`model`].
//! * `Bank → Document → MemoryUnit → (Entity, MemoryLink)` graph stored in
//!   SQLite — see [`storage`].
//! * Three-phase [`retain`] pipeline (extract+embed, transactional write,
//!   link build).
//! * Multi-arm [`recall`] (semantic + BM25 + 1-hop graph) fused with
//!   Reciprocal Rank Fusion.
//! * LLM-driven [`consolidate`] with a deterministic fallback when no LLM
//!   is configured.
//!
//! The public types ([`MemoryManager`], [`MemoryEntry`], [`MemoryType`],
//! [`MemoryStatus`]) keep the same signatures the previous implementation
//! exposed, so callers outside `src/memory/` need no changes.

pub mod bm25;
pub mod consolidate;
pub mod context;
pub mod embed;
pub mod fusion;
pub mod graph_retrieval;
pub mod history;
pub mod llm;
pub mod model;
pub mod recall;
pub mod reranker;
pub mod retain;
pub mod session;
pub mod storage;

// Note: `client.rs` from earlier iterations is intentionally not declared
// as a submodule here — it has been replaced by the native engine below.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::{DateTime, Utc};

pub use consolidate::{ConsolidationConfig, ConsolidationEngine, ConsolidationResult};
pub use context::{ContextEntry, ContextManager, ContextWindow};
pub use embed::{default_embedder, Embedder, HashEmbedder, HttpEmbedder, SharedEmbedder};
pub use history::{HistoryEntry, HistoryFilter, HistoryManager, HistoryStats, HistoryType};
pub use llm::{
    default_llm, AgentRustLlm, ConsolidatedObservation, ConsolidationDecision, ExtractedEntity,
    ExtractedFact, MemoryLlm, NoopLlm, SharedLlm,
};
pub use model::{
    Bank, Budget, Document, Entity, FactType, LinkType, MemoryEntry, MemoryLink, MemoryStatus,
    MemoryType, MemoryUnit, ObservationHistoryEntry, TagMatch,
};
pub use bm25::{BM25Config, BM25Hit, BM25Index};
pub use fusion::{normalize_on_delta, reciprocal_rank_fusion, DEFAULT_RRF_K};
pub use graph_retrieval::{expand as graph_expand_with, GraphConfig, GraphHit};
pub use recall::{RecallEngine, RecallHit, RecallRequest, RecallResponse};
pub use reranker::{HeuristicReranker, RerankerConfig};
pub use retain::{RetainEngine, RetainItem, RetainResult};
pub use session::{Session, SessionInfo, SessionManager, SessionMessage, SessionStatus};
pub use storage::{Storage, StorageBackend};

/// Default bank id for single-tenant AgentRust runs.
pub const DEFAULT_BANK_ID: &str = "agentrust";

/// Configuration knob for [`MemoryManager::with_config`].
#[derive(Debug, Clone)]
pub struct MemoryConfig {
    pub bank_id: String,
    pub db_path: PathBuf,
    pub embedding_dim: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        Self {
            bank_id: DEFAULT_BANK_ID.to_string(),
            db_path: home.join(".agentrust").join("memory.db"),
            embedding_dim: HashEmbedder::DEFAULT_DIM,
        }
    }
}

/// Top-level facade. Owns the SQLite store and the engine pipelines and
/// hands references to sub-managers (sessions, history, context).
pub struct MemoryManager {
    bank_id: String,
    storage: Arc<Storage>,
    embedder: SharedEmbedder,
    llm: SharedLlm,
    retain: Arc<RetainEngine>,
    recall: Arc<RecallEngine>,
    consolidation: Arc<ConsolidationEngine>,
    sessions: Arc<SessionManager>,
    history: Arc<HistoryManager>,
    context: Arc<ContextManager>,
}

impl MemoryManager {
    /// Build a manager with default configuration (single bank, SQLite db
    /// under `~/.agentrust/memory.db`, hash embedder, no LLM).
    pub fn new() -> Self {
        Self::with_config(MemoryConfig::default()).expect("default MemoryManager")
    }

    pub fn with_config(cfg: MemoryConfig) -> anyhow::Result<Self> {
        let storage = Arc::new(Storage::open(&cfg.db_path)?);
        storage.ensure_bank(&cfg.bank_id)?;
        let embedder: SharedEmbedder = Arc::new(HashEmbedder::with_dim(cfg.embedding_dim));
        let llm: SharedLlm = default_llm();
        Ok(Self::assemble(cfg.bank_id, storage, embedder, llm))
    }

    /// Wire up the manager with a caller-provided LLM and embedder.
    pub fn with_components(
        bank_id: impl Into<String>,
        db_path: impl AsRef<Path>,
        embedder: SharedEmbedder,
        llm: SharedLlm,
    ) -> anyhow::Result<Self> {
        let storage = Arc::new(Storage::open(db_path)?);
        let bank_id = bank_id.into();
        storage.ensure_bank(&bank_id)?;
        Ok(Self::assemble(bank_id, storage, embedder, llm))
    }

    /// Construct from the AgentRust [`crate::config::MemorySettings`] block.
    pub fn from_settings(settings: &crate::config::MemorySettings) -> anyhow::Result<Self> {
        // The settings path historically pointed to a JSON file; if it ends
        // in `.json` we route to a sibling `.db` so existing on-disk data
        // is not stomped.
        let db_path = if settings.path.extension().map(|e| e == "json").unwrap_or(false) {
            settings.path.with_extension("db")
        } else {
            settings.path.clone()
        };
        let bank_id = if settings.bank_id.is_empty() {
            DEFAULT_BANK_ID.to_string()
        } else {
            settings.bank_id.clone()
        };
        Self::with_config(MemoryConfig {
            bank_id,
            db_path,
            embedding_dim: HashEmbedder::DEFAULT_DIM,
        })
    }

    fn assemble(
        bank_id: String,
        storage: Arc<Storage>,
        embedder: SharedEmbedder,
        llm: SharedLlm,
    ) -> Self {
        let retain = Arc::new(RetainEngine::new(
            storage.clone(),
            embedder.clone(),
            llm.clone(),
        ));
        // Default recall engine wires in the AgentCpp-style heuristic
        // reranker (0.55 RRF + 0.20 recency + 0.25 temporal proximity) and
        // 2-hop graph spreading. Advanced callers can re-build a recall
        // engine without these via `RecallEngine::new(...)` directly.
        let recall = Arc::new(
            RecallEngine::new(storage.clone(), embedder.clone())
                .with_reranker(HeuristicReranker::new())
                .with_graph_hops(2),
        );
        let consolidation = Arc::new(ConsolidationEngine::new(
            storage.clone(),
            recall.clone(),
            embedder.clone(),
            llm.clone(),
            ConsolidationConfig::default(),
        ));
        let sessions = Arc::new(SessionManager::new(
            bank_id.clone(),
            storage.clone(),
            retain.clone(),
            recall.clone(),
        ));
        let history = Arc::new(HistoryManager::new(
            bank_id.clone(),
            storage.clone(),
            retain.clone(),
            recall.clone(),
        ));
        let context = Arc::new(ContextManager::new());

        Self {
            bank_id,
            storage,
            embedder,
            llm,
            retain,
            recall,
            consolidation,
            sessions,
            history,
            context,
        }
    }

    // ---------------------------------------------------------------
    // Public facade — preserved signatures matching the previous module.
    // ---------------------------------------------------------------

    pub fn bank_id(&self) -> &str {
        &self.bank_id
    }

    pub async fn status(&self) -> anyhow::Result<MemoryStatus> {
        let total = self.storage.count_units(&self.bank_id)?;
        let session_units = self
            .storage
            .list_units(&self.bank_id, None, total.max(1), 0)?;
        let mut session_count = 0usize;
        let mut conversation_count = 0usize;
        let mut knowledge_count = 0usize;
        for u in &session_units {
            for t in &u.tags {
                match t.as_str() {
                    "type:session" => session_count += 1,
                    "type:conversation" => conversation_count += 1,
                    "type:knowledge" => knowledge_count += 1,
                    _ => {}
                }
            }
            if u.fact_type == FactType::World {
                knowledge_count += if u.tags.iter().any(|t| t == "type:knowledge") {
                    0
                } else {
                    1
                };
            }
        }
        Ok(MemoryStatus {
            total_memories: total,
            session_count,
            conversation_count,
            knowledge_count,
            last_consolidation: self.consolidation.last_consolidation().await,
            storage_size_bytes: self.storage.db_size_bytes().unwrap_or(0),
        })
    }

    pub async fn add_memory(&self, entry: MemoryEntry) -> anyhow::Result<()> {
        let unit = entry.into_unit(&self.bank_id);
        self.retain
            .retain(
                &self.bank_id,
                vec![RetainItem {
                    content: unit.text.clone(),
                    timestamp: Some(unit.event_date.unwrap_or(unit.mentioned_at)),
                    context: unit.context.clone(),
                    document_id: unit.document_id.clone(),
                    tags: unit.tags.clone(),
                    metadata: unit.metadata.clone(),
                    raw_fact_type: Some(unit.fact_type),
                }],
            )
            .await?;
        Ok(())
    }

    pub async fn get_memory(&self, id: &str) -> Option<MemoryEntry> {
        self.storage
            .get_unit(&self.bank_id, id)
            .ok()
            .flatten()
            .map(|u| MemoryEntry::from_unit(&u))
    }

    pub async fn search_memories(&self, query: &str) -> Vec<MemoryEntry> {
        let resp = self
            .recall
            .recall(
                &self.bank_id,
                RecallRequest {
                    query: query.to_string(),
                    budget: Budget::Mid,
                    max_tokens: Some(8192),
                    ..Default::default()
                },
            )
            .await
            .unwrap_or_default();
        resp.results
            .into_iter()
            .map(|h| MemoryEntry::from_unit(&h.unit))
            .collect()
    }

    pub async fn get_memories_by_type(&self, memory_type: MemoryType) -> Vec<MemoryEntry> {
        let ft = memory_type.fact_type();
        let units = self
            .storage
            .list_units(&self.bank_id, Some(ft), 1000, 0)
            .unwrap_or_default();
        let tag = format!("type:{}", memory_type.as_tag());
        units
            .into_iter()
            .filter(|u| u.tags.iter().any(|t| t == &tag) || u.fact_type == ft)
            .map(|u| MemoryEntry::from_unit(&u))
            .collect()
    }

    pub async fn get_important_memories(&self, threshold: f32) -> Vec<MemoryEntry> {
        let mut out = Vec::new();
        for u in self
            .storage
            .list_units(&self.bank_id, None, 1000, 0)
            .unwrap_or_default()
        {
            let importance = u
                .metadata
                .get("importance")
                .and_then(|s| s.parse::<f32>().ok())
                .unwrap_or(0.5);
            if importance >= threshold {
                out.push(MemoryEntry::from_unit(&u));
            }
        }
        out
    }

    pub async fn clear(&self) -> anyhow::Result<()> {
        self.storage.clear_bank(&self.bank_id)?;
        self.storage.ensure_bank(&self.bank_id)?;
        Ok(())
    }

    pub async fn export(&self, output: &PathBuf) -> anyhow::Result<()> {
        let units = self
            .storage
            .list_units(&self.bank_id, None, 1_000_000, 0)?;
        let entries: Vec<MemoryEntry> = units.iter().map(MemoryEntry::from_unit).collect();
        let content = serde_json::to_string_pretty(&entries)?;
        if let Some(parent) = output.parent() {
            tokio::fs::create_dir_all(parent).await.ok();
        }
        tokio::fs::write(output, content).await?;
        Ok(())
    }

    pub async fn import(&self, input: &PathBuf) -> anyhow::Result<()> {
        let content = tokio::fs::read_to_string(input).await?;
        let imported: Vec<MemoryEntry> = serde_json::from_str(&content)?;
        for entry in imported {
            self.add_memory(entry).await?;
        }
        Ok(())
    }

    pub async fn consolidate(&self) -> anyhow::Result<()> {
        self.consolidation.run(&self.bank_id).await?;
        Ok(())
    }

    /// No persistent load step needed; SQLite is opened in [`new`] /
    /// [`with_config`]. Kept for API compatibility.
    pub async fn load(&self) -> anyhow::Result<()> {
        self.storage.ensure_bank(&self.bank_id)?;
        Ok(())
    }

    /// Writes happen on every [`add_memory`] / retain call. Kept for API
    /// compatibility.
    pub async fn save(&self) -> anyhow::Result<()> {
        Ok(())
    }

    pub fn sessions(&self) -> Arc<SessionManager> {
        self.sessions.clone()
    }

    pub fn history(&self) -> Arc<HistoryManager> {
        self.history.clone()
    }

    pub fn context(&self) -> Arc<ContextManager> {
        self.context.clone()
    }

    pub fn storage(&self) -> Arc<Storage> {
        self.storage.clone()
    }

    pub fn consolidation(&self) -> Arc<ConsolidationEngine> {
        self.consolidation.clone()
    }

    pub fn retain_engine(&self) -> Arc<RetainEngine> {
        self.retain.clone()
    }

    pub fn recall_engine(&self) -> Arc<RecallEngine> {
        self.recall.clone()
    }

    pub fn embedder(&self) -> SharedEmbedder {
        self.embedder.clone()
    }

    pub fn llm(&self) -> SharedLlm {
        self.llm.clone()
    }
}

impl Default for MemoryManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience: build a manager from `MemorySettings` and an existing
/// [`crate::api::ApiClient`] so fact extraction and consolidation are
/// LLM-driven instead of falling back to `NoopLlm`.
pub fn manager_from_settings_with_llm(
    settings: &crate::config::MemorySettings,
    api: Arc<crate::api::ApiClient>,
) -> anyhow::Result<MemoryManager> {
    let db_path = if settings.path.extension().map(|e| e == "json").unwrap_or(false) {
        settings.path.with_extension("db")
    } else {
        settings.path.clone()
    };
    let bank_id = if settings.bank_id.is_empty() {
        DEFAULT_BANK_ID.to_string()
    } else {
        settings.bank_id.clone()
    };
    let embedder: SharedEmbedder = Arc::new(HashEmbedder::default());
    let llm: SharedLlm = Arc::new(AgentRustLlm::new(api));
    MemoryManager::with_components(bank_id, db_path, embedder, llm)
}

#[doc(hidden)]
pub type _MemoryLast = Option<DateTime<Utc>>; // kept for type-completeness
