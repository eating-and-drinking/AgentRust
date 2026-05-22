//! Consolidation pipeline.
//!
//! Hindsight's `engine/consolidation/consolidator.py` runs an LLM call over
//! a batch of unprocessed `world`/`experience` memories plus the recall'd
//! set of existing `observation`s, then applies the returned
//! `creates / updates / deletes` decisions to the store.
//!
//! The Rust port keeps the same pipeline. When no real LLM is configured
//! ([`super::llm::NoopLlm`]) the engine degrades gracefully to a
//! deterministic dedup pass over the source memories — every cluster of
//! highly-similar units yields one observation.

use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::embed::{cosine, SharedEmbedder};
use super::llm::{ConsolidatedObservation, ConsolidationDecision, SharedLlm};
use super::model::{FactType, MemoryUnit, ObservationHistoryEntry};
use super::recall::{RecallEngine, RecallRequest};
use super::storage::Storage;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationConfig {
    pub max_memories: usize,
    pub importance_threshold: f32,
    pub age_threshold_hours: u64,
    pub consolidation_interval_hours: u64,
    pub enable_auto_consolidation: bool,
    /// Cosine threshold for the deterministic fallback clusterer.
    pub fallback_similarity_threshold: f32,
    /// Optional agent mission appended to the LLM consolidation prompt.
    pub mission: Option<String>,
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        Self {
            max_memories: 10000,
            importance_threshold: 0.3,
            age_threshold_hours: 24,
            consolidation_interval_hours: 6,
            enable_auto_consolidation: true,
            fallback_similarity_threshold: 0.7,
            mission: None,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConsolidationResult {
    pub memories_before: usize,
    pub memories_after: usize,
    pub observations_created: usize,
    pub observations_updated: usize,
    pub observations_deleted: usize,
    pub duration_ms: u64,
    pub timestamp: DateTime<Utc>,
    pub used_llm: bool,
}

pub struct ConsolidationEngine {
    storage: Arc<Storage>,
    recall: Arc<RecallEngine>,
    embedder: SharedEmbedder,
    llm: SharedLlm,
    config: ConsolidationConfig,
    last_consolidation: tokio::sync::RwLock<Option<DateTime<Utc>>>,
}

impl ConsolidationEngine {
    pub fn new(
        storage: Arc<Storage>,
        recall: Arc<RecallEngine>,
        embedder: SharedEmbedder,
        llm: SharedLlm,
        config: ConsolidationConfig,
    ) -> Self {
        Self {
            storage,
            recall,
            embedder,
            llm,
            config,
            last_consolidation: tokio::sync::RwLock::new(None),
        }
    }

    pub fn config(&self) -> &ConsolidationConfig {
        &self.config
    }

    pub async fn last_consolidation(&self) -> Option<DateTime<Utc>> {
        *self.last_consolidation.read().await
    }

    pub fn should_consolidate(&self, memory_count: usize) -> bool {
        memory_count >= self.config.max_memories
    }

    /// Run one consolidation pass for `bank_id`.
    pub async fn run(&self, bank_id: &str) -> anyhow::Result<ConsolidationResult> {
        let start = std::time::Instant::now();
        let before = self.storage.count_units(bank_id)?;

        // Sources: world + experience units not yet consolidated.
        let mut sources: Vec<MemoryUnit> = Vec::new();
        sources.extend(
            self.storage
                .list_units(bank_id, Some(FactType::World), self.config.max_memories, 0)?
                .into_iter()
                .filter(|u| !u.metadata.contains_key("consolidated_at")),
        );
        sources.extend(
            self.storage
                .list_units(bank_id, Some(FactType::Experience), self.config.max_memories, 0)?
                .into_iter()
                .filter(|u| !u.metadata.contains_key("consolidated_at")),
        );
        if sources.is_empty() {
            *self.last_consolidation.write().await = Some(Utc::now());
            return Ok(ConsolidationResult {
                memories_before: before,
                memories_after: before,
                timestamp: Utc::now(),
                duration_ms: start.elapsed().as_millis() as u64,
                used_llm: false,
                ..Default::default()
            });
        }

        // Pull related observations as context for the LLM.
        let merged_query = sources
            .iter()
            .map(|u| u.text.clone())
            .collect::<Vec<_>>()
            .join("\n");
        let related = self
            .recall
            .recall(
                bank_id,
                RecallRequest {
                    query: merged_query.clone(),
                    fact_types: vec![FactType::Observation],
                    budget: super::model::Budget::High,
                    ..Default::default()
                },
            )
            .await
            .map(|r| r.results.into_iter().map(|h| h.unit).collect::<Vec<_>>())
            .unwrap_or_default();

        // Decide.
        let decision = self
            .llm
            .consolidate(
                &sources
                    .iter()
                    .map(|u| (u.id.clone(), u.text.clone()))
                    .collect::<Vec<_>>(),
                &related
                    .iter()
                    .map(|u| (u.id.clone(), u.text.clone()))
                    .collect::<Vec<_>>(),
                self.config.mission.as_deref(),
            )
            .await
            .unwrap_or_default();

        let used_llm = !decision.creates.is_empty()
            || !decision.updates.is_empty()
            || !decision.deletes.is_empty()
            || self.llm.provider_name() != "noop";

        let mut effective = decision;
        if !used_llm {
            // No usable LLM output — run the deterministic fallback.
            effective = self.fallback_decision(&sources).await?;
        }

        let mut created = 0usize;
        let mut updated = 0usize;
        let mut deleted = 0usize;

        // Creates.
        for c in &effective.creates {
            if let Err(_e) = self.apply_create(bank_id, c).await {
                continue;
            }
            created += 1;
        }
        // Updates.
        for u in &effective.updates {
            if let Some(id) = &u.id {
                if let Err(_e) = self.apply_update(bank_id, id, u).await {
                    continue;
                }
                updated += 1;
            }
        }
        // Deletes.
        for id in &effective.deletes {
            self.storage.delete_unit(bank_id, id).ok();
            deleted += 1;
        }

        // Mark sources as consolidated.
        for src in &sources {
            let mut src = src.clone();
            src.metadata
                .insert("consolidated_at".to_string(), Utc::now().to_rfc3339());
            self.storage.insert_unit(&src).ok();
        }

        *self.last_consolidation.write().await = Some(Utc::now());

        let after = self.storage.count_units(bank_id)?;
        Ok(ConsolidationResult {
            memories_before: before,
            memories_after: after,
            observations_created: created,
            observations_updated: updated,
            observations_deleted: deleted,
            duration_ms: start.elapsed().as_millis() as u64,
            timestamp: Utc::now(),
            used_llm: self.llm.provider_name() != "noop",
        })
    }

    async fn apply_create(
        &self,
        bank_id: &str,
        c: &ConsolidatedObservation,
    ) -> anyhow::Result<()> {
        let embedding = self
            .embedder
            .encode(&[c.text.clone()])
            .await?
            .pop()
            .unwrap_or_default();
        let mut unit = MemoryUnit::new(bank_id, c.text.clone(), FactType::Observation);
        unit.tags = c.tags.clone();
        unit.source_memory_ids = c.source_memory_ids.clone();
        unit.proof_count = c.source_memory_ids.len().max(1) as u32;
        unit.embedding = embedding;
        unit.history.push(ObservationHistoryEntry {
            at: Utc::now(),
            action: "create".to_string(),
            note: None,
        });
        self.storage.insert_unit(&unit)
    }

    async fn apply_update(
        &self,
        bank_id: &str,
        id: &str,
        u: &ConsolidatedObservation,
    ) -> anyhow::Result<()> {
        let mut existing = match self.storage.get_unit(bank_id, id)? {
            Some(e) => e,
            None => return Err(anyhow::anyhow!("observation {} not found", id)),
        };
        existing.text = u.text.clone();
        // Union source ids.
        for sid in &u.source_memory_ids {
            if !existing.source_memory_ids.iter().any(|s| s == sid) {
                existing.source_memory_ids.push(sid.clone());
            }
        }
        existing.proof_count = existing.source_memory_ids.len() as u32;
        if !u.tags.is_empty() {
            existing.tags = u.tags.clone();
        }
        existing.history.push(ObservationHistoryEntry {
            at: Utc::now(),
            action: "update".to_string(),
            note: None,
        });
        let new_embed = self
            .embedder
            .encode(&[existing.text.clone()])
            .await?
            .pop()
            .unwrap_or_default();
        existing.embedding = new_embed;
        self.storage.insert_unit(&existing)
    }

    /// Deterministic fallback used when no LLM is configured. Groups source
    /// memories by cosine-similarity threshold and creates one observation
    /// per cluster.
    async fn fallback_decision(
        &self,
        sources: &[MemoryUnit],
    ) -> anyhow::Result<ConsolidationDecision> {
        let mut clusters: Vec<Vec<&MemoryUnit>> = Vec::new();
        for unit in sources {
            let mut placed = false;
            for cluster in clusters.iter_mut() {
                if let Some(head) = cluster.first() {
                    if cosine(&head.embedding, &unit.embedding)
                        >= self.config.fallback_similarity_threshold
                    {
                        cluster.push(unit);
                        placed = true;
                        break;
                    }
                }
            }
            if !placed {
                clusters.push(vec![unit]);
            }
        }
        let creates: Vec<ConsolidatedObservation> = clusters
            .into_iter()
            .filter(|c| c.len() >= 2)
            .map(|cluster| {
                let head = cluster[0];
                let mut tags = head.tags.clone();
                let mut sources: Vec<String> = Vec::new();
                let mut texts: Vec<String> = Vec::new();
                for u in &cluster {
                    sources.push(u.id.clone());
                    if texts.len() < 3 {
                        texts.push(u.text.clone());
                    }
                    for t in &u.tags {
                        if !tags.contains(t) {
                            tags.push(t.clone());
                        }
                    }
                }
                let summary = format!(
                    "Pattern across {} memories: {}",
                    cluster.len(),
                    texts.join("; ")
                );
                ConsolidatedObservation {
                    id: None,
                    text: summary,
                    source_memory_ids: sources,
                    tags,
                }
            })
            .collect();

        let mut hash_map_collisions: HashMap<String, ()> = HashMap::new();
        let creates = creates
            .into_iter()
            .filter(|c| hash_map_collisions.insert(c.text.clone(), ()).is_none())
            .collect();
        Ok(ConsolidationDecision {
            creates,
            updates: Vec::new(),
            deletes: Vec::new(),
        })
    }
}
