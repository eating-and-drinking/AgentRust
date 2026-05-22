//! Retain pipeline.
//!
//! Hindsight's `engine/retain/orchestrator.py` runs three phases:
//!
//! 1. **Pre-resolve (outside the transaction)** — fact extraction (LLM
//!    call) and embedding generation.
//! 2. **Transactional write** — INSERT memory units, link them to entities,
//!    write temporal + semantic + entity links.
//! 3. **Post-commit** — build entity-to-entity visualisation links and
//!    enqueue consolidation.
//!
//! The Rust port keeps the same shape, with all three phases on the same
//! tokio task — there's no worker queue in AgentRust today.

use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::embed::{cosine, SharedEmbedder};
use super::llm::SharedLlm;
use super::model::{
    Document, Entity, FactType, LinkType, MemoryLink, MemoryUnit,
};
use super::storage::Storage;

/// One incoming retain item — the unit of work the orchestrator processes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RetainItem {
    pub content: String,
    #[serde(default)]
    pub timestamp: Option<DateTime<Utc>>,
    #[serde(default)]
    pub context: Option<String>,
    #[serde(default)]
    pub document_id: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
    /// If set, the orchestrator skips fact extraction and stores the raw
    /// text as a single MemoryUnit of this type. Used by [`super::history`]
    /// and [`super::session`] which already know their facts.
    #[serde(default)]
    pub raw_fact_type: Option<FactType>,
}

#[derive(Debug, Clone, Default)]
pub struct RetainResult {
    pub stored_unit_ids: Vec<String>,
    pub linked_entities: usize,
    pub created_links: usize,
}

pub struct RetainEngine {
    storage: Arc<Storage>,
    embedder: SharedEmbedder,
    llm: SharedLlm,
    /// Cosine threshold above which two units are considered semantically
    /// linked. Hindsight uses an ANN cutoff; we use a fixed value.
    semantic_link_threshold: f32,
}

impl RetainEngine {
    pub fn new(storage: Arc<Storage>, embedder: SharedEmbedder, llm: SharedLlm) -> Self {
        Self {
            storage,
            embedder,
            llm,
            semantic_link_threshold: 0.55,
        }
    }

    pub async fn retain(
        &self,
        bank_id: &str,
        items: Vec<RetainItem>,
    ) -> anyhow::Result<RetainResult> {
        self.storage.ensure_bank(bank_id)?;

        // ----- Phase 1: extract + embed (outside the transaction) ------
        let mut prepared: Vec<PreparedUnit> = Vec::new();
        for item in items {
            self.prepare_item(bank_id, item, &mut prepared).await?;
        }
        if prepared.is_empty() {
            return Ok(RetainResult::default());
        }

        let texts: Vec<String> = prepared.iter().map(|p| p.unit.text.clone()).collect();
        let embeddings = self.embedder.encode(&texts).await?;
        for (p, emb) in prepared.iter_mut().zip(embeddings.into_iter()) {
            p.unit.embedding = emb;
        }

        // ----- Phase 2: write transactionally --------------------------
        let mut result = RetainResult::default();
        // Ensure documents exist.
        let mut seen_docs: HashMap<String, ()> = HashMap::new();
        for p in &prepared {
            if let Some(doc_id) = &p.unit.document_id {
                if seen_docs.insert(doc_id.clone(), ()).is_none() {
                    let doc = Document::new(doc_id.clone(), bank_id.to_string());
                    self.storage.upsert_document(&doc)?;
                }
            }
        }

        // Persist units.
        let units: Vec<MemoryUnit> = prepared.iter().map(|p| p.unit.clone()).collect();
        self.storage.insert_units(&units)?;
        result.stored_unit_ids = units.iter().map(|u| u.id.clone()).collect();

        // Entity link-up (phase 2 part B).
        for p in &prepared {
            for ent in &p.entities {
                let entity = Entity::new(bank_id.to_string(), ent.canonical.clone(), ent.kind.clone());
                let entity_id = self.storage.upsert_entity(&entity)?;
                self.storage.link_unit_to_entity(&p.unit.id, &entity_id)?;
                result.linked_entities += 1;
            }
        }

        // ----- Phase 3 (still on this task): build memory_links --------
        // Temporal links — chain units within the same document by event_date.
        for window in prepared.windows(2) {
            if let (Some(a), Some(b)) = (window.first(), window.get(1)) {
                if a.unit.document_id == b.unit.document_id && a.unit.document_id.is_some() {
                    self.storage.insert_link(&MemoryLink {
                        from_unit_id: a.unit.id.clone(),
                        to_unit_id: b.unit.id.clone(),
                        link_type: LinkType::Temporal,
                        entity_id: None,
                        weight: 1.0,
                    })?;
                    result.created_links += 1;
                }
            }
        }

        // Semantic links — pairwise within this batch, and against a
        // sample of existing units in the bank.
        let existing_sample =
            self.storage.list_units(bank_id, None, 200, 0).unwrap_or_default();
        for (i, a) in prepared.iter().enumerate() {
            for b in prepared.iter().skip(i + 1) {
                let sim = cosine(&a.unit.embedding, &b.unit.embedding);
                if sim >= self.semantic_link_threshold {
                    self.storage.insert_link(&MemoryLink {
                        from_unit_id: a.unit.id.clone(),
                        to_unit_id: b.unit.id.clone(),
                        link_type: LinkType::Semantic,
                        entity_id: None,
                        weight: sim,
                    })?;
                    result.created_links += 1;
                }
            }
            for other in &existing_sample {
                if other.id == a.unit.id {
                    continue;
                }
                if other.embedding.is_empty() {
                    continue;
                }
                let sim = cosine(&a.unit.embedding, &other.embedding);
                if sim >= self.semantic_link_threshold {
                    self.storage.insert_link(&MemoryLink {
                        from_unit_id: a.unit.id.clone(),
                        to_unit_id: other.id.clone(),
                        link_type: LinkType::Semantic,
                        entity_id: None,
                        weight: sim,
                    })?;
                    result.created_links += 1;
                }
            }
        }

        // Entity-mediated links — within this batch, any two units sharing
        // an entity get an Entity link.
        for (i, a) in prepared.iter().enumerate() {
            for b in prepared.iter().skip(i + 1) {
                let shared: Vec<&PreparedEntity> = a
                    .entities
                    .iter()
                    .filter(|ea| b.entities.iter().any(|eb| eb.canonical == ea.canonical))
                    .collect();
                for ent in shared {
                    self.storage.insert_link(&MemoryLink {
                        from_unit_id: a.unit.id.clone(),
                        to_unit_id: b.unit.id.clone(),
                        link_type: LinkType::Entity,
                        entity_id: None,
                        weight: (1.0_f32).min(0.3 + (ent.canonical.len() as f32) * 0.01),
                    })?;
                    result.created_links += 1;
                }
            }
        }

        Ok(result)
    }

    async fn prepare_item(
        &self,
        bank_id: &str,
        item: RetainItem,
        out: &mut Vec<PreparedUnit>,
    ) -> anyhow::Result<()> {
        // Raw item — skip extraction.
        if let Some(ft) = item.raw_fact_type {
            let mut unit = MemoryUnit::new(bank_id, item.content.clone(), ft);
            unit.document_id = item.document_id.clone();
            unit.context = item.context.clone();
            unit.event_date = item.timestamp;
            unit.occurred_start = item.timestamp;
            unit.mentioned_at = item.timestamp.unwrap_or_else(Utc::now);
            unit.tags = item.tags.clone();
            unit.metadata = item.metadata.clone();
            out.push(PreparedUnit {
                unit,
                entities: Vec::new(),
            });
            return Ok(());
        }

        // Extract facts via LLM.
        let facts = match self.llm.extract_facts(&item.content).await {
            Ok(f) if !f.is_empty() => f,
            _ => {
                // No facts extracted — store the whole input as a single
                // experience unit so nothing is silently dropped.
                let mut unit = MemoryUnit::new(bank_id, item.content.clone(), FactType::Experience);
                unit.document_id = item.document_id.clone();
                unit.context = item.context.clone();
                unit.event_date = item.timestamp;
                unit.occurred_start = item.timestamp;
                unit.mentioned_at = item.timestamp.unwrap_or_else(Utc::now);
                unit.tags = item.tags.clone();
                unit.metadata = item.metadata.clone();
                out.push(PreparedUnit {
                    unit,
                    entities: Vec::new(),
                });
                return Ok(());
            }
        };

        for fact in facts {
            let ft = FactType::from_str(&fact.fact_type).unwrap_or(FactType::World);
            let mut unit = MemoryUnit::new(bank_id, fact.text.clone(), ft);
            unit.document_id = item.document_id.clone();
            unit.context = item.context.clone().or(fact.where_.clone());
            unit.event_date = item.timestamp;
            unit.occurred_start = item.timestamp;
            unit.mentioned_at = item.timestamp.unwrap_or_else(Utc::now);
            unit.tags = item.tags.clone();
            unit.metadata = item.metadata.clone();
            if let Some(when) = &fact.when {
                unit.metadata.insert("when".to_string(), when.clone());
            }
            if let Some(who) = &fact.who {
                unit.metadata.insert("who".to_string(), who.clone());
            }
            let entities: Vec<PreparedEntity> = fact
                .entities
                .into_iter()
                .map(|e| PreparedEntity {
                    canonical: e.text.trim().to_string(),
                    kind: e.entity_type,
                })
                .filter(|e| !e.canonical.is_empty())
                .collect();
            out.push(PreparedUnit { unit, entities });
        }
        Ok(())
    }
}

struct PreparedUnit {
    unit: MemoryUnit,
    entities: Vec<PreparedEntity>,
}

struct PreparedEntity {
    canonical: String,
    kind: Option<String>,
}
