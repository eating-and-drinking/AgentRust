//! Layer 3 persistence adapter — `SelfModelMemoryAdapter`.
//!
//! Bridges the in-memory `SelfModelStore` to the AgentRust memory engine
//! (`MemoryManager`). Each `SelfProposition` is persisted as a `MemoryUnit`
//! with `fact_type = Observation` and a reserved tag `type:selfprop` plus
//! metadata that serialises confidence / evidence_count / timestamps.
//!
//! Load + persist callbacks are installed onto a `SelfModelStore` via
//! `wire()` so the store can call them at appropriate lifecycle points
//! (per turn end, on cold start).

use std::sync::Arc;

use crate::memory::{FactType, MemoryEntry, MemoryManager, MemoryType, MemoryUnit};

use super::self_model_store::SelfModelStore;
use super::self_proposition::SelfProposition;

/// Reserved tag put on every Layer-3 proposition stored in the memory
/// engine. Used both as a write marker and as the filter at read time.
pub const SELFPROP_TAG: &str = "type:selfprop";

/// Default bank id when no caller-supplied one is provided. Matches the
/// AgentCpp adapter for cross-tool parity.
pub const DEFAULT_BANK_ID: &str = "metacog_self_model";

#[derive(Clone)]
pub struct SelfModelMemoryAdapter {
    manager: Arc<MemoryManager>,
    bank_id: String,
}

impl SelfModelMemoryAdapter {
    pub fn new(manager: Arc<MemoryManager>) -> Self {
        Self::with_bank(manager, DEFAULT_BANK_ID)
    }

    pub fn with_bank(manager: Arc<MemoryManager>, bank_id: impl Into<String>) -> Self {
        Self {
            manager,
            bank_id: bank_id.into(),
        }
    }

    pub fn bank_id(&self) -> &str {
        &self.bank_id
    }

    /// Read all persisted propositions, blocking on the async memory
    /// engine. Returns `Vec::new()` on any error.
    pub fn load_blocking(&self) -> Vec<SelfProposition> {
        let storage = self.manager.storage();
        let bank = self.bank_id.clone();
        // We can run a blocking SQLite call from any context: storage uses
        // an inner Mutex over a synchronous rusqlite connection.
        let units = match storage.list_units(&bank, Some(FactType::Observation), 10_000, 0) {
            Ok(u) => u,
            Err(_) => return Vec::new(),
        };
        units
            .iter()
            .filter(|u| u.tags.iter().any(|t| t == SELFPROP_TAG))
            .filter_map(|u| unit_to_proposition(u).ok())
            .collect()
    }

    /// Persist the current snapshot of propositions, blocking on the
    /// underlying async `add_memory`. Errors are logged but not returned —
    /// Layer 3 persistence is best-effort.
    pub fn persist_blocking(&self, props: &[SelfProposition]) {
        for p in props {
            let entry = proposition_to_entry(p);
            // Best-effort: drop into a temporary runtime when not in one,
            // or block_on the current handle if we are.
            let manager = self.manager.clone();
            let result =
                if let Ok(handle) = tokio::runtime::Handle::try_current() {
                    tokio::task::block_in_place(|| handle.block_on(manager.add_memory(entry)))
                } else {
                    let rt = match tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                    {
                        Ok(r) => r,
                        Err(_) => continue,
                    };
                    rt.block_on(manager.add_memory(entry))
                };
            if let Err(e) = result {
                tracing::debug!("SelfModelMemoryAdapter persist failed: {}", e);
            }
        }
    }

    /// Wire the adapter onto a store: install load/persist closures and
    /// hydrate the store from disk.
    pub fn wire(self: Arc<Self>, store: &mut SelfModelStore) {
        let load_self = self.clone();
        store.set_load_fn(Arc::new(move || load_self.load_blocking()));
        let persist_self = self.clone();
        store.set_persist_fn(Arc::new(move |props: &[SelfProposition]| {
            persist_self.persist_blocking(props)
        }));
        store.load_from_external();
    }
}

fn proposition_to_entry(p: &SelfProposition) -> MemoryEntry {
    // We use MemoryType::Insight so it maps to FactType::Observation.
    let mut entry = MemoryEntry::new(MemoryType::Insight, &p.text)
        .with_importance(p.confidence as f32);
    // Reserved tag + caller tags.
    let mut tags: Vec<String> = vec![SELFPROP_TAG.to_string()];
    for t in &p.tags {
        if !tags.iter().any(|x| x == t) {
            tags.push(t.clone());
        }
    }
    entry = entry.with_tags(tags);
    entry = entry.with_metadata("selfprop_id", serde_json::Value::String(p.id.clone()));
    entry = entry.with_metadata(
        "confidence",
        serde_json::Value::Number(serde_json::Number::from_f64(p.confidence).unwrap_or_else(
            || serde_json::Number::from_f64(0.0).unwrap(),
        )),
    );
    entry = entry.with_metadata("evidence_count", serde_json::Value::from(p.evidence_count));
    entry = entry.with_metadata("created_at_ms", serde_json::Value::from(p.created_at_ms));
    entry = entry.with_metadata("updated_at_ms", serde_json::Value::from(p.updated_at_ms));
    entry
}

fn unit_to_proposition(unit: &MemoryUnit) -> anyhow::Result<SelfProposition> {
    let text = unit.text.clone();
    let tags: Vec<String> = unit
        .tags
        .iter()
        .filter(|t| *t != SELFPROP_TAG && !t.starts_with("type:"))
        .cloned()
        .collect();
    let confidence = unit
        .metadata
        .get("confidence")
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.5);
    let evidence_count = unit
        .metadata
        .get("evidence_count")
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(1);
    let created_at_ms = unit
        .metadata
        .get("created_at_ms")
        .and_then(|s| s.parse::<i64>().ok())
        .unwrap_or_else(|| unit.created_at.timestamp_millis());
    let updated_at_ms = unit
        .metadata
        .get("updated_at_ms")
        .and_then(|s| s.parse::<i64>().ok())
        .unwrap_or(created_at_ms);
    let id = unit
        .metadata
        .get("selfprop_id")
        .cloned()
        .unwrap_or_else(|| super::self_proposition::make_proposition_id(&text, &tags));
    Ok(SelfProposition {
        id,
        text,
        tags,
        confidence,
        evidence_count,
        created_at_ms,
        updated_at_ms,
    })
}
