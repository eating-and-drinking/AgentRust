//! Command / query history on top of the native memory engine.
//!
//! Every history entry is a `MemoryUnit` of `fact_type=experience` tagged
//! with `type:history` and `history_type:<command|query|...>`. The history
//! taxonomy is preserved as tags so the existing filter API still works.

use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::model::{FactType, MemoryUnit, TagMatch};
use super::recall::{RecallEngine, RecallRequest};
use super::retain::{RetainEngine, RetainItem};
use super::storage::Storage;

const HISTORY_TAG: &str = "type:history";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub id: String,
    pub entry_type: HistoryType,
    pub content: String,
    pub timestamp: DateTime<Utc>,
    pub session_id: Option<String>,
    pub success: bool,
    pub duration_ms: Option<u64>,
    pub metadata: serde_json::Value,
}

impl HistoryEntry {
    pub fn new(entry_type: HistoryType, content: &str) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            entry_type,
            content: content.to_string(),
            timestamp: Utc::now(),
            session_id: None,
            success: true,
            duration_ms: None,
            metadata: serde_json::Value::Null,
        }
    }

    pub fn with_session(mut self, session_id: &str) -> Self {
        self.session_id = Some(session_id.to_string());
        self
    }

    pub fn with_duration(mut self, duration_ms: u64) -> Self {
        self.duration_ms = Some(duration_ms);
        self
    }

    pub fn with_success(mut self, success: bool) -> Self {
        self.success = success;
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum HistoryType {
    Command,
    Query,
    ToolCall,
    FileOperation,
    Search,
    Agent,
}

fn history_type_tag(t: &HistoryType) -> &'static str {
    match t {
        HistoryType::Command => "command",
        HistoryType::Query => "query",
        HistoryType::ToolCall => "tool_call",
        HistoryType::FileOperation => "file_op",
        HistoryType::Search => "search",
        HistoryType::Agent => "agent",
    }
}

fn history_type_from_tag(s: &str) -> Option<HistoryType> {
    Some(match s {
        "command" => HistoryType::Command,
        "query" => HistoryType::Query,
        "tool_call" => HistoryType::ToolCall,
        "file_op" => HistoryType::FileOperation,
        "search" => HistoryType::Search,
        "agent" => HistoryType::Agent,
        _ => return None,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryFilter {
    pub entry_type: Option<HistoryType>,
    pub session_id: Option<String>,
    pub success_only: bool,
    pub from_time: Option<DateTime<Utc>>,
    pub to_time: Option<DateTime<Utc>>,
    pub limit: usize,
}

impl Default for HistoryFilter {
    fn default() -> Self {
        Self {
            entry_type: None,
            session_id: None,
            success_only: false,
            from_time: None,
            to_time: None,
            limit: 100,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryStats {
    pub total_entries: usize,
    pub commands: usize,
    pub queries: usize,
    pub tool_calls: usize,
    pub successful: usize,
    pub failed: usize,
}

pub struct HistoryManager {
    bank_id: String,
    storage: Arc<Storage>,
    retain: Arc<RetainEngine>,
    recall: Arc<RecallEngine>,
}

impl HistoryManager {
    pub fn new(
        bank_id: String,
        storage: Arc<Storage>,
        retain: Arc<RetainEngine>,
        recall: Arc<RecallEngine>,
    ) -> Self {
        Self {
            bank_id,
            storage,
            retain,
            recall,
        }
    }

    pub async fn add(&self, entry: HistoryEntry) -> anyhow::Result<()> {
        let mut metadata: HashMap<String, String> = HashMap::new();
        metadata.insert("agentrust_id".to_string(), entry.id.clone());
        metadata.insert(
            "entry_type".to_string(),
            history_type_tag(&entry.entry_type).to_string(),
        );
        metadata.insert("success".to_string(), entry.success.to_string());
        if let Some(d) = entry.duration_ms {
            metadata.insert("duration_ms".to_string(), d.to_string());
        }
        if let Some(s) = &entry.session_id {
            metadata.insert("session_id".to_string(), s.clone());
        }
        if !entry.metadata.is_null() {
            metadata.insert("payload".to_string(), entry.metadata.to_string());
        }

        let mut tags = vec![
            HISTORY_TAG.to_string(),
            format!("history_type:{}", history_type_tag(&entry.entry_type)),
        ];
        if let Some(s) = &entry.session_id {
            tags.push(format!("session:{s}"));
        }
        if !entry.success {
            tags.push("failed".to_string());
        }

        self.retain
            .retain(
                &self.bank_id,
                vec![RetainItem {
                    content: entry.content.clone(),
                    timestamp: Some(entry.timestamp),
                    context: entry.session_id.clone(),
                    document_id: entry.session_id.clone(),
                    tags,
                    metadata,
                    raw_fact_type: Some(FactType::Experience),
                }],
            )
            .await?;
        Ok(())
    }

    pub async fn get(&self, id: &str) -> Option<HistoryEntry> {
        let units = self
            .storage
            .list_units(&self.bank_id, Some(FactType::Experience), 1000, 0)
            .unwrap_or_default();
        units
            .iter()
            .find(|u| u.metadata.get("agentrust_id").map(|s| s.as_str()) == Some(id))
            .map(unit_to_history_entry)
    }

    pub async fn list(&self, filter: HistoryFilter) -> Vec<HistoryEntry> {
        let mut tags = vec![HISTORY_TAG.to_string()];
        if let Some(t) = &filter.entry_type {
            tags.push(format!("history_type:{}", history_type_tag(t)));
        }
        if let Some(s) = &filter.session_id {
            tags.push(format!("session:{s}"));
        }
        let resp = self
            .recall
            .recall(
                &self.bank_id,
                RecallRequest {
                    query: String::new(),
                    fact_types: vec![FactType::Experience],
                    tags,
                    tag_match: TagMatch::AllStrict,
                    budget: super::model::Budget::High,
                    max_tokens: None,
                },
            )
            .await
            .unwrap_or_default();

        let mut entries: Vec<HistoryEntry> = resp
            .results
            .into_iter()
            .map(|h| unit_to_history_entry(&h.unit))
            .filter(|e| {
                if filter.success_only && !e.success {
                    return false;
                }
                if let Some(from) = filter.from_time {
                    if e.timestamp < from {
                        return false;
                    }
                }
                if let Some(to) = filter.to_time {
                    if e.timestamp > to {
                        return false;
                    }
                }
                true
            })
            .collect();
        entries.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        entries.truncate(filter.limit);
        entries
    }

    pub async fn search(&self, query: &str) -> Vec<HistoryEntry> {
        let resp = self
            .recall
            .recall(
                &self.bank_id,
                RecallRequest {
                    query: query.to_string(),
                    fact_types: vec![FactType::Experience],
                    tags: vec![HISTORY_TAG.to_string()],
                    tag_match: TagMatch::AllStrict,
                    budget: super::model::Budget::Mid,
                    max_tokens: Some(8192),
                },
            )
            .await
            .unwrap_or_default();
        resp.results
            .into_iter()
            .map(|h| unit_to_history_entry(&h.unit))
            .collect()
    }

    pub async fn get_recent(&self, count: usize) -> Vec<HistoryEntry> {
        self.list(HistoryFilter {
            limit: count,
            ..Default::default()
        })
        .await
    }

    pub async fn get_by_type(&self, entry_type: HistoryType, limit: usize) -> Vec<HistoryEntry> {
        self.list(HistoryFilter {
            entry_type: Some(entry_type),
            limit,
            ..Default::default()
        })
        .await
    }

    pub async fn clear(&self) -> anyhow::Result<()> {
        // Targeted delete: walk units and drop the history-tagged ones.
        let units =
            self.storage
                .list_units(&self.bank_id, Some(FactType::Experience), 10_000, 0)?;
        for u in units {
            if u.tags.iter().any(|t| t == HISTORY_TAG) {
                self.storage.delete_unit(&self.bank_id, &u.id)?;
            }
        }
        Ok(())
    }

    pub async fn stats(&self) -> HistoryStats {
        let all = self
            .list(HistoryFilter {
                limit: usize::MAX,
                ..Default::default()
            })
            .await;
        let mut commands = 0;
        let mut queries = 0;
        let mut tool_calls = 0;
        let mut successful = 0;
        let mut failed = 0;
        for e in &all {
            match e.entry_type {
                HistoryType::Command => commands += 1,
                HistoryType::Query => queries += 1,
                HistoryType::ToolCall => tool_calls += 1,
                _ => {}
            }
            if e.success {
                successful += 1;
            } else {
                failed += 1;
            }
        }
        HistoryStats {
            total_entries: all.len(),
            commands,
            queries,
            tool_calls,
            successful,
            failed,
        }
    }

    /// Kept for API compatibility — there is no separate history file to
    /// load, it's all in the unified store now.
    pub async fn load(&self) -> anyhow::Result<()> {
        Ok(())
    }
}

fn unit_to_history_entry(unit: &MemoryUnit) -> HistoryEntry {
    let id = unit
        .metadata
        .get("agentrust_id")
        .cloned()
        .unwrap_or_else(|| unit.id.clone());
    let entry_type = unit
        .metadata
        .get("entry_type")
        .and_then(|s| history_type_from_tag(s))
        .or_else(|| {
            unit.tags.iter().find_map(|t| {
                t.strip_prefix("history_type:")
                    .and_then(history_type_from_tag)
            })
        })
        .unwrap_or(HistoryType::Command);
    let success = unit
        .metadata
        .get("success")
        .and_then(|s| s.parse().ok())
        .unwrap_or(true);
    let duration_ms = unit
        .metadata
        .get("duration_ms")
        .and_then(|s| s.parse().ok());
    let session_id = unit
        .metadata
        .get("session_id")
        .cloned()
        .or_else(|| unit.document_id.clone());
    let timestamp = unit.event_date.unwrap_or(unit.mentioned_at);
    let metadata = unit
        .metadata
        .get("payload")
        .and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok())
        .unwrap_or(serde_json::Value::Null);

    HistoryEntry {
        id,
        entry_type,
        content: unit.text.clone(),
        timestamp,
        session_id,
        success,
        duration_ms,
        metadata,
    }
}
