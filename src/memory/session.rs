//! Session management on top of the native memory engine.
//!
//! Sessions correspond to [`super::model::Document`] rows; each message is
//! a `MemoryUnit` of `fact_type=experience` whose `document_id` equals the
//! session id. This mirrors Hindsight's "documents group a conversation"
//! convention.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use super::model::{Document, FactType};
use super::recall::{RecallEngine, RecallRequest};
use super::retain::{RetainEngine, RetainItem};
use super::storage::Storage;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub name: String,
    pub project_path: Option<PathBuf>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub messages: Vec<SessionMessage>,
    pub metadata: HashMap<String, serde_json::Value>,
    pub status: SessionStatus,
}

impl Session {
    pub fn new(name: Option<&str>) -> Self {
        let id = uuid::Uuid::new_v4().to_string();
        Self {
            id: id.clone(),
            name: name.unwrap_or(&id).to_string(),
            project_path: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            messages: Vec::new(),
            metadata: HashMap::new(),
            status: SessionStatus::Active,
        }
    }

    pub fn with_project(mut self, path: PathBuf) -> Self {
        self.project_path = Some(path);
        self
    }

    pub fn add_message(&mut self, role: &str, content: &str) {
        self.messages.push(SessionMessage {
            role: role.to_string(),
            content: content.to_string(),
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        });
        self.updated_at = Utc::now();
    }

    pub fn message_count(&self) -> usize {
        self.messages.len()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMessage {
    pub role: String,
    pub content: String,
    pub timestamp: DateTime<Utc>,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SessionStatus {
    Active,
    Paused,
    Archived,
    Error,
}

impl SessionStatus {
    fn as_str(&self) -> &'static str {
        match self {
            SessionStatus::Active => "active",
            SessionStatus::Paused => "paused",
            SessionStatus::Archived => "archived",
            SessionStatus::Error => "error",
        }
    }
    fn from_str(s: &str) -> Self {
        match s {
            "active" => SessionStatus::Active,
            "paused" => SessionStatus::Paused,
            "archived" => SessionStatus::Archived,
            "error" => SessionStatus::Error,
            _ => SessionStatus::Active,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInfo {
    pub id: String,
    pub name: String,
    pub project_path: Option<PathBuf>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub message_count: usize,
    pub status: SessionStatus,
}

pub struct SessionManager {
    bank_id: String,
    storage: Arc<Storage>,
    retain: Arc<RetainEngine>,
    recall: Arc<RecallEngine>,
    sessions: Arc<RwLock<HashMap<String, Session>>>,
    active: Arc<RwLock<Option<Session>>>,
}

impl SessionManager {
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
            sessions: Arc::new(RwLock::new(HashMap::new())),
            active: Arc::new(RwLock::new(None)),
        }
    }

    pub async fn create(&self, name: Option<&str>) -> anyhow::Result<Session> {
        let session = Session::new(name);
        self.save(&session).await?;
        let mut cache = self.sessions.write().await;
        cache.insert(session.id.clone(), session.clone());
        Ok(session)
    }

    pub async fn save(&self, session: &Session) -> anyhow::Result<()> {
        let mut doc = Document::new(session.id.clone(), self.bank_id.clone());
        doc.metadata
            .insert("session_name".to_string(), session.name.clone());
        doc.metadata
            .insert("status".to_string(), session.status.as_str().to_string());
        if let Some(p) = &session.project_path {
            doc.metadata
                .insert("project_path".to_string(), p.display().to_string());
        }
        doc.metadata
            .insert("message_count".to_string(), session.messages.len().to_string());
        doc.updated_at = session.updated_at;
        doc.created_at = session.created_at;
        self.storage.upsert_document(&doc)?;
        Ok(())
    }

    pub async fn load(&self, id: &str) -> anyhow::Result<Option<Session>> {
        if let Some(s) = self.sessions.read().await.get(id).cloned() {
            return Ok(Some(s));
        }
        let doc = match self.storage.get_document(&self.bank_id, id)? {
            Some(d) => d,
            None => return Ok(None),
        };
        let name = doc
            .metadata
            .get("session_name")
            .cloned()
            .unwrap_or_else(|| id.to_string());
        let project_path = doc.metadata.get("project_path").map(PathBuf::from);
        let status = doc
            .metadata
            .get("status")
            .map(|s| SessionStatus::from_str(s))
            .unwrap_or(SessionStatus::Active);

        // Reconstruct messages by listing all units that belong to this
        // document. Mirrors Hindsight's "document -> memory_units" join.
        let messages = self
            .storage
            .list_units(&self.bank_id, Some(FactType::Experience), 5000, 0)?
            .into_iter()
            .filter(|u| u.document_id.as_deref() == Some(id))
            .map(|u| {
                let role = u
                    .metadata
                    .get("role")
                    .cloned()
                    .unwrap_or_else(|| "user".to_string());
                SessionMessage {
                    role,
                    content: u.text,
                    timestamp: u.event_date.unwrap_or(u.mentioned_at),
                    metadata: HashMap::new(),
                }
            })
            .collect();

        let session = Session {
            id: id.to_string(),
            name,
            project_path,
            created_at: doc.created_at,
            updated_at: doc.updated_at,
            messages,
            metadata: HashMap::new(),
            status,
        };
        self.sessions
            .write()
            .await
            .insert(id.to_string(), session.clone());
        Ok(Some(session))
    }

    pub async fn delete(&self, id: &str) -> anyhow::Result<()> {
        self.storage.delete_document(&self.bank_id, id)?;
        self.sessions.write().await.remove(id);
        let mut active = self.active.write().await;
        if active.as_ref().map(|s| s.id == id).unwrap_or(false) {
            *active = None;
        }
        Ok(())
    }

    pub async fn list(&self) -> anyhow::Result<Vec<SessionInfo>> {
        let docs = self.storage.list_documents(&self.bank_id)?;
        let mut infos: Vec<SessionInfo> = docs
            .into_iter()
            .map(|d| {
                let name = d
                    .metadata
                    .get("session_name")
                    .cloned()
                    .unwrap_or_else(|| d.id.clone());
                let project_path = d.metadata.get("project_path").map(PathBuf::from);
                let status = d
                    .metadata
                    .get("status")
                    .map(|s| SessionStatus::from_str(s))
                    .unwrap_or(SessionStatus::Active);
                let message_count = d
                    .metadata
                    .get("message_count")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                SessionInfo {
                    id: d.id,
                    name,
                    project_path,
                    created_at: d.created_at,
                    updated_at: d.updated_at,
                    message_count,
                    status,
                }
            })
            .collect();

        // Overlay in-memory cache for sessions not yet flushed.
        let cache = self.sessions.read().await;
        for s in cache.values() {
            if !infos.iter().any(|i| i.id == s.id) {
                infos.push(SessionInfo {
                    id: s.id.clone(),
                    name: s.name.clone(),
                    project_path: s.project_path.clone(),
                    created_at: s.created_at,
                    updated_at: s.updated_at,
                    message_count: s.messages.len(),
                    status: s.status.clone(),
                });
            }
        }
        Ok(infos)
    }

    pub async fn get(&self, id: &str) -> Option<Session> {
        self.sessions.read().await.get(id).cloned()
    }

    pub async fn set_active(&self, session: Session) {
        *self.active.write().await = Some(session);
    }

    pub async fn get_active(&self) -> Option<Session> {
        self.active.read().await.clone()
    }

    pub async fn clear_active(&self) {
        *self.active.write().await = None;
    }

    pub async fn add_message(&self, id: &str, role: &str, content: &str) -> anyhow::Result<()> {
        // Update cache.
        {
            let mut cache = self.sessions.write().await;
            if let Some(s) = cache.get_mut(id) {
                s.add_message(role, content);
            }
        }
        // Persist the message itself as a `MemoryUnit`.
        let mut metadata = HashMap::new();
        metadata.insert("role".to_string(), role.to_string());
        metadata.insert("session_id".to_string(), id.to_string());

        let mut tags = vec![
            "type:conversation".to_string(),
            format!("session:{id}"),
            format!("role:{role}"),
        ];
        if role == "system" {
            tags.push("type:session".to_string());
        }

        self.retain
            .retain(
                &self.bank_id,
                vec![RetainItem {
                    content: content.to_string(),
                    timestamp: Some(Utc::now()),
                    context: Some(format!("session/{id}/{role}")),
                    document_id: Some(id.to_string()),
                    tags,
                    metadata,
                    raw_fact_type: Some(FactType::Experience),
                }],
            )
            .await?;
        // Refresh the document so message_count metadata is current.
        let snapshot = self.sessions.read().await.get(id).cloned();
        if let Some(s) = snapshot {
            self.save(&s).await?;
        }
        Ok(())
    }

    pub async fn archive(&self, id: &str) -> anyhow::Result<()> {
        let snapshot = {
            let mut cache = self.sessions.write().await;
            if let Some(s) = cache.get_mut(id) {
                s.status = SessionStatus::Archived;
                Some(s.clone())
            } else {
                None
            }
        };
        if let Some(s) = snapshot {
            self.save(&s).await?;
        }
        Ok(())
    }

    pub async fn search(&self, query: &str) -> Vec<SessionInfo> {
        let resp = self
            .recall
            .recall(
                &self.bank_id,
                RecallRequest {
                    query: query.to_string(),
                    fact_types: vec![FactType::Experience],
                    tags: vec!["type:conversation".to_string()],
                    tag_match: super::model::TagMatch::Any,
                    budget: super::model::Budget::Mid,
                    max_tokens: Some(4096),
                },
            )
            .await
            .unwrap_or_default();

        let mut seen: HashMap<String, SessionInfo> = HashMap::new();
        for hit in resp.results {
            let session_id = match hit.unit.document_id.clone() {
                Some(id) => id,
                None => continue,
            };
            seen.entry(session_id.clone()).or_insert(SessionInfo {
                id: session_id.clone(),
                name: session_id,
                project_path: None,
                created_at: Utc::now(),
                updated_at: hit.unit.event_date.unwrap_or(hit.unit.mentioned_at),
                message_count: 0,
                status: SessionStatus::Active,
            });
        }
        seen.into_values().collect()
    }
}
