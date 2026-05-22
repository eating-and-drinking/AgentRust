//! SQLite-backed storage layer.
//!
//! This is the AgentRust port of Hindsight's PostgreSQL/pgvector storage
//! plane (`hindsight_api/models.py` + the Alembic migrations). SQLite was
//! chosen so AgentRust can keep its single-binary, zero-ops CLI feel.
//!
//! Trade-offs vs upstream:
//! * Vector search is brute-force cosine over an in-process scan. SQLite
//!   has no native ANN index, and at AgentRust's expected scale
//!   (<10 K units per bank) brute force is fine.
//! * BM25 is wired up via SQLite's `fts5` virtual table.
//! * Everything else (entity graph, links, observation history) maps
//!   directly to ordinary rowid tables.
//!
//! `StorageBackend` is preserved as a public type for API compatibility —
//! every variant resolves to the same SQLite implementation.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};

use super::embed::cosine;
use super::model::{
    Bank, Document, Entity, FactType, LinkType, MemoryLink, MemoryUnit, ObservationHistoryEntry,
    TagMatch,
};

/// Legacy alias retained for API compatibility. All variants now route to
/// SQLite.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub enum StorageBackend {
    #[default]
    Sqlite,
    File,
    Memory,
}

pub struct Storage {
    conn: Arc<Mutex<Connection>>,
    db_path: PathBuf,
    backend: StorageBackend,
}

impl Storage {
    pub fn open(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let db_path = path.as_ref().to_path_buf();
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let conn = Connection::open(&db_path)?;
        Self::install_schema(&conn)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            db_path,
            backend: StorageBackend::Sqlite,
        })
    }

    /// In-memory database. Useful for tests and ephemeral runs.
    pub fn open_in_memory() -> anyhow::Result<Self> {
        let conn = Connection::open_in_memory()?;
        Self::install_schema(&conn)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            db_path: PathBuf::from(":memory:"),
            backend: StorageBackend::Memory,
        })
    }

    pub fn db_path(&self) -> &Path {
        &self.db_path
    }

    pub fn backend(&self) -> &StorageBackend {
        &self.backend
    }

    pub fn with_backend(mut self, b: StorageBackend) -> Self {
        self.backend = b;
        self
    }

    fn install_schema(conn: &Connection) -> anyhow::Result<()> {
        conn.execute_batch(
            r#"
            PRAGMA journal_mode = WAL;
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS banks (
                bank_id TEXT PRIMARY KEY,
                background TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS documents (
                id TEXT NOT NULL,
                bank_id TEXT NOT NULL REFERENCES banks(bank_id) ON DELETE CASCADE,
                original_text TEXT,
                content_hash TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (id, bank_id)
            );

            CREATE TABLE IF NOT EXISTS memory_units (
                id TEXT PRIMARY KEY,
                bank_id TEXT NOT NULL REFERENCES banks(bank_id) ON DELETE CASCADE,
                document_id TEXT,
                text TEXT NOT NULL,
                fact_type TEXT NOT NULL CHECK (fact_type IN ('world','experience','observation')),
                context TEXT,
                event_date TEXT,
                occurred_start TEXT,
                occurred_end TEXT,
                mentioned_at TEXT NOT NULL,
                tags TEXT NOT NULL,
                metadata TEXT NOT NULL,
                embedding BLOB NOT NULL,
                source_memory_ids TEXT NOT NULL,
                proof_count INTEGER NOT NULL DEFAULT 0,
                history TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_units_bank_type_date
                ON memory_units(bank_id, fact_type, event_date DESC);
            CREATE INDEX IF NOT EXISTS idx_units_doc
                ON memory_units(document_id);

            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                bank_id TEXT NOT NULL REFERENCES banks(bank_id) ON DELETE CASCADE,
                canonical_name TEXT NOT NULL,
                entity_type TEXT,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                mention_count INTEGER NOT NULL DEFAULT 0,
                UNIQUE (bank_id, canonical_name)
            );

            CREATE TABLE IF NOT EXISTS unit_entities (
                unit_id TEXT NOT NULL REFERENCES memory_units(id) ON DELETE CASCADE,
                entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
                PRIMARY KEY (unit_id, entity_id)
            );

            CREATE TABLE IF NOT EXISTS memory_links (
                from_unit_id TEXT NOT NULL REFERENCES memory_units(id) ON DELETE CASCADE,
                to_unit_id TEXT NOT NULL REFERENCES memory_units(id) ON DELETE CASCADE,
                link_type TEXT NOT NULL,
                entity_id TEXT,
                weight REAL NOT NULL DEFAULT 0.0,
                PRIMARY KEY (from_unit_id, to_unit_id, link_type, COALESCE(entity_id, ''))
            );
            CREATE INDEX IF NOT EXISTS idx_links_to ON memory_links(to_unit_id);
            CREATE INDEX IF NOT EXISTS idx_links_entity ON memory_links(entity_id);

            -- BM25 / full-text-search virtual table mirroring Hindsight's
            -- `text_signals` GIN-on-tsvector arm.
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_units_fts USING fts5(
                unit_id UNINDEXED,
                bank_id UNINDEXED,
                text,
                tags,
                tokenize = 'unicode61'
            );
            "#,
        )?;
        Ok(())
    }

    // -----------------------------------------------------------------
    // Bank operations
    // -----------------------------------------------------------------

    pub fn upsert_bank(&self, bank: &Bank) -> anyhow::Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO banks (bank_id, background, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(bank_id) DO UPDATE SET
                background = excluded.background,
                updated_at = excluded.updated_at",
            params![
                bank.bank_id,
                bank.background,
                bank.created_at.to_rfc3339(),
                bank.updated_at.to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    pub fn ensure_bank(&self, bank_id: &str) -> anyhow::Result<()> {
        let conn = self.conn.lock().unwrap();
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "INSERT OR IGNORE INTO banks (bank_id, background, created_at, updated_at)
             VALUES (?1, NULL, ?2, ?2)",
            params![bank_id, now],
        )?;
        Ok(())
    }

    pub fn delete_bank(&self, bank_id: &str) -> anyhow::Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute("DELETE FROM banks WHERE bank_id = ?1", params![bank_id])?;
        conn.execute(
            "DELETE FROM memory_units_fts WHERE bank_id = ?1",
            params![bank_id],
        )?;
        Ok(())
    }

    // -----------------------------------------------------------------
    // Document operations
    // -----------------------------------------------------------------

    pub fn upsert_document(&self, doc: &Document) -> anyhow::Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO documents (id, bank_id, original_text, content_hash, metadata, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT(id, bank_id) DO UPDATE SET
                original_text = excluded.original_text,
                content_hash = excluded.content_hash,
                metadata = excluded.metadata,
                updated_at = excluded.updated_at",
            params![
                doc.id,
                doc.bank_id,
                doc.original_text,
                doc.content_hash,
                serde_json::to_string(&doc.metadata)?,
                doc.created_at.to_rfc3339(),
                doc.updated_at.to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    pub fn get_document(&self, bank_id: &str, id: &str) -> anyhow::Result<Option<Document>> {
        let conn = self.conn.lock().unwrap();
        let row = conn
            .query_row(
                "SELECT id, bank_id, original_text, content_hash, metadata, created_at, updated_at
                 FROM documents WHERE id = ?1 AND bank_id = ?2",
                params![id, bank_id],
                Self::row_to_document,
            )
            .optional()?;
        Ok(row)
    }

    pub fn list_documents(&self, bank_id: &str) -> anyhow::Result<Vec<Document>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, bank_id, original_text, content_hash, metadata, created_at, updated_at
             FROM documents WHERE bank_id = ?1 ORDER BY updated_at DESC",
        )?;
        let rows = stmt
            .query_map(params![bank_id], Self::row_to_document)?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    pub fn delete_document(&self, bank_id: &str, id: &str) -> anyhow::Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "DELETE FROM memory_units_fts
             WHERE unit_id IN (SELECT id FROM memory_units WHERE document_id = ?1 AND bank_id = ?2)",
            params![id, bank_id],
        )?;
        conn.execute(
            "DELETE FROM documents WHERE id = ?1 AND bank_id = ?2",
            params![id, bank_id],
        )?;
        Ok(())
    }

    // -----------------------------------------------------------------
    // MemoryUnit operations
    // -----------------------------------------------------------------

    pub fn insert_unit(&self, unit: &MemoryUnit) -> anyhow::Result<()> {
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction()?;
        Self::insert_unit_in(&tx, unit)?;
        tx.commit()?;
        Ok(())
    }

    pub fn insert_units(&self, units: &[MemoryUnit]) -> anyhow::Result<()> {
        if units.is_empty() {
            return Ok(());
        }
        let mut conn = self.conn.lock().unwrap();
        let tx = conn.transaction()?;
        for u in units {
            Self::insert_unit_in(&tx, u)?;
        }
        tx.commit()?;
        Ok(())
    }

    fn insert_unit_in(tx: &rusqlite::Transaction, unit: &MemoryUnit) -> anyhow::Result<()> {
        tx.execute(
            "INSERT OR REPLACE INTO memory_units
             (id, bank_id, document_id, text, fact_type, context,
              event_date, occurred_start, occurred_end, mentioned_at,
              tags, metadata, embedding, source_memory_ids, proof_count, history, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)",
            params![
                unit.id,
                unit.bank_id,
                unit.document_id,
                unit.text,
                unit.fact_type.as_str(),
                unit.context,
                unit.event_date.map(|d| d.to_rfc3339()),
                unit.occurred_start.map(|d| d.to_rfc3339()),
                unit.occurred_end.map(|d| d.to_rfc3339()),
                unit.mentioned_at.to_rfc3339(),
                serde_json::to_string(&unit.tags)?,
                serde_json::to_string(&unit.metadata)?,
                embedding_to_blob(&unit.embedding),
                serde_json::to_string(&unit.source_memory_ids)?,
                unit.proof_count,
                serde_json::to_string(&unit.history)?,
                unit.created_at.to_rfc3339(),
            ],
        )?;
        // FTS5 mirror — rebuild row by deleting then inserting.
        tx.execute(
            "DELETE FROM memory_units_fts WHERE unit_id = ?1",
            params![unit.id],
        )?;
        tx.execute(
            "INSERT INTO memory_units_fts (unit_id, bank_id, text, tags) VALUES (?1, ?2, ?3, ?4)",
            params![unit.id, unit.bank_id, unit.text, unit.tags.join(" ")],
        )?;
        Ok(())
    }

    pub fn get_unit(&self, bank_id: &str, id: &str) -> anyhow::Result<Option<MemoryUnit>> {
        let conn = self.conn.lock().unwrap();
        let unit = conn
            .query_row(
                "SELECT id, bank_id, document_id, text, fact_type, context,
                        event_date, occurred_start, occurred_end, mentioned_at,
                        tags, metadata, embedding, source_memory_ids, proof_count, history, created_at
                 FROM memory_units WHERE id = ?1 AND bank_id = ?2",
                params![id, bank_id],
                Self::row_to_unit,
            )
            .optional()?;
        Ok(unit)
    }

    pub fn delete_unit(&self, bank_id: &str, id: &str) -> anyhow::Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "DELETE FROM memory_units_fts WHERE unit_id = ?1 AND bank_id = ?2",
            params![id, bank_id],
        )?;
        conn.execute(
            "DELETE FROM memory_units WHERE id = ?1 AND bank_id = ?2",
            params![id, bank_id],
        )?;
        Ok(())
    }

    pub fn list_units(
        &self,
        bank_id: &str,
        fact_type: Option<FactType>,
        limit: usize,
        offset: usize,
    ) -> anyhow::Result<Vec<MemoryUnit>> {
        let conn = self.conn.lock().unwrap();
        let (sql, units) = if let Some(ft) = fact_type {
            let mut stmt = conn.prepare(
                "SELECT id, bank_id, document_id, text, fact_type, context,
                        event_date, occurred_start, occurred_end, mentioned_at,
                        tags, metadata, embedding, source_memory_ids, proof_count, history, created_at
                 FROM memory_units
                 WHERE bank_id = ?1 AND fact_type = ?2
                 ORDER BY COALESCE(event_date, mentioned_at) DESC
                 LIMIT ?3 OFFSET ?4",
            )?;
            let rows = stmt
                .query_map(
                    params![bank_id, ft.as_str(), limit as i64, offset as i64],
                    Self::row_to_unit,
                )?
                .collect::<Result<Vec<_>, _>>()?;
            ("", rows)
        } else {
            let mut stmt = conn.prepare(
                "SELECT id, bank_id, document_id, text, fact_type, context,
                        event_date, occurred_start, occurred_end, mentioned_at,
                        tags, metadata, embedding, source_memory_ids, proof_count, history, created_at
                 FROM memory_units
                 WHERE bank_id = ?1
                 ORDER BY COALESCE(event_date, mentioned_at) DESC
                 LIMIT ?2 OFFSET ?3",
            )?;
            let rows = stmt
                .query_map(
                    params![bank_id, limit as i64, offset as i64],
                    Self::row_to_unit,
                )?
                .collect::<Result<Vec<_>, _>>()?;
            ("", rows)
        };
        let _ = sql; // silence unused
        Ok(units)
    }

    pub fn count_units(&self, bank_id: &str) -> anyhow::Result<usize> {
        let conn = self.conn.lock().unwrap();
        let n: i64 = conn.query_row(
            "SELECT COUNT(*) FROM memory_units WHERE bank_id = ?1",
            params![bank_id],
            |r| r.get(0),
        )?;
        Ok(n as usize)
    }

    /// Clear every unit (and dependent rows) for a given bank.
    pub fn clear_bank(&self, bank_id: &str) -> anyhow::Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "DELETE FROM memory_units_fts WHERE bank_id = ?1",
            params![bank_id],
        )?;
        conn.execute(
            "DELETE FROM memory_units WHERE bank_id = ?1",
            params![bank_id],
        )?;
        conn.execute(
            "DELETE FROM documents WHERE bank_id = ?1",
            params![bank_id],
        )?;
        conn.execute(
            "DELETE FROM entities WHERE bank_id = ?1",
            params![bank_id],
        )?;
        Ok(())
    }

    pub fn db_size_bytes(&self) -> anyhow::Result<u64> {
        if self.db_path == PathBuf::from(":memory:") {
            return Ok(0);
        }
        Ok(std::fs::metadata(&self.db_path)?.len())
    }

    // -----------------------------------------------------------------
    // Entity operations
    // -----------------------------------------------------------------

    pub fn upsert_entity(&self, entity: &Entity) -> anyhow::Result<String> {
        let conn = self.conn.lock().unwrap();
        // Try to find an existing one first.
        let existing: Option<String> = conn
            .query_row(
                "SELECT id FROM entities WHERE bank_id = ?1 AND canonical_name = ?2",
                params![entity.bank_id, entity.canonical_name],
                |r| r.get(0),
            )
            .optional()?;
        if let Some(id) = existing {
            conn.execute(
                "UPDATE entities SET last_seen = ?1, mention_count = mention_count + 1
                 WHERE id = ?2",
                params![entity.last_seen.to_rfc3339(), id],
            )?;
            return Ok(id);
        }
        conn.execute(
            "INSERT INTO entities (id, bank_id, canonical_name, entity_type, first_seen, last_seen, mention_count)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                entity.id,
                entity.bank_id,
                entity.canonical_name,
                entity.entity_type,
                entity.first_seen.to_rfc3339(),
                entity.last_seen.to_rfc3339(),
                entity.mention_count.max(1),
            ],
        )?;
        Ok(entity.id.clone())
    }

    pub fn link_unit_to_entity(&self, unit_id: &str, entity_id: &str) -> anyhow::Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR IGNORE INTO unit_entities (unit_id, entity_id) VALUES (?1, ?2)",
            params![unit_id, entity_id],
        )?;
        Ok(())
    }

    pub fn entities_for_unit(&self, unit_id: &str) -> anyhow::Result<Vec<String>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT entity_id FROM unit_entities WHERE unit_id = ?1",
        )?;
        let rows = stmt
            .query_map(params![unit_id], |r| r.get::<_, String>(0))?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    // -----------------------------------------------------------------
    // Memory links
    // -----------------------------------------------------------------

    pub fn insert_link(&self, link: &MemoryLink) -> anyhow::Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO memory_links (from_unit_id, to_unit_id, link_type, entity_id, weight)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                link.from_unit_id,
                link.to_unit_id,
                link.link_type.as_str(),
                link.entity_id,
                link.weight,
            ],
        )?;
        Ok(())
    }

    pub fn neighbors(
        &self,
        unit_id: &str,
        link_type: Option<LinkType>,
    ) -> anyhow::Result<Vec<(String, f32)>> {
        let conn = self.conn.lock().unwrap();
        let sql = match link_type {
            Some(_) => "SELECT to_unit_id, weight FROM memory_links
                        WHERE from_unit_id = ?1 AND link_type = ?2
                        UNION ALL
                        SELECT from_unit_id, weight FROM memory_links
                        WHERE to_unit_id = ?1 AND link_type = ?2",
            None => "SELECT to_unit_id, weight FROM memory_links WHERE from_unit_id = ?1
                     UNION ALL
                     SELECT from_unit_id, weight FROM memory_links WHERE to_unit_id = ?1",
        };
        let mut stmt = conn.prepare(sql)?;
        let rows = if let Some(lt) = link_type {
            stmt.query_map(params![unit_id, lt.as_str()], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, f32>(1)?))
            })?
            .collect::<Result<Vec<_>, _>>()?
        } else {
            stmt.query_map(params![unit_id], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, f32>(1)?))
            })?
            .collect::<Result<Vec<_>, _>>()?
        };
        Ok(rows)
    }

    // -----------------------------------------------------------------
    // Vector + BM25 retrieval primitives
    // -----------------------------------------------------------------

    /// Brute-force cosine search. Returns `(unit, score)` pairs in
    /// descending score order. `fact_types` filters by `fact_type IN (...)`
    /// when non-empty; an empty slice means "all".
    pub fn semantic_search(
        &self,
        bank_id: &str,
        query_vec: &[f32],
        fact_types: &[FactType],
        tags: &[String],
        tag_match: TagMatch,
        limit: usize,
    ) -> anyhow::Result<Vec<(MemoryUnit, f32)>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, bank_id, document_id, text, fact_type, context,
                    event_date, occurred_start, occurred_end, mentioned_at,
                    tags, metadata, embedding, source_memory_ids, proof_count, history, created_at
             FROM memory_units WHERE bank_id = ?1",
        )?;
        let rows = stmt
            .query_map(params![bank_id], Self::row_to_unit)?
            .collect::<Result<Vec<_>, _>>()?;

        let mut scored: Vec<(MemoryUnit, f32)> = rows
            .into_iter()
            .filter(|u| fact_types.is_empty() || fact_types.contains(&u.fact_type))
            .filter(|u| tags_match(&u.tags, tags, tag_match))
            .map(|u| {
                let s = cosine(&u.embedding, query_vec);
                (u, s)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);
        Ok(scored)
    }

    /// BM25 keyword search via the FTS5 mirror.
    pub fn bm25_search(
        &self,
        bank_id: &str,
        query: &str,
        fact_types: &[FactType],
        tags: &[String],
        tag_match: TagMatch,
        limit: usize,
    ) -> anyhow::Result<Vec<(MemoryUnit, f32)>> {
        if query.trim().is_empty() {
            return Ok(Vec::new());
        }
        let conn = self.conn.lock().unwrap();
        // bm25() returns a negative-ish score; lower is better. Convert to
        // a positive rank by negation.
        let sql = "SELECT m.id, m.bank_id, m.document_id, m.text, m.fact_type, m.context,
                          m.event_date, m.occurred_start, m.occurred_end, m.mentioned_at,
                          m.tags, m.metadata, m.embedding, m.source_memory_ids, m.proof_count,
                          m.history, m.created_at, bm25(memory_units_fts) as score
                   FROM memory_units_fts
                   JOIN memory_units m ON m.id = memory_units_fts.unit_id
                   WHERE memory_units_fts MATCH ?1 AND m.bank_id = ?2
                   ORDER BY score ASC
                   LIMIT ?3";
        let mut stmt = conn.prepare(sql)?;
        let q = sanitize_fts_query(query);
        let raw = stmt
            .query_map(params![q, bank_id, (limit * 3) as i64], |row| {
                let unit = Self::row_to_unit(row)?;
                let score: f64 = row.get(17)?;
                Ok((unit, score as f32))
            })?
            .collect::<Result<Vec<_>, _>>()?;

        let mut out: Vec<(MemoryUnit, f32)> = raw
            .into_iter()
            .filter(|(u, _)| fact_types.is_empty() || fact_types.contains(&u.fact_type))
            .filter(|(u, _)| tags_match(&u.tags, tags, tag_match))
            .map(|(u, s)| (u, -s)) // higher = better
            .collect();
        out.truncate(limit);
        Ok(out)
    }

    // -----------------------------------------------------------------
    // Row decoders
    // -----------------------------------------------------------------

    fn row_to_document(row: &rusqlite::Row) -> rusqlite::Result<Document> {
        let metadata_str: Option<String> = row.get(4)?;
        let metadata: HashMap<String, String> = metadata_str
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        Ok(Document {
            id: row.get(0)?,
            bank_id: row.get(1)?,
            original_text: row.get(2)?,
            content_hash: row.get(3)?,
            metadata,
            created_at: parse_dt(row.get::<_, String>(5)?)?,
            updated_at: parse_dt(row.get::<_, String>(6)?)?,
        })
    }

    fn row_to_unit(row: &rusqlite::Row) -> rusqlite::Result<MemoryUnit> {
        let fact_type: String = row.get(4)?;
        let tags_str: String = row.get(10)?;
        let metadata_str: String = row.get(11)?;
        let embedding_blob: Vec<u8> = row.get(12)?;
        let sources_str: String = row.get(13)?;
        let history_str: String = row.get(15)?;
        Ok(MemoryUnit {
            id: row.get(0)?,
            bank_id: row.get(1)?,
            document_id: row.get(2)?,
            text: row.get(3)?,
            fact_type: FactType::from_str(&fact_type).unwrap_or(FactType::Experience),
            context: row.get(5)?,
            event_date: parse_opt_dt(row.get::<_, Option<String>>(6)?),
            occurred_start: parse_opt_dt(row.get::<_, Option<String>>(7)?),
            occurred_end: parse_opt_dt(row.get::<_, Option<String>>(8)?),
            mentioned_at: parse_dt(row.get::<_, String>(9)?)?,
            tags: serde_json::from_str(&tags_str).unwrap_or_default(),
            metadata: serde_json::from_str(&metadata_str).unwrap_or_default(),
            embedding: blob_to_embedding(&embedding_blob),
            source_memory_ids: serde_json::from_str(&sources_str).unwrap_or_default(),
            proof_count: row.get::<_, i64>(14)? as u32,
            history: serde_json::from_str::<Vec<ObservationHistoryEntry>>(&history_str)
                .unwrap_or_default(),
            created_at: parse_dt(row.get::<_, String>(16)?)?,
        })
    }
}

fn parse_dt(s: String) -> rusqlite::Result<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(&s)
        .map(|d| d.with_timezone(&Utc))
        .map_err(|e| rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(e)))
}

fn parse_opt_dt(s: Option<String>) -> Option<DateTime<Utc>> {
    s.as_deref()
        .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
        .map(|d| d.with_timezone(&Utc))
}

fn embedding_to_blob(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn blob_to_embedding(b: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(b.len() / 4);
    for chunk in b.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    out
}

fn sanitize_fts_query(q: &str) -> String {
    // FTS5 treats unquoted single-quotes and a handful of punctuation
    // as syntax. Build a permissive OR of trigram-ish words instead of
    // forwarding the raw user string.
    let mut parts: Vec<String> = q
        .split_whitespace()
        .filter(|w| w.chars().any(char::is_alphanumeric))
        .map(|w| {
            let cleaned: String = w
                .chars()
                .filter(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if cleaned.is_empty() {
                String::new()
            } else {
                format!("\"{}\"", cleaned)
            }
        })
        .filter(|s| !s.is_empty())
        .collect();
    if parts.is_empty() {
        return String::from("\"\"");
    }
    parts.dedup();
    parts.join(" OR ")
}

/// Returns true when `unit_tags` satisfies a `(query_tags, mode)` filter.
/// Mirrors the four modes from Hindsight's `engine/search/tags.py`.
pub fn tags_match(unit_tags: &[String], query_tags: &[String], mode: TagMatch) -> bool {
    if query_tags.is_empty() {
        return true;
    }
    let untagged = unit_tags.is_empty();
    match mode {
        TagMatch::Any => {
            untagged || query_tags.iter().any(|q| unit_tags.iter().any(|t| t == q))
        }
        TagMatch::All => {
            untagged || query_tags.iter().all(|q| unit_tags.iter().any(|t| t == q))
        }
        TagMatch::AnyStrict => {
            !untagged && query_tags.iter().any(|q| unit_tags.iter().any(|t| t == q))
        }
        TagMatch::AllStrict => {
            !untagged && query_tags.iter().all(|q| unit_tags.iter().any(|t| t == q))
        }
    }
}
