//! Recall pipeline.
//!
//! Hindsight's `engine/search/retrieval.py` runs four parallel retrieval
//! arms (semantic, BM25, graph, temporal) and fuses them with Reciprocal
//! Rank Fusion (`engine/search/fusion.py`, `k = 60`). We port:
//!
//! * Semantic arm — vector cosine search against `memory_units`.
//! * BM25 arm — SQLite FTS5 over the `memory_units_fts` mirror.
//! * Graph arm — 1-hop expansion through `memory_links` from the top
//!   semantic seeds.
//!
//! The temporal arm and cross-encoder rerank are intentionally simplified;
//! see Hindsight's docs for the full algorithm.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::embed::SharedEmbedder;
use super::model::{Budget, FactType, MemoryUnit, TagMatch};
use super::reranker::HeuristicReranker;
use super::storage::Storage;

/// One recall request — mirrors Hindsight's `RecallRequest`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RecallRequest {
    pub query: String,
    #[serde(default)]
    pub fact_types: Vec<FactType>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub tag_match: TagMatch,
    #[serde(default)]
    pub budget: Budget,
    /// Soft cap on returned content size. Counted in characters.
    #[serde(default)]
    pub max_tokens: Option<usize>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RecallResponse {
    pub results: Vec<RecallHit>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallHit {
    pub unit: MemoryUnit,
    /// Final fused score after RRF.
    pub score: f32,
    /// Per-arm rank trace (1-based; absent if the arm didn't surface it).
    pub semantic_rank: Option<usize>,
    pub bm25_rank: Option<usize>,
    pub graph_rank: Option<usize>,
}

pub struct RecallEngine {
    storage: Arc<Storage>,
    embedder: SharedEmbedder,
    /// Reciprocal Rank Fusion constant; Hindsight uses 60.
    rrf_k: f32,
    /// Optional heuristic reranker. When set, replaces the legacy inline
    /// recency × proof boost with the AgentCpp-style blend (0.55 RRF +
    /// 0.20 recency + 0.25 temporal proximity).
    reranker: Option<HeuristicReranker>,
    /// Number of graph-expansion hops. `1` matches the original behaviour;
    /// the multi-hop spread is wired via [`crate::memory::graph_retrieval`].
    graph_hops: u32,
}

impl RecallEngine {
    pub fn new(storage: Arc<Storage>, embedder: SharedEmbedder) -> Self {
        Self {
            storage,
            embedder,
            rrf_k: 60.0,
            reranker: None,
            graph_hops: 1,
        }
    }

    /// Enable the heuristic reranker. Returns `self` for builder-style use.
    pub fn with_reranker(mut self, r: HeuristicReranker) -> Self {
        self.reranker = Some(r);
        self
    }

    /// Set the graph expansion depth. `1` keeps the legacy 1-hop path; `2`
    /// and above route through `graph_retrieval::expand` with the default
    /// 0.5 hop-decay.
    pub fn with_graph_hops(mut self, hops: u32) -> Self {
        self.graph_hops = hops.max(1);
        self
    }

    pub fn graph_hops(&self) -> u32 {
        self.graph_hops
    }

    pub fn has_reranker(&self) -> bool {
        self.reranker.is_some()
    }

    pub async fn recall(
        &self,
        bank_id: &str,
        req: RecallRequest,
    ) -> anyhow::Result<RecallResponse> {
        let per_arm = req.budget.per_arm().min(500);

        // ----- Semantic arm ------------------------------------------
        let query_vec = if req.query.trim().is_empty() {
            vec![0.0; self.embedder.dimension()]
        } else {
            self.embedder
                .encode(&[req.query.clone()])
                .await?
                .into_iter()
                .next()
                .unwrap_or_else(|| vec![0.0; self.embedder.dimension()])
        };
        let semantic_hits = self.storage.semantic_search(
            bank_id,
            &query_vec,
            &req.fact_types,
            &req.tags,
            req.tag_match,
            per_arm,
        )?;

        // ----- BM25 arm ----------------------------------------------
        let bm25_hits = self.storage.bm25_search(
            bank_id,
            &req.query,
            &req.fact_types,
            &req.tags,
            req.tag_match,
            per_arm,
        )?;

        // ----- Graph arm: 1-hop expansion from semantic seeds --------
        let graph_hits = self.graph_expand(bank_id, &semantic_hits, per_arm)?;

        // ----- RRF fusion --------------------------------------------
        let mut fused: HashMap<String, FusedEntry> = HashMap::new();
        for (rank, (u, _)) in semantic_hits.iter().enumerate() {
            let e = fused.entry(u.id.clone()).or_insert_with(|| FusedEntry::new(u));
            e.score += 1.0 / (self.rrf_k + (rank + 1) as f32);
            e.semantic_rank.get_or_insert(rank + 1);
        }
        for (rank, (u, _)) in bm25_hits.iter().enumerate() {
            let e = fused.entry(u.id.clone()).or_insert_with(|| FusedEntry::new(u));
            e.score += 1.0 / (self.rrf_k + (rank + 1) as f32);
            e.bm25_rank.get_or_insert(rank + 1);
        }
        for (rank, (u, _)) in graph_hits.iter().enumerate() {
            let e = fused.entry(u.id.clone()).or_insert_with(|| FusedEntry::new(u));
            e.score += 1.0 / (self.rrf_k + (rank + 1) as f32);
            e.graph_rank.get_or_insert(rank + 1);
        }

        let mut results: Vec<RecallHit> = fused
            .into_values()
            .map(|f| RecallHit {
                unit: f.unit,
                score: f.score,
                semantic_rank: f.semantic_rank,
                bm25_rank: f.bm25_rank,
                graph_rank: f.graph_rank,
            })
            .collect();
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        // Reranking. If a HeuristicReranker is configured we use the
        // 0.55/0.20/0.25 RRF + recency + temporal blend ported from
        // AgentCpp. Otherwise we keep the legacy Hindsight-style
        // `final = score × recency × proof_count_boost`.
        let now = chrono::Utc::now();
        if let Some(rr) = &self.reranker {
            rr.rerank(&mut results, now, None);
        } else {
            for r in &mut results {
                let age_days = (now - r.unit.event_date.unwrap_or(r.unit.mentioned_at))
                    .num_days()
                    .max(0) as f32;
                let recency = (1.0 - age_days / 365.0).max(0.5);
                let proof = 1.0 + 0.1 * ((r.unit.proof_count as f32) - 0.5);
                r.score *= recency * proof.max(0.5);
            }
            results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        }

        // Token budget capping.
        if let Some(max_tokens) = req.max_tokens {
            let mut used = 0usize;
            let mut capped = Vec::new();
            for r in results.drain(..) {
                let cost = r.unit.text.len();
                if used + cost > max_tokens && !capped.is_empty() {
                    break;
                }
                used += cost;
                capped.push(r);
            }
            results = capped;
        }

        Ok(RecallResponse { results })
    }

    fn graph_expand(
        &self,
        bank_id: &str,
        seeds: &[(MemoryUnit, f32)],
        limit: usize,
    ) -> anyhow::Result<Vec<(MemoryUnit, f32)>> {
        if self.graph_hops > 1 {
            return self.graph_expand_multi_hop(bank_id, seeds, limit);
        }
        let mut by_id: HashMap<String, f32> = HashMap::new();
        for (u, score) in seeds.iter().take(20) {
            for (neighbor_id, weight) in self.storage.neighbors(&u.id, None)? {
                let entry = by_id.entry(neighbor_id).or_insert(0.0);
                *entry += score * weight;
            }
        }
        let mut ordered: Vec<(String, f32)> = by_id.into_iter().collect();
        ordered.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ordered.truncate(limit);
        let mut hits = Vec::with_capacity(ordered.len());
        for (id, score) in ordered {
            if let Some(unit) = self.storage.get_unit(bank_id, &id)? {
                hits.push((unit, score));
            }
        }
        Ok(hits)
    }

    /// Multi-hop spreading-activation expansion, routed through
    /// [`crate::memory::graph_retrieval::expand`]. Uses the configured
    /// `graph_hops` as the max depth and `0.5` as the per-hop decay.
    fn graph_expand_multi_hop(
        &self,
        bank_id: &str,
        seeds: &[(MemoryUnit, f32)],
        limit: usize,
    ) -> anyhow::Result<Vec<(MemoryUnit, f32)>> {
        use super::graph_retrieval::{expand as graph_expand, GraphConfig};

        let seed_pairs: Vec<(String, f32)> = seeds
            .iter()
            .take(20)
            .map(|(u, s)| (u.id.clone(), *s))
            .collect();
        let storage = self.storage.clone();
        let cfg = GraphConfig {
            hop_decay: 0.5,
            max_hops: self.graph_hops,
        };
        let activations = graph_expand(&seed_pairs, cfg, limit, |id| {
            storage.neighbors(id, None).unwrap_or_default()
        });
        let mut hits = Vec::with_capacity(activations.len());
        for h in activations {
            if let Some(unit) = self.storage.get_unit(bank_id, &h.unit_id)? {
                hits.push((unit, h.activation));
            }
        }
        Ok(hits)
    }
}

struct FusedEntry {
    unit: MemoryUnit,
    score: f32,
    semantic_rank: Option<usize>,
    bm25_rank: Option<usize>,
    graph_rank: Option<usize>,
}

impl FusedEntry {
    fn new(u: &MemoryUnit) -> Self {
        Self {
            unit: u.clone(),
            score: 0.0,
            semantic_rank: None,
            bm25_rank: None,
            graph_rank: None,
        }
    }
}
