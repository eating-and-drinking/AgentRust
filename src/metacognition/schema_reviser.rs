//! Layer 4 (schema revision) — `SchemaReviser`.
//!
//! Periodically inspects a bounded buffer of recent `FailureEvent`s,
//! clusters them by token-set Jaccard similarity, and proposes either:
//!
//! * a new competence dimension to add to `SelfBelief` (with a pessimistic
//!   prior `α=1, β=2`), or
//! * a new `SelfProposition` to add to `SelfModelStore`.
//!
//! Each proposal is gated by a novelty score (1 − max Jaccard against
//! existing dimensions/propositions) and an evidence-count floor.

use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::{HashSet, VecDeque};

use super::self_belief::SelfBelief;
use super::self_model_store::SelfModelStore;
use super::self_proposition::SelfProposition;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FailureEvent {
    /// Epoch ms; auto-filled on insertion if zero.
    pub ts: i64,
    pub tool: String,
    pub description: String,
    /// Optional coarse task category.
    pub task_type: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProposalKind {
    AddDimension,
    AddProposition,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaProposal {
    pub kind: ProposalKind,
    /// Dimension name (snake_case) or proposition text.
    pub name_or_text: String,
    pub tags: Vec<String>,
    pub rationale: String,
    pub evidence_count: i32,
    pub novelty: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviserConfig {
    pub min_evidence_count: usize,
    pub review_every_n_episodes: usize,
    pub cluster_jaccard_thresh: f64,
    pub novelty_threshold: f64,
    pub max_failures_buffer: usize,
}

impl Default for ReviserConfig {
    fn default() -> Self {
        Self {
            min_evidence_count: 3,
            review_every_n_episodes: 5,
            cluster_jaccard_thresh: 0.35,
            novelty_threshold: 0.6,
            max_failures_buffer: 200,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaReviser {
    cfg: ReviserConfig,
    failures: VecDeque<FailureEvent>,
    episode_count: i32,
    last_review_episode: i32,
}

impl Default for SchemaReviser {
    fn default() -> Self {
        Self::with_config(ReviserConfig::default())
    }
}

impl SchemaReviser {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_config(cfg: ReviserConfig) -> Self {
        Self {
            cfg,
            failures: VecDeque::new(),
            episode_count: 0,
            last_review_episode: 0,
        }
    }

    pub fn config(&self) -> &ReviserConfig {
        &self.cfg
    }

    pub fn failures(&self) -> &VecDeque<FailureEvent> {
        &self.failures
    }

    pub fn record_failure(&mut self, mut ev: FailureEvent) {
        if ev.ts == 0 {
            ev.ts = Utc::now().timestamp_millis();
        }
        self.failures.push_back(ev);
        while self.failures.len() > self.cfg.max_failures_buffer {
            self.failures.pop_front();
        }
    }

    /// Reserved hook — currently a no-op (mirrors the C++).
    pub fn record_success(&mut self, _dim_or_tag: &str) {}

    pub fn note_episode_complete(&mut self) {
        self.episode_count += 1;
    }

    pub fn should_review(&self) -> bool {
        self.failures.len() >= self.cfg.min_evidence_count
            && (self.episode_count - self.last_review_episode) as usize
                >= self.cfg.review_every_n_episodes
    }

    pub fn propose_revisions(
        &mut self,
        belief: &SelfBelief,
        store: &SelfModelStore,
    ) -> Vec<SchemaProposal> {
        let clusters = self.cluster_failures();
        let mut out = Vec::new();
        for cluster in clusters {
            if cluster.len() < self.cfg.min_evidence_count {
                continue;
            }
            let keywords = top_keywords(&cluster, 4);
            if keywords.is_empty() {
                continue;
            }
            let text = format!(
                "I tend to fail when the task involves {}.",
                keywords.join(", ")
            );
            let novelty = novelty_score(&keywords, belief, store);
            let matches_existing_dim =
                keywords.iter().any(|k| belief.has_dimension(k));
            let kind = if matches_existing_dim {
                ProposalKind::AddProposition
            } else {
                ProposalKind::AddDimension
            };
            let name_or_text = match kind {
                ProposalKind::AddDimension => {
                    let head: Vec<_> = keywords.iter().take(2).cloned().collect();
                    head.join("_")
                }
                ProposalKind::AddProposition => text.clone(),
            };
            let rationale = format!(
                "{} failures cluster around tokens [{}]",
                cluster.len(),
                keywords.join(", ")
            );
            out.push(SchemaProposal {
                kind,
                name_or_text,
                tags: keywords,
                rationale,
                evidence_count: cluster.len() as i32,
                novelty,
            });
        }
        self.last_review_episode = self.episode_count;
        out
    }

    /// Apply a proposal if it passes both gates. Returns whether anything
    /// was applied.
    pub fn apply(
        &self,
        prop: &SchemaProposal,
        belief: &mut SelfBelief,
        store: &mut SelfModelStore,
    ) -> bool {
        if prop.novelty < self.cfg.novelty_threshold
            || (prop.evidence_count as usize) < self.cfg.min_evidence_count
        {
            return false;
        }
        match prop.kind {
            ProposalKind::AddDimension => {
                if belief.has_dimension(&prop.name_or_text) {
                    return false;
                }
                // Pessimistic prior: α=1, β=2 — assume new dim is hard until
                // we've seen evidence to the contrary.
                belief.add_dimension(&prop.name_or_text, 1.0, 2.0);
                true
            }
            ProposalKind::AddProposition => {
                let conf = (0.4 + 0.1 * prop.evidence_count as f64).min(0.8);
                let sp = SelfProposition::new(prop.name_or_text.clone(), prop.tags.clone())
                    .with_confidence(conf)
                    .with_evidence_count(prop.evidence_count);
                store.add_proposition(sp);
                true
            }
        }
    }

    /// Greedy single-pass Jaccard clustering over `description+tool` tokens.
    fn cluster_failures(&self) -> Vec<Vec<&FailureEvent>> {
        let mut clusters: Vec<Vec<&FailureEvent>> = Vec::new();
        let mut centroids: Vec<HashSet<String>> = Vec::new();
        for ev in self.failures.iter() {
            let toks: HashSet<String> =
                tokenize(&format!("{} {}", ev.description, ev.tool)).into_iter().collect();
            if toks.is_empty() {
                continue;
            }
            let mut best_i: Option<usize> = None;
            let mut best_j = self.cfg.cluster_jaccard_thresh;
            for (i, c) in centroids.iter().enumerate() {
                let j = jaccard(&toks, c);
                if j >= best_j {
                    best_j = j;
                    best_i = Some(i);
                }
            }
            if let Some(i) = best_i {
                clusters[i].push(ev);
                centroids[i].extend(toks);
            } else {
                clusters.push(vec![ev]);
                centroids.push(toks);
            }
        }
        clusters
    }
}

fn tokenize(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut buf = String::new();
    for c in s.chars() {
        if c.is_alphanumeric() {
            for low in c.to_lowercase() {
                buf.push(low);
            }
        } else if !buf.is_empty() {
            if buf.chars().count() >= 3 && !is_stopword(&buf) {
                out.push(std::mem::take(&mut buf));
            } else {
                buf.clear();
            }
        }
    }
    if buf.chars().count() >= 3 && !is_stopword(&buf) {
        out.push(buf);
    }
    out
}

fn is_stopword(s: &str) -> bool {
    matches!(
        s,
        "the" | "and" | "for" | "with" | "this" | "that" | "from" | "have" | "are" | "was"
            | "but" | "not" | "you" | "your" | "all" | "any" | "can" | "out" | "use" | "than"
            | "into" | "when" | "then" | "what" | "which" | "while"
    )
}

fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let inter = a.intersection(b).count() as f64;
    let union = a.union(b).count() as f64;
    if union == 0.0 { 0.0 } else { inter / union }
}

fn top_keywords(cluster: &[&FailureEvent], k: usize) -> Vec<String> {
    use std::collections::HashMap;
    let mut counts: HashMap<String, i32> = HashMap::new();
    for ev in cluster {
        for tok in tokenize(&format!("{} {}", ev.description, ev.tool)) {
            *counts.entry(tok).or_insert(0) += 1;
        }
    }
    let mut v: Vec<(String, i32)> = counts.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    v.into_iter().take(k).map(|(t, _)| t).collect()
}

fn novelty_score(keywords: &[String], belief: &SelfBelief, store: &SelfModelStore) -> f64 {
    let kw_set: HashSet<String> = keywords.iter().cloned().collect();
    let mut max_j: f64 = 0.0;
    for dim in belief.dimensions() {
        let toks: HashSet<String> = tokenize(dim).into_iter().collect();
        let j = jaccard(&kw_set, &toks);
        if j > max_j {
            max_j = j;
        }
    }
    for p in store.propositions() {
        let mut toks: HashSet<String> = tokenize(&p.text).into_iter().collect();
        for t in &p.tags {
            for tk in tokenize(t) {
                toks.insert(tk);
            }
        }
        let j = jaccard(&kw_set, &toks);
        if j > max_j {
            max_j = j;
        }
    }
    (1.0 - max_j).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(desc: &str, tool: &str) -> FailureEvent {
        FailureEvent {
            ts: 0,
            tool: tool.to_string(),
            description: desc.to_string(),
            task_type: String::new(),
        }
    }

    #[test]
    fn no_review_until_enough_evidence_and_episodes() {
        let mut r = SchemaReviser::new();
        assert!(!r.should_review());
        for _ in 0..3 {
            r.record_failure(ev("file_edit failed: file not found", "file_edit"));
        }
        // Not enough episodes yet.
        for _ in 0..2 {
            r.note_episode_complete();
        }
        assert!(!r.should_review());
        for _ in 0..3 {
            r.note_episode_complete();
        }
        assert!(r.should_review());
    }

    #[test]
    fn clusters_similar_failures() {
        let mut r = SchemaReviser::new();
        for _ in 0..5 {
            r.record_failure(ev("file_edit failed: path missing", "file_edit"));
        }
        for _ in 0..3 {
            r.record_failure(ev(
                "execute_command failed: command not found",
                "execute_command",
            ));
        }
        for _ in 0..5 {
            r.note_episode_complete();
        }
        let belief = SelfBelief::new();
        let store = SelfModelStore::new();
        let props = r.propose_revisions(&belief, &store);
        // Expect at least one proposal, and ideally two (one per cluster).
        assert!(!props.is_empty());
        assert!(props.iter().all(|p| p.evidence_count >= 3));
    }

    #[test]
    fn apply_blocks_low_novelty() {
        let mut belief = SelfBelief::new();
        let mut store = SelfModelStore::new();
        let r = SchemaReviser::new();
        let p = SchemaProposal {
            kind: ProposalKind::AddDimension,
            name_or_text: "tool_use".to_string(),
            tags: vec![],
            rationale: "test".to_string(),
            evidence_count: 5,
            novelty: 0.1, // below threshold
        };
        assert!(!r.apply(&p, &mut belief, &mut store));
    }

    #[test]
    fn apply_adds_dimension_when_novel() {
        let mut belief = SelfBelief::new();
        let mut store = SelfModelStore::new();
        let r = SchemaReviser::new();
        let p = SchemaProposal {
            kind: ProposalKind::AddDimension,
            name_or_text: "sql_migration".to_string(),
            tags: vec![],
            rationale: "test".to_string(),
            evidence_count: 5,
            novelty: 0.9,
        };
        assert!(r.apply(&p, &mut belief, &mut store));
        assert!(belief.has_dimension("sql_migration"));
    }
}
