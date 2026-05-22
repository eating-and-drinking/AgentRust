//! Layer 3 (structured self-knowledge) — `SelfModelStore`.
//!
//! Bag of `SelfProposition`s with Jaccard-relevance retrieval and a
//! markdown-rendered system-prompt section.
//!
//! Persistence is delegated to caller-supplied closures so the store stays
//! independent of any particular memory backend. `SelfModelMemoryAdapter`
//! wires those closures to `MemoryManager`.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;

use super::self_proposition::{make_proposition_id, SelfProposition};

/// Boxed callback that writes the current proposition set to backing
/// storage. Implementations should be fast and side-effect-only.
pub type PersistFn = Arc<dyn Fn(&[SelfProposition]) + Send + Sync>;

/// Boxed callback that reads the current proposition set from backing
/// storage. Returns `Vec::new()` on cold start.
pub type LoadFn = Arc<dyn Fn() -> Vec<SelfProposition> + Send + Sync>;

#[derive(Clone, Default, Serialize, Deserialize)]
pub struct SelfModelStore {
    props: Vec<SelfProposition>,
    #[serde(skip)]
    persist_fn: Option<PersistFn>,
    #[serde(skip)]
    load_fn: Option<LoadFn>,
}

impl std::fmt::Debug for SelfModelStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SelfModelStore")
            .field("props_len", &self.props.len())
            .field("has_persist_fn", &self.persist_fn.is_some())
            .field("has_load_fn", &self.load_fn.is_some())
            .finish()
    }
}

impl SelfModelStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn propositions(&self) -> &[SelfProposition] {
        &self.props
    }

    pub fn len(&self) -> usize {
        self.props.len()
    }

    pub fn is_empty(&self) -> bool {
        self.props.is_empty()
    }

    pub fn set_persist_fn(&mut self, f: PersistFn) {
        self.persist_fn = Some(f);
    }

    pub fn set_load_fn(&mut self, f: LoadFn) {
        self.load_fn = Some(f);
    }

    pub fn load_from_external(&mut self) {
        if let Some(f) = &self.load_fn {
            self.props = f();
        }
    }

    pub fn save_to_external(&self) {
        if let Some(f) = &self.persist_fn {
            f(&self.props);
        }
    }

    /// Upsert a proposition. If an entry with the same id exists,
    /// increments `evidence_count`, nudges confidence up by 0.05 (clamped),
    /// merges tags, and bumps `updated_at_ms`.
    pub fn add_proposition(&mut self, mut p: SelfProposition) {
        if p.id.is_empty() {
            p.id = make_proposition_id(&p.text, &p.tags);
        }
        if let Some(existing) = self.props.iter_mut().find(|x| x.id == p.id) {
            existing.evidence_count += 1;
            existing.confidence = (existing.confidence + 0.05).min(1.0);
            for tag in p.tags.iter() {
                if !existing.tags.iter().any(|t| t == tag) {
                    existing.tags.push(tag.clone());
                }
            }
            existing.touch_now();
        } else {
            self.props.push(p);
        }
    }

    pub fn reinforce(&mut self, id: &str, delta: f64) {
        if let Some(p) = self.props.iter_mut().find(|x| x.id == id) {
            p.confidence = (p.confidence + delta).clamp(0.0, 1.0);
            p.touch_now();
        }
    }

    pub fn weaken(&mut self, id: &str, delta: f64) {
        if let Some(p) = self.props.iter_mut().find(|x| x.id == id) {
            p.confidence = (p.confidence - delta).clamp(0.0, 1.0);
            p.touch_now();
        }
    }

    /// Top-`k` propositions ranked by Jaccard token overlap against the
    /// task description, multiplied by `(0.5 + 0.5·confidence)`.
    pub fn retrieve_relevant(&self, task_desc: &str, k: usize) -> Vec<SelfProposition> {
        if self.props.is_empty() || k == 0 {
            return Vec::new();
        }
        let q_tokens: HashSet<String> = tokenize(task_desc).into_iter().collect();
        let mut scored: Vec<(f64, &SelfProposition)> = self
            .props
            .iter()
            .map(|p| (score_relevance(p, &q_tokens), p))
            .filter(|(s, _)| *s > 0.0)
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(k).map(|(_, p)| p.clone()).collect()
    }

    /// Render top-`k` relevant propositions as a markdown system-prompt
    /// section. Empty when no propositions match.
    pub fn render_for_prompt(&self, task_desc: &str, k: usize) -> String {
        let picks = self.retrieve_relevant(task_desc, k);
        if picks.is_empty() {
            return String::new();
        }
        let mut s = String::new();
        s.push_str("## Self-knowledge relevant to this task (Layer 3)\n\n");
        s.push_str(
            "These are propositions you have accumulated about your own tendencies. \
             Use them to set priors, not as ground truth.\n\n",
        );
        for (i, p) in picks.iter().enumerate() {
            s.push_str(&format!(
                "{}. {}  (confidence={:.2}, seen={})\n",
                i + 1,
                p.text,
                p.confidence,
                p.evidence_count
            ));
        }
        s
    }

    /// Drop everything. Does not touch the external callbacks.
    pub fn clear(&mut self) {
        self.props.clear();
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
            if buf.chars().count() >= 3 {
                out.push(std::mem::take(&mut buf));
            } else {
                buf.clear();
            }
        }
    }
    if buf.chars().count() >= 3 {
        out.push(buf);
    }
    out
}

fn score_relevance(prop: &SelfProposition, query_set: &HashSet<String>) -> f64 {
    let mut prop_tokens: HashSet<String> = tokenize(&prop.text).into_iter().collect();
    for tag in &prop.tags {
        for t in tokenize(tag) {
            prop_tokens.insert(t);
        }
    }
    if prop_tokens.is_empty() || query_set.is_empty() {
        return 0.0;
    }
    let inter = prop_tokens.intersection(query_set).count() as f64;
    let union = prop_tokens.union(query_set).count() as f64;
    if union == 0.0 {
        return 0.0;
    }
    let jaccard = inter / union;
    jaccard * (0.5 + 0.5 * prop.confidence)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_proposition_is_idempotent_on_same_text() {
        let mut s = SelfModelStore::new();
        s.add_proposition(SelfProposition::new("I forget noexcept on moves", vec!["c++".into()]));
        s.add_proposition(SelfProposition::new("I forget noexcept on moves", vec!["c++".into()]));
        assert_eq!(s.len(), 1);
        assert_eq!(s.propositions()[0].evidence_count, 2);
    }

    #[test]
    fn retrieve_picks_most_relevant() {
        let mut s = SelfModelStore::new();
        s.add_proposition(SelfProposition::new(
            "I tend to forget noexcept on c++ move constructors",
            vec!["c++".into()],
        ));
        s.add_proposition(SelfProposition::new(
            "I sometimes choose wrong python virtualenv",
            vec!["python".into()],
        ));
        let r = s.retrieve_relevant("review my c++ move constructor code", 1);
        assert_eq!(r.len(), 1);
        assert!(r[0].text.contains("c++"));
    }

    #[test]
    fn render_includes_confidence() {
        let mut s = SelfModelStore::new();
        s.add_proposition(SelfProposition::new("I am terse", vec!["style".into()]).with_confidence(0.9));
        let out = s.render_for_prompt("write a terse summary", 3);
        assert!(out.contains("confidence=0.90") || out.contains("0.90"));
    }

    #[test]
    fn empty_render_is_empty_string() {
        let s = SelfModelStore::new();
        let out = s.render_for_prompt("anything", 5);
        assert!(out.is_empty());
    }
}
