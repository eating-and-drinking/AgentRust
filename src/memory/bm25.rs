//! In-memory BM25 index.
//!
//! AgentRust's primary BM25 path goes through SQLite FTS5 in
//! [`crate::memory::storage::Storage::bm25_search`]. This module ports
//! AgentCpp's pure-Rust BM25 implementation as a smaller, allocation-only
//! fallback — useful for ranking ad-hoc unit collections (e.g. the Layer-3
//! self-model bank) without round-tripping through SQLite.

use std::collections::HashMap;

use super::model::MemoryUnit;

/// Tunables for `BM25Index`. Defaults are the canonical `k1 = 1.5, b = 0.75`.
#[derive(Debug, Clone, Copy)]
pub struct BM25Config {
    pub k1: f32,
    pub b: f32,
}

impl Default for BM25Config {
    fn default() -> Self {
        Self { k1: 1.5, b: 0.75 }
    }
}

/// One ranked hit. The unit is cloned on output so callers don't have to
/// keep the index alive while consuming results.
#[derive(Debug, Clone)]
pub struct BM25Hit {
    pub unit: MemoryUnit,
    pub score: f32,
}

#[derive(Debug, Default)]
pub struct BM25Index {
    cfg: BM25Config,
    units: Vec<MemoryUnit>,
    doc_terms: Vec<HashMap<String, u32>>,
    doc_len: Vec<u32>,
    inverted: HashMap<String, Vec<usize>>,
    doc_freq: HashMap<String, u32>,
    avg_len: f32,
}

impl BM25Index {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_config(cfg: BM25Config) -> Self {
        Self {
            cfg,
            ..Self::default()
        }
    }

    pub fn is_empty(&self) -> bool {
        self.units.is_empty()
    }

    pub fn len(&self) -> usize {
        self.units.len()
    }

    /// (Re)build the index from a slice of memory units. Tokenizes both
    /// `text` and `context` for each unit.
    pub fn build(&mut self, units: &[MemoryUnit]) {
        self.units.clear();
        self.doc_terms.clear();
        self.doc_len.clear();
        self.inverted.clear();
        self.doc_freq.clear();

        let mut total_len: u64 = 0;
        for u in units {
            let body = match &u.context {
                Some(c) => format!("{} {}", u.text, c),
                None => u.text.clone(),
            };
            let toks = tokenize(&body);
            let mut tf: HashMap<String, u32> = HashMap::new();
            for t in &toks {
                *tf.entry(t.clone()).or_insert(0) += 1;
            }
            let doc_idx = self.units.len();
            for term in tf.keys() {
                self.inverted
                    .entry(term.clone())
                    .or_default()
                    .push(doc_idx);
                *self.doc_freq.entry(term.clone()).or_insert(0) += 1;
            }
            total_len += toks.len() as u64;
            self.doc_len.push(toks.len() as u32);
            self.doc_terms.push(tf);
            self.units.push(u.clone());
        }
        self.avg_len = if self.units.is_empty() {
            0.0
        } else {
            total_len as f32 / self.units.len() as f32
        };
    }

    /// Top-`k` Okapi BM25 hits for the query string.
    pub fn search(&self, query: &str, k: usize) -> Vec<BM25Hit> {
        if self.units.is_empty() || k == 0 {
            return Vec::new();
        }
        let q_tokens = tokenize(query);
        if q_tokens.is_empty() {
            return Vec::new();
        }
        let mut q_tf: HashMap<String, u32> = HashMap::new();
        for t in &q_tokens {
            *q_tf.entry(t.clone()).or_insert(0) += 1;
        }
        let n = self.units.len() as f32;
        let mut scores: HashMap<usize, f32> = HashMap::new();
        for (term, qtf) in &q_tf {
            let postings = match self.inverted.get(term) {
                Some(p) => p,
                None => continue,
            };
            let df = *self.doc_freq.get(term).unwrap_or(&0) as f32;
            let idf = (1.0_f32 + (n - df + 0.5) / (df + 0.5)).ln();
            for &doc_idx in postings {
                let tf = *self.doc_terms[doc_idx].get(term).unwrap_or(&0) as f32;
                let dl = self.doc_len[doc_idx] as f32;
                let denom = tf
                    + self.cfg.k1
                        * (1.0 - self.cfg.b
                            + self.cfg.b * (dl / self.avg_len.max(1.0)));
                let term_score = idf * ((tf * (self.cfg.k1 + 1.0)) / denom.max(1e-6))
                    * (*qtf as f32);
                *scores.entry(doc_idx).or_insert(0.0) += term_score;
            }
        }
        let mut ranked: Vec<(usize, f32)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(k);
        ranked
            .into_iter()
            .map(|(i, s)| BM25Hit {
                unit: self.units[i].clone(),
                score: s,
            })
            .collect()
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
            if buf.chars().count() >= 2 {
                out.push(std::mem::take(&mut buf));
            } else {
                buf.clear();
            }
        }
    }
    if buf.chars().count() >= 2 {
        out.push(buf);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::model::FactType;

    fn u(text: &str) -> MemoryUnit {
        MemoryUnit::new("test", text, FactType::Experience)
    }

    #[test]
    fn empty_index_returns_empty() {
        let idx = BM25Index::new();
        assert!(idx.search("anything", 5).is_empty());
    }

    #[test]
    fn ranks_more_relevant_higher() {
        let mut idx = BM25Index::new();
        idx.build(&[
            u("apple banana cherry"),
            u("banana cherry date"),
            u("zebra cherry"),
        ]);
        let hits = idx.search("apple banana", 3);
        assert!(!hits.is_empty());
        assert!(hits[0].unit.text.starts_with("apple"));
    }

    #[test]
    fn missing_terms_return_zero_hits() {
        let mut idx = BM25Index::new();
        idx.build(&[u("apple")]);
        assert!(idx.search("xenon", 5).is_empty());
    }
}
