//! Reciprocal Rank Fusion (RRF).
//!
//! Merges multiple ranked candidate lists into a single ranked list by
//! summing `1 / (k + rank_i)` across all lists. `k = 60` is the value used
//! by Hindsight and the AgentCpp port.
//!
//! Use this when you have several retrieval channels (semantic / BM25 /
//! graph / temporal) and want one merged ordering without committing to a
//! cross-encoder reranker.

use std::collections::HashMap;

/// Default Reciprocal Rank Fusion constant.
pub const DEFAULT_RRF_K: f32 = 60.0;

/// Merge several ranked id lists into one. Each input list is in
/// best-first order; ties between equal positions in different lists are
/// broken by total accumulated score.
///
/// Returns `(id, fused_score)` pairs sorted descending.
pub fn reciprocal_rank_fusion(lists: &[&[String]], k: f32) -> Vec<(String, f32)> {
    let mut score: HashMap<String, f32> = HashMap::new();
    for list in lists {
        for (rank, id) in list.iter().enumerate() {
            let s = 1.0 / (k + (rank + 1) as f32);
            *score.entry(id.clone()).or_insert(0.0) += s;
        }
    }
    let mut out: Vec<(String, f32)> = score.into_iter().collect();
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// Min-max normalise a slice of scores into `[0, 1]`. If `max == min` all
/// outputs are `0.5` (the AgentCpp `normalizeOnDelta` convention).
pub fn normalize_on_delta(values: &mut [f32]) {
    if values.is_empty() {
        return;
    }
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for &v in values.iter() {
        if v < lo {
            lo = v;
        }
        if v > hi {
            hi = v;
        }
    }
    let span = hi - lo;
    if span.abs() < 1e-9 {
        for v in values.iter_mut() {
            *v = 0.5;
        }
        return;
    }
    for v in values.iter_mut() {
        *v = ((*v - lo) / span).clamp(0.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fusion_prefers_top_of_every_list() {
        let l1: Vec<String> = vec!["a".into(), "b".into(), "c".into()];
        let l2: Vec<String> = vec!["a".into(), "x".into(), "y".into()];
        let merged = reciprocal_rank_fusion(&[&l1, &l2], DEFAULT_RRF_K);
        assert_eq!(merged[0].0, "a");
    }

    #[test]
    fn normalize_collapses_when_constant() {
        let mut v = vec![3.0_f32, 3.0, 3.0];
        normalize_on_delta(&mut v);
        assert!(v.iter().all(|&x| (x - 0.5).abs() < 1e-6));
    }

    #[test]
    fn normalize_extremes_become_0_and_1() {
        let mut v = vec![1.0_f32, 2.0, 3.0];
        normalize_on_delta(&mut v);
        assert!((v[0] - 0.0).abs() < 1e-6);
        assert!((v[2] - 1.0).abs() < 1e-6);
    }
}
