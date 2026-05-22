//! End-to-end tests for the new memory retrieval helpers
//! (BM25 / RRF fusion / graph spreading / heuristic reranker).

use agentrust::memory::{
    reciprocal_rank_fusion, BM25Index, GraphConfig, HeuristicReranker, RecallHit,
    RerankerConfig,
};
use agentrust::memory::graph_retrieval::expand as graph_expand;
use agentrust::memory::model::{FactType, MemoryUnit};
use chrono::{Duration, Utc};

fn unit(text: &str) -> MemoryUnit {
    MemoryUnit::new("test", text, FactType::Experience)
}

#[test]
fn bm25_index_ranks_query_matching_text_highest() {
    let mut idx = BM25Index::new();
    idx.build(&[
        unit("the quick brown fox jumps over the lazy dog"),
        unit("rusty old fox in a cage"),
        unit("a python script for parsing logs"),
    ]);
    let hits = idx.search("fox", 3);
    assert!(!hits.is_empty());
    // The doc that mentions "fox" with more weight should come first; both
    // first two docs mention it, the BM25 length normalisation should rank
    // the shorter one higher.
    assert!(hits[0].unit.text.contains("fox"));
}

#[test]
fn rrf_fusion_merges_three_lists_consistently() {
    let semantic = vec!["a".to_string(), "b".to_string(), "c".to_string()];
    let bm25 = vec!["b".to_string(), "a".to_string(), "d".to_string()];
    let graph = vec!["a".to_string(), "d".to_string(), "e".to_string()];

    let merged = reciprocal_rank_fusion(&[&semantic, &bm25, &graph], 60.0);
    // `a` appears top-2 in two lists and top-1 in one → should win.
    assert_eq!(merged[0].0, "a");
    // `b` should be second (top-1 in BM25, top-2 in semantic).
    assert_eq!(merged[1].0, "b");
}

#[test]
fn graph_spreading_visits_two_hops_with_decay() {
    use std::collections::HashMap;
    let mut adj: HashMap<String, Vec<(String, f32)>> = HashMap::new();
    adj.insert("seed".into(), vec![("hop1".into(), 1.0)]);
    adj.insert("hop1".into(), vec![("hop2".into(), 1.0)]);
    adj.insert("hop2".into(), vec![("hop3".into(), 1.0)]);

    let hits = graph_expand(
        &[("seed".into(), 1.0)],
        GraphConfig {
            hop_decay: 0.5,
            max_hops: 2,
        },
        10,
        |id| adj.get(id).cloned().unwrap_or_default(),
    );
    let ids: Vec<&str> = hits.iter().map(|h| h.unit_id.as_str()).collect();
    assert!(ids.contains(&"hop1"));
    assert!(ids.contains(&"hop2"));
    assert!(!ids.contains(&"hop3"), "hop3 is depth 3, must be excluded");
    let hop1_act = hits.iter().find(|h| h.unit_id == "hop1").unwrap().activation;
    let hop2_act = hits.iter().find(|h| h.unit_id == "hop2").unwrap().activation;
    assert!(hop1_act > hop2_act);
}

#[test]
fn heuristic_reranker_blends_rrf_recency_temporal() {
    let now = Utc::now();
    let make = |text: &str, raw: f32, age_days: i64| {
        let mut u = unit(text);
        u.event_date = Some(now - Duration::days(age_days));
        u.occurred_start = u.event_date;
        u.mentioned_at = u.event_date.unwrap();
        RecallHit {
            unit: u,
            score: raw,
            semantic_rank: None,
            bm25_rank: None,
            graph_rank: None,
        }
    };

    // Three candidates: weak/old, medium/mid, strong/fresh.
    let mut hits = vec![
        make("weak old", 0.1, 365),
        make("medium mid", 0.5, 30),
        make("strong fresh", 0.9, 1),
    ];
    let r = HeuristicReranker::with_config(RerankerConfig::default());
    r.rerank(&mut hits, now, None);
    assert_eq!(hits[0].unit.text, "strong fresh");
    assert_eq!(hits.last().unwrap().unit.text, "weak old");
}

#[test]
fn weights_sum_invariant() {
    let c = RerankerConfig::default();
    let sum = c.w_rrf + c.w_rec + c.w_tmp;
    assert!((sum - 1.0).abs() < 1e-6);
}
