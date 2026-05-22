//! Multi-hop graph retrieval via spreading activation.
//!
//! The default recall path in [`super::recall::RecallEngine::recall`] does
//! a single hop expansion from semantic seeds. AgentCpp's
//! `GraphRetrieval::expand` walks the link graph for several hops with a
//! per-hop decay, so distant-but-strongly-connected units can still
//! surface.
//!
//! This module is a port of that algorithm. It takes a `seed_id -> seed
//! score` map plus a closure that resolves each unit id to its neighbours
//! (already weighted in `[0, 1]`), so it can run against either the SQLite
//! `Storage::neighbors` or an in-memory adjacency list.

use std::collections::HashMap;

/// One spreading-activation hit, identified by the unit id.
#[derive(Debug, Clone)]
pub struct GraphHit {
    pub unit_id: String,
    pub activation: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct GraphConfig {
    /// Multiplier applied at every hop. `0.5` halves activation per step
    /// (the AgentCpp default).
    pub hop_decay: f32,
    /// Maximum BFS depth from the seed set.
    pub max_hops: u32,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            hop_decay: 0.5,
            max_hops: 2,
        }
    }
}

/// Spread activation from `seeds` outward through the link graph. `neighbours`
/// must return `(neighbour_id, edge_weight)` pairs for a given unit id.
///
/// Returns top-`k` non-seed hits sorted descending by activation. Best-hit
/// per unit id is kept if a node is reached via multiple paths.
pub fn expand<F>(
    seeds: &[(String, f32)],
    cfg: GraphConfig,
    k: usize,
    mut neighbours: F,
) -> Vec<GraphHit>
where
    F: FnMut(&str) -> Vec<(String, f32)>,
{
    if seeds.is_empty() || k == 0 {
        return Vec::new();
    }
    let mut activation: HashMap<String, f32> = HashMap::new();
    let mut frontier: Vec<(String, u32, f32)> = Vec::new();
    let mut seed_ids: std::collections::HashSet<String> = std::collections::HashSet::new();

    for (id, score) in seeds {
        activation.insert(id.clone(), *score);
        seed_ids.insert(id.clone());
        frontier.push((id.clone(), 0, *score));
    }

    let mut next_frontier: Vec<(String, u32, f32)> = Vec::new();
    while !frontier.is_empty() {
        for (uid, hop, score) in frontier.drain(..) {
            if hop >= cfg.max_hops {
                continue;
            }
            for (nb, w) in neighbours(&uid) {
                let new_score = score * cfg.hop_decay * w;
                if new_score <= 0.0 {
                    continue;
                }
                let entry = activation.entry(nb.clone()).or_insert(0.0);
                if new_score > *entry {
                    *entry = new_score;
                    next_frontier.push((nb, hop + 1, new_score));
                }
            }
        }
        std::mem::swap(&mut frontier, &mut next_frontier);
    }

    let mut hits: Vec<GraphHit> = activation
        .into_iter()
        .filter(|(id, _)| !seed_ids.contains(id))
        .map(|(unit_id, activation)| GraphHit { unit_id, activation })
        .collect();
    hits.sort_by(|a, b| b.activation.partial_cmp(&a.activation).unwrap_or(std::cmp::Ordering::Equal));
    hits.truncate(k);
    hits
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn no_seeds_returns_empty() {
        let cfg = GraphConfig::default();
        let hits = expand(&[], cfg, 10, |_| Vec::new());
        assert!(hits.is_empty());
    }

    #[test]
    fn spreads_two_hops_with_decay() {
        // a -1.0-> b -1.0-> c -1.0-> d
        // With hop_decay=0.5 and max_hops=2, c should appear (depth 2),
        // d should not.
        let mut adj: HashMap<String, Vec<(String, f32)>> = HashMap::new();
        adj.insert("a".into(), vec![("b".into(), 1.0)]);
        adj.insert("b".into(), vec![("c".into(), 1.0)]);
        adj.insert("c".into(), vec![("d".into(), 1.0)]);
        let cfg = GraphConfig::default();
        let hits = expand(&[("a".into(), 1.0)], cfg, 10, |id| {
            adj.get(id).cloned().unwrap_or_default()
        });
        let ids: Vec<&str> = hits.iter().map(|h| h.unit_id.as_str()).collect();
        assert!(ids.contains(&"b"));
        assert!(ids.contains(&"c"));
        assert!(!ids.contains(&"d"));
    }

    #[test]
    fn keeps_best_path_when_multiple_paths_exist() {
        // a -1.0-> c
        // a -1.0-> b -0.1-> c
        // best path is direct a→c (0.5), not a→b→c (0.025).
        let mut adj: HashMap<String, Vec<(String, f32)>> = HashMap::new();
        adj.insert("a".into(), vec![("b".into(), 1.0), ("c".into(), 1.0)]);
        adj.insert("b".into(), vec![("c".into(), 0.1)]);
        let cfg = GraphConfig {
            hop_decay: 0.5,
            max_hops: 3,
        };
        let hits = expand(&[("a".into(), 1.0)], cfg, 10, |id| {
            adj.get(id).cloned().unwrap_or_default()
        });
        let c_hit = hits.iter().find(|h| h.unit_id == "c").expect("c reached");
        assert!((c_hit.activation - 0.5).abs() < 1e-6);
    }
}
