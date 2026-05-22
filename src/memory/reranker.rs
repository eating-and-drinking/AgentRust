//! Heuristic reranker.
//!
//! Port of `HeuristicReranker` from the AgentCpp tree. Replaces the
//! ad-hoc recency-times-proof boost in the existing recall pipeline with a
//! principled 3-term blend:
//!
//! ```text
//!   combined = w_rrf  · normalize(rrf)
//!            + w_rec  · exp(-Δt_recency / halflife_recency)
//!            + w_tmp  · exp(-Δt_to_query_date / halflife_temporal)
//! ```
//!
//! Defaults: `w_rrf=0.55, w_rec=0.20, w_tmp=0.25`, recency half-life 30
//! days, temporal half-life 7 days.

use chrono::{DateTime, Utc};

use super::fusion::normalize_on_delta;
use super::recall::RecallHit;

#[derive(Debug, Clone, Copy)]
pub struct RerankerConfig {
    pub w_rrf: f32,
    pub w_rec: f32,
    pub w_tmp: f32,
    /// Half-life for recency decay (seconds). 30 days by default.
    pub recency_halflife_sec: f32,
    /// Half-life for temporal-proximity decay (seconds). 7 days by default.
    pub temporal_halflife_sec: f32,
}

impl Default for RerankerConfig {
    fn default() -> Self {
        Self {
            w_rrf: 0.55,
            w_rec: 0.20,
            w_tmp: 0.25,
            recency_halflife_sec: 30.0 * 24.0 * 3600.0,
            temporal_halflife_sec: 7.0 * 24.0 * 3600.0,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct HeuristicReranker {
    cfg: RerankerConfig,
}

impl HeuristicReranker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_config(cfg: RerankerConfig) -> Self {
        Self { cfg }
    }

    pub fn config(&self) -> &RerankerConfig {
        &self.cfg
    }

    /// Score and resort `hits` in-place. `question_date` is the
    /// caller-supplied anchor for temporal proximity; `None` falls back to
    /// the neutral score `0.5`.
    pub fn rerank(
        &self,
        hits: &mut Vec<RecallHit>,
        now: DateTime<Utc>,
        question_date: Option<DateTime<Utc>>,
    ) {
        if hits.is_empty() {
            return;
        }
        // 1. Normalise the RRF scores.
        let mut rrf: Vec<f32> = hits.iter().map(|h| h.score).collect();
        normalize_on_delta(&mut rrf);

        // 2. Per-hit recency + temporal proximity.
        let mut recency: Vec<f32> = Vec::with_capacity(hits.len());
        let mut temporal: Vec<f32> = Vec::with_capacity(hits.len());
        for h in hits.iter() {
            // Recency is anchored at "now"; pick the most-specific time we
            // have for the unit.
            let unit_time = h
                .unit
                .occurred_start
                .or(h.unit.event_date)
                .unwrap_or(h.unit.mentioned_at);
            let delta_rec = (now - unit_time).num_seconds().max(0) as f32;
            recency.push((-delta_rec / self.cfg.recency_halflife_sec).exp());

            // Temporal proximity is anchored at the query's question date.
            let t = if let Some(qd) = question_date {
                let delta_tmp = (qd - unit_time).num_seconds().abs() as f32;
                (-delta_tmp / self.cfg.temporal_halflife_sec).exp()
            } else {
                0.5
            };
            temporal.push(t);
        }

        // 3. Blend.
        for (i, h) in hits.iter_mut().enumerate() {
            h.score = self.cfg.w_rrf * rrf[i]
                + self.cfg.w_rec * recency[i]
                + self.cfg.w_tmp * temporal[i];
        }

        // 4. Sort.
        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::model::{FactType, MemoryUnit};
    use chrono::Duration;

    fn hit(text: &str, age_days: i64, raw: f32) -> RecallHit {
        let mut u = MemoryUnit::new("t", text, FactType::Experience);
        u.event_date = Some(Utc::now() - Duration::days(age_days));
        u.occurred_start = u.event_date;
        u.mentioned_at = u.event_date.unwrap();
        RecallHit {
            unit: u,
            score: raw,
            semantic_rank: None,
            bm25_rank: None,
            graph_rank: None,
        }
    }

    #[test]
    fn fresh_hit_beats_stale_hit_when_rrf_is_tied() {
        let mut hits = vec![hit("old", 365, 1.0), hit("fresh", 1, 1.0)];
        let r = HeuristicReranker::new();
        r.rerank(&mut hits, Utc::now(), None);
        assert_eq!(hits[0].unit.text, "fresh");
    }

    #[test]
    fn higher_rrf_wins_when_age_matches() {
        let mut hits = vec![hit("a", 5, 0.1), hit("b", 5, 0.9)];
        let r = HeuristicReranker::new();
        r.rerank(&mut hits, Utc::now(), None);
        assert_eq!(hits[0].unit.text, "b");
    }

    #[test]
    fn weights_sum_to_one_by_default() {
        let c = RerankerConfig::default();
        assert!(((c.w_rrf + c.w_rec + c.w_tmp) - 1.0).abs() < 1e-6);
    }
}
