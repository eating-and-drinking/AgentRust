//! Layer 3 atom — `SelfProposition`.
//!
//! A natural-language claim the agent has accumulated about itself
//! ("I tend to forget noexcept on move constructors"). Carries confidence,
//! evidence count, tags, and timestamps so the store can rank and decay
//! them.
//!
//! IDs are derived deterministically from `text + tags` via FNV-1a so that
//! repeated observations of the same claim collapse to one row.

use chrono::{DateTime, TimeZone, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfProposition {
    pub id: String,
    pub text: String,
    pub tags: Vec<String>,
    pub confidence: f64,
    pub evidence_count: i32,
    /// Epoch milliseconds (matches AgentCpp's on-disk format).
    pub created_at_ms: i64,
    pub updated_at_ms: i64,
}

impl SelfProposition {
    /// Build a fresh proposition. The id is computed from text+tags.
    pub fn new(text: impl Into<String>, tags: Vec<String>) -> Self {
        let text = text.into();
        let id = make_proposition_id(&text, &tags);
        let now = Utc::now().timestamp_millis();
        Self {
            id,
            text,
            tags,
            confidence: 0.5,
            evidence_count: 1,
            created_at_ms: now,
            updated_at_ms: now,
        }
    }

    pub fn with_confidence(mut self, c: f64) -> Self {
        self.confidence = c.clamp(0.0, 1.0);
        self
    }

    pub fn with_evidence_count(mut self, n: i32) -> Self {
        self.evidence_count = n.max(0);
        self
    }

    pub fn created_at(&self) -> DateTime<Utc> {
        Utc.timestamp_millis_opt(self.created_at_ms).single().unwrap_or_else(Utc::now)
    }

    pub fn updated_at(&self) -> DateTime<Utc> {
        Utc.timestamp_millis_opt(self.updated_at_ms).single().unwrap_or_else(Utc::now)
    }

    pub fn touch_now(&mut self) {
        self.updated_at_ms = Utc::now().timestamp_millis();
    }
}

/// FNV-1a 64-bit hash over `text` and each tag, exactly as
/// `agent/SelfModelStore.cpp::makePropositionId`. Produces a stable id that
/// repeats for the same logical claim.
pub fn make_proposition_id(text: &str, tags: &[String]) -> String {
    let mut h: u64 = 0x14650FB0739D0383; // FNV offset basis (64-bit)
    let prime: u64 = 0x100000001B3;
    for &b in text.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(prime);
    }
    for tag in tags {
        for &b in b"|" {
            h ^= b as u64;
            h = h.wrapping_mul(prime);
        }
        for &b in tag.as_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(prime);
        }
    }
    format!("sp-{:016x}", h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_is_stable_for_same_text_and_tags() {
        let a = make_proposition_id("foo", &["x".to_string(), "y".to_string()]);
        let b = make_proposition_id("foo", &["x".to_string(), "y".to_string()]);
        assert_eq!(a, b);
    }

    #[test]
    fn id_differs_on_tag_change() {
        let a = make_proposition_id("foo", &["x".to_string()]);
        let b = make_proposition_id("foo", &["y".to_string()]);
        assert_ne!(a, b);
    }

    #[test]
    fn id_differs_on_text_change() {
        let a = make_proposition_id("foo", &[]);
        let b = make_proposition_id("bar", &[]);
        assert_ne!(a, b);
    }
}
