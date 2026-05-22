//! Embedding layer.
//!
//! Mirrors Hindsight's `engine/embeddings.py:Embeddings` abstraction with a
//! single async trait. Two implementations ship:
//!
//! * [`HashEmbedder`] — deterministic, network-free, low-fidelity. Default
//!   for environments that don't have an embedding API configured. It's
//!   feature-hashing (trigram bag) projected onto a fixed-dim vector.
//!   Quality is enough to make the rest of the pipeline functional and
//!   testable without external services.
//! * [`HttpEmbedder`] — OpenAI-compatible `POST /v1/embeddings`. Works
//!   against OpenAI, Azure OpenAI, DeepSeek, Ollama, vLLM, LM Studio, etc.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

#[async_trait]
pub trait Embedder: Send + Sync {
    fn provider_name(&self) -> &'static str;
    fn dimension(&self) -> usize;
    /// Encode each input string into a vector. Length of the result equals
    /// the input length; each vector has `dimension()` elements.
    async fn encode(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>>;
}

/// Network-free embedder. Hashes trigram features into a fixed dimension
/// and l2-normalises the result. Two strings with high lexical overlap
/// produce vectors with high cosine — enough for the architecture to
/// function in offline / CI environments.
#[derive(Debug, Clone)]
pub struct HashEmbedder {
    dim: usize,
}

impl HashEmbedder {
    pub const DEFAULT_DIM: usize = 64;

    pub fn new() -> Self {
        Self {
            dim: Self::DEFAULT_DIM,
        }
    }

    pub fn with_dim(dim: usize) -> Self {
        Self { dim }
    }

    fn encode_one(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0.0f32; self.dim];
        let lower = text.to_lowercase();
        let chars: Vec<char> = lower.chars().collect();
        if chars.len() < 3 {
            // Fall back to word features for very short texts.
            for w in lower.split_whitespace() {
                let mut h = DefaultHasher::new();
                w.hash(&mut h);
                let idx = (h.finish() as usize) % self.dim;
                v[idx] += 1.0;
            }
        } else {
            for i in 0..=chars.len().saturating_sub(3) {
                let mut h = DefaultHasher::new();
                chars[i].hash(&mut h);
                chars[i + 1].hash(&mut h);
                chars[i + 2].hash(&mut h);
                let idx = (h.finish() as usize) % self.dim;
                v[idx] += 1.0;
            }
        }
        // l2-normalise so cosine sim is a straight dot product.
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }
}

impl Default for HashEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Embedder for HashEmbedder {
    fn provider_name(&self) -> &'static str {
        "hash"
    }
    fn dimension(&self) -> usize {
        self.dim
    }
    async fn encode(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.encode_one(t)).collect())
    }
}

/// OpenAI-compatible HTTP embedder.
#[derive(Clone)]
pub struct HttpEmbedder {
    http: reqwest::Client,
    endpoint: String,
    api_key: Option<String>,
    model: String,
    dim: usize,
}

impl HttpEmbedder {
    pub fn new(endpoint: impl Into<String>, model: impl Into<String>, dim: usize) -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self {
            http,
            endpoint: endpoint.into(),
            api_key: None,
            model: model.into(),
            dim,
        }
    }

    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a [String],
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedDatum>,
}

#[derive(Deserialize)]
struct EmbedDatum {
    embedding: Vec<f32>,
}

#[async_trait]
impl Embedder for HttpEmbedder {
    fn provider_name(&self) -> &'static str {
        "http"
    }
    fn dimension(&self) -> usize {
        self.dim
    }
    async fn encode(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let url = if self.endpoint.ends_with("/embeddings") {
            self.endpoint.clone()
        } else {
            format!("{}/v1/embeddings", self.endpoint.trim_end_matches('/'))
        };

        let req = EmbedRequest {
            model: &self.model,
            input: texts,
        };
        let mut builder = self.http.post(&url).json(&req);
        if let Some(k) = &self.api_key {
            builder = builder.bearer_auth(k);
        }
        let resp = builder.send().await?.error_for_status()?;
        let parsed: EmbedResponse = resp.json().await?;
        Ok(parsed.data.into_iter().map(|d| d.embedding).collect())
    }
}

/// Either-or wrapper so [`super::MemoryManager`] can hold a single
/// concrete type without dynamic dispatch in its struct definition.
pub type SharedEmbedder = Arc<dyn Embedder>;

pub fn default_embedder() -> SharedEmbedder {
    Arc::new(HashEmbedder::default())
}

/// Cosine similarity between two equal-length vectors. Returns 0.0 when
/// either vector is empty or zero-norm.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}
