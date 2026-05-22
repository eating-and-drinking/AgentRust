//! HTTP fetch tool — minimal native web access without going through
//! `execute_command` + `curl`.
//!
//! Two tools live here:
//! - [`HttpFetchTool`] (`http_fetch`): GET a URL, return body + status +
//!   selected headers. Optional `max_bytes` cap.
//! - [`WebSearchTool`] (`web_search`): a thin wrapper around
//!   DuckDuckGo's HTML endpoint. Returns the top N hits' titles + URLs.
//!   This is intentionally a low-fidelity baseline — real production
//!   use should swap in a proper search provider via configuration.

use std::collections::HashMap;
use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;
use serde_json::{json, Value};

use super::{Tool, ToolError, ToolOutput};

/// Plain HTTP GET tool.
pub struct HttpFetchTool {
    client: Client,
}

impl Default for HttpFetchTool {
    fn default() -> Self {
        Self::new()
    }
}

impl HttpFetchTool {
    pub fn new() -> Self {
        let client = Client::builder()
            .user_agent("agentrust/0.1 (+https://github.com/agentrust)")
            .timeout(Duration::from_secs(20))
            .build()
            .unwrap_or_default();
        Self { client }
    }
}

#[async_trait]
impl Tool for HttpFetchTool {
    fn name(&self) -> &str {
        "http_fetch"
    }

    fn description(&self) -> &str {
        "Fetch an HTTP(S) URL via GET and return the response body. \
         Useful for reading web pages, JSON APIs, or downloading small \
         files. Capped at `max_bytes` (default 200_000) to avoid \
         flooding the agent's context."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Absolute http(s) URL to fetch."
                },
                "max_bytes": {
                    "type": "integer",
                    "description": "Truncate the body to this many bytes. Default 200000.",
                    "default": 200000
                },
                "headers": {
                    "type": "object",
                    "description": "Optional request headers as a flat string→string map.",
                    "additionalProperties": {"type": "string"}
                }
            },
            "required": ["url"]
        })
    }

    async fn execute(&self, input: Value) -> Result<ToolOutput, ToolError> {
        let url = input["url"].as_str().ok_or_else(|| ToolError {
            message: "url is required".to_string(),
            code: Some("missing_parameter".into()),
        })?;
        if !(url.starts_with("http://") || url.starts_with("https://")) {
            return Err(ToolError {
                message: "url must start with http:// or https://".into(),
                code: Some("invalid_url".into()),
            });
        }
        let max_bytes = input["max_bytes"].as_u64().unwrap_or(200_000) as usize;

        let mut req = self.client.get(url);
        if let Some(hdrs) = input.get("headers").and_then(|v| v.as_object()) {
            for (k, v) in hdrs {
                if let Some(s) = v.as_str() {
                    req = req.header(k, s);
                }
            }
        }

        let resp = req.send().await.map_err(|e| ToolError {
            message: format!("request failed: {}", e),
            code: Some("network_error".into()),
        })?;

        let status = resp.status().as_u16();
        let final_url = resp.url().to_string();
        let content_type = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();

        let full_body = resp.bytes().await.map_err(|e| ToolError {
            message: format!("read body failed: {}", e),
            code: Some("network_error".into()),
        })?;

        let truncated = full_body.len() > max_bytes;
        let slice = &full_body[..full_body.len().min(max_bytes)];
        let body = String::from_utf8_lossy(slice).to_string();

        let mut metadata: HashMap<String, Value> = HashMap::new();
        metadata.insert("status".into(), json!(status));
        metadata.insert("url".into(), json!(final_url));
        metadata.insert("content_type".into(), json!(content_type));
        metadata.insert("bytes".into(), json!(full_body.len()));
        metadata.insert("truncated".into(), json!(truncated));

        Ok(ToolOutput {
            output_type: "text".into(),
            content: body,
            metadata,
        })
    }
}

/// Minimal web search — uses DuckDuckGo's HTML endpoint and scrapes
/// titles + URLs with a simple regex. Best-effort, no third-party API
/// key required. Swap in a real provider for production use.
pub struct WebSearchTool {
    client: Client,
}

impl Default for WebSearchTool {
    fn default() -> Self {
        Self::new()
    }
}

impl WebSearchTool {
    pub fn new() -> Self {
        let client = Client::builder()
            .user_agent("Mozilla/5.0 (compatible; agentrust/0.1)")
            .timeout(Duration::from_secs(20))
            .build()
            .unwrap_or_default();
        Self { client }
    }
}

#[async_trait]
impl Tool for WebSearchTool {
    fn name(&self) -> &str {
        "web_search"
    }

    fn description(&self) -> &str {
        "Search the web (DuckDuckGo HTML endpoint) and return the top \
         N hits' titles and URLs. Low-fidelity baseline — swap in a \
         proper provider for serious work."
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query":   {"type": "string", "description": "Free-text search query."},
                "limit":   {"type": "integer", "description": "Max hits to return (default 5, max 20).", "default": 5}
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, input: Value) -> Result<ToolOutput, ToolError> {
        let query = input["query"].as_str().ok_or_else(|| ToolError {
            message: "query is required".to_string(),
            code: Some("missing_parameter".into()),
        })?;
        let limit = input["limit"].as_u64().unwrap_or(5).clamp(1, 20) as usize;

        let resp = self
            .client
            .get("https://html.duckduckgo.com/html/")
            .query(&[("q", query)])
            .send()
            .await
            .map_err(|e| ToolError {
                message: format!("search request failed: {}", e),
                code: Some("network_error".into()),
            })?;

        let html = resp.text().await.map_err(|e| ToolError {
            message: format!("read body failed: {}", e),
            code: Some("network_error".into()),
        })?;

        // Best-effort scraping — DDG's HTML markup is stable enough for
        // a cheap regex pass. We don't pull in a full HTML parser here
        // to keep the dependency surface tight.
        let re = regex::Regex::new(
            r#"(?s)<a\s+[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>"#,
        )
        .map_err(|e| ToolError {
            message: format!("bad regex: {}", e),
            code: Some("internal_error".into()),
        })?;

        let mut hits: Vec<Value> = Vec::new();
        for cap in re.captures_iter(&html).take(limit) {
            let url = cap.get(1).map(|m| m.as_str().to_string()).unwrap_or_default();
            let title_html = cap.get(2).map(|m| m.as_str()).unwrap_or("");
            // Strip tags / collapse whitespace.
            let tag_re = regex::Regex::new(r"<[^>]+>").unwrap();
            let title = tag_re.replace_all(title_html, "").trim().to_string();
            hits.push(json!({"title": title, "url": url}));
        }

        let mut metadata: HashMap<String, Value> = HashMap::new();
        metadata.insert("query".into(), json!(query));
        metadata.insert("hits".into(), json!(hits.len()));

        Ok(ToolOutput {
            output_type: "json".into(),
            content: serde_json::to_string_pretty(&json!({"results": hits}))
                .unwrap_or_else(|_| "{}".into()),
            metadata,
        })
    }
}
