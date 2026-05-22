//! API Module - OpenAI/DeepSeek compatible API Client

use crate::config::Settings;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Clone)]
pub struct ApiClient {
    settings: Settings,
    http_client: std::sync::Arc<Client>,
}

impl ApiClient {
    pub fn new(settings: Settings) -> Self {
        let http_client = Client::builder()
            .timeout(Duration::from_secs(settings.api.timeout))
            .build()
            .unwrap_or_default();

        Self {
            settings,
            http_client: std::sync::Arc::new(http_client),
        }
    }

    pub fn get_api_key(&self) -> Option<String> {
        self.settings.api.get_api_key()
    }

    pub fn get_base_url(&self) -> String {
        self.settings.api.get_base_url()
    }

    pub fn get_model(&self) -> &str {
        &self.settings.model
    }

    pub async fn chat(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<ToolDefinition>>,
    ) -> anyhow::Result<ChatResponse> {
        let api_key = self
            .get_api_key()
            .ok_or_else(|| anyhow::anyhow!("API key not configured"))?;

        let request = ChatRequest {
            model: self.settings.api.get_model_id(&self.settings.model),
            messages,
            max_tokens: self.settings.api.max_tokens,
            stream: false,
            temperature: 0.7,
            tools,
        };

        let url = format!("{}/v1/chat/completions", self.get_base_url());

        let response = self
            .http_client
            .post(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!("API error ({}): {}", status, body));
        }

        let chat_response: ChatResponse = response.json().await?;
        Ok(chat_response)
    }

    pub async fn chat_stream(
        &self,
        messages: Vec<ChatMessage>,
        tools: Option<Vec<ToolDefinition>>,
    ) -> anyhow::Result<reqwest::Response> {
        let api_key = self
            .get_api_key()
            .ok_or_else(|| anyhow::anyhow!("API key not configured"))?;

        let request = ChatRequest {
            model: self.settings.api.get_model_id(&self.settings.model),
            messages,
            max_tokens: self.settings.api.max_tokens,
            stream: true,
            temperature: 0.7,
            tools,
        };

        let url = format!("{}/v1/chat/completions", self.get_base_url());

        let response = self
            .http_client
            .post(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        Ok(response)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub r#type: String,
    pub function: ToolFunction,
}

impl ToolDefinition {
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: serde_json::Value) -> Self {
        Self {
            r#type: "function".to_string(),
            function: ToolFunction {
                name: name.into(),
                description: description.into(),
                parameters,
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunction {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub r#type: String,
    pub function: ToolCallFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: String,
}

/// One inline image attachment for a multimodal chat message.
///
/// Maps to the OpenAI / Anthropic vision payload —
/// `{"type": "image_url", "image_url": {"url": "..."}}` — when a
/// message containing it is serialised.
#[derive(Debug, Clone, Deserialize)]
pub struct ImageRef {
    /// `http(s)://` URL or `data:image/<kind>;base64,...` URI.
    pub url: String,
    /// Optional `low` / `high` detail hint.
    pub detail: Option<String>,
}

impl ImageRef {
    pub fn from_url(url: impl Into<String>) -> Self {
        Self { url: url.into(), detail: None }
    }

    pub fn from_data_uri(uri: impl Into<String>) -> Self {
        Self { url: uri.into(), detail: None }
    }

    /// Read a local file and wrap it in a `data:` URI. Falls back to
    /// `image/png` if the extension is unrecognised.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        let p = path.as_ref();
        let bytes = std::fs::read(p)?;
        let mime = match p.extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase()).as_deref() {
            Some("png") => "image/png",
            Some("jpg") | Some("jpeg") => "image/jpeg",
            Some("gif") => "image/gif",
            Some("webp") => "image/webp",
            _ => "image/png",
        };
        use base64::Engine as _;
        let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
        Ok(Self {
            url: format!("data:{};base64,{}", mime, b64),
            detail: None,
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub tool_call_id: Option<String>,
    /// Optional inline image attachments. When non-empty, the message
    /// is serialised in OpenAI multimodal "parts" form
    /// (`content: [{type:text,...}, {type:image_url,...}]`); otherwise
    /// it serialises as a plain `content: "..."`.
    ///
    /// Deserialised responses never set this — incoming model output
    /// goes through `content: Option<String>`.
    #[serde(skip, default)]
    pub images: Vec<ImageRef>,
}

impl ChatMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
            images: Vec::new(),
        }
    }

    /// Multimodal user message — text plus one or more images.
    pub fn user_with_images(content: impl Into<String>, images: Vec<ImageRef>) -> Self {
        Self {
            role: "user".to_string(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
            images,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
            images: Vec::new(),
        }
    }

    pub fn assistant_with_tools(tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: None,
            tool_calls: Some(tool_calls),
            tool_call_id: None,
            images: Vec::new(),
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
            images: Vec::new(),
        }
    }

    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: "tool".to_string(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
            images: Vec::new(),
        }
    }

    /// Attach an image to an existing message in builder style.
    pub fn with_image(mut self, image: ImageRef) -> Self {
        self.images.push(image);
        self
    }
}

// Hand-rolled Serialize: emits the OpenAI multimodal "parts" form when
// `images` is non-empty, otherwise the simple `content: "..."` form. We
// can't use serde's derive here because the JSON shape depends on a
// sibling field.
impl Serialize for ChatMessage {
    fn serialize<S: serde::Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut field_count = 1; // role
        if !self.images.is_empty() || self.content.is_some() { field_count += 1; }
        if self.tool_calls.is_some() { field_count += 1; }
        if self.tool_call_id.is_some() { field_count += 1; }

        let mut map = ser.serialize_map(Some(field_count))?;
        map.serialize_entry("role", &self.role)?;

        if self.images.is_empty() {
            if let Some(c) = &self.content {
                map.serialize_entry("content", c)?;
            }
        } else {
            // Multimodal parts array. We always include the text part
            // (even if empty) for compatibility — most vision backends
            // require it.
            let mut parts: Vec<serde_json::Value> = Vec::with_capacity(self.images.len() + 1);
            let text = self.content.clone().unwrap_or_default();
            parts.push(serde_json::json!({"type": "text", "text": text}));
            for img in &self.images {
                let mut url_obj = serde_json::json!({"url": img.url});
                if let Some(d) = &img.detail {
                    url_obj["detail"] = serde_json::Value::String(d.clone());
                }
                parts.push(serde_json::json!({
                    "type": "image_url",
                    "image_url": url_obj,
                }));
            }
            map.serialize_entry("content", &parts)?;
        }

        if let Some(tcs) = &self.tool_calls {
            map.serialize_entry("tool_calls", tcs)?;
        }
        if let Some(id) = &self.tool_call_id {
            map.serialize_entry("tool_call_id", id)?;
        }
        map.end()
    }
}

#[derive(Debug, Clone, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    max_tokens: usize,
    stream: bool,
    temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ToolDefinition>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Choice {
    pub index: i32,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StreamChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<StreamChoice>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StreamChoice {
    pub index: i32,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Delta {
    pub role: Option<String>,
    pub content: Option<String>,
}

pub type AnthropicClient = ApiClient;
