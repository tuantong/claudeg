// ─── imports ───
use serde::{Deserialize, Serialize};
use serde_json::Value;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use axum::routing::{get, post};
use axum::{Router, extract::State};
use std::collections::HashMap;
use std::sync::Arc;

// ─── types: Anthropic wire format ───

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AnthropicReq {
    pub model: String,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub system: Option<SystemField>,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub tools: Option<Vec<Tool>>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum SystemField {
    Text(String),
    Blocks(Vec<SystemBlock>),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum SystemBlock {
    #[serde(rename = "text")]
    Text { text: String },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Message {
    pub role: String, // "user" | "assistant"
    pub content: MessageContent,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    Image {
        source: ImageSource,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        #[serde(default)]
        content: Value,
        #[serde(default)]
        is_error: Option<bool>,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageSource {
    Base64 { media_type: String, data: String },
    Url { url: String },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Tool {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    pub input_schema: Value,
}

// ─── types: Codex wire format ───

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CodexReq {
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    pub input: Vec<CodexItem>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<CodexTool>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    /// ChatGPT-account flow requires `store: false` on every request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CodexItem {
    Message {
        role: String,
        content: Vec<CodexContent>,
    },
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String, // JSON-stringified
    },
    FunctionCallOutput {
        call_id: String,
        output: String, // JSON-stringified or plain text
    },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CodexContent {
    InputText { text: String },
    OutputText { text: String },
    InputImage { image_url: String },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CodexTool {
    Function {
        name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        parameters: Value,
    },
}

// Codex SSE events. Exact names depend on upstream; verified at impl time.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum CodexEvent {
    #[serde(rename = "response.created")]
    Created { response: CodexResponseMeta },
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded {
        #[serde(default)] output_index: u32,
        item: CodexOutputItem,
    },
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta {
        #[serde(default)] item_id: String,
        delta: String,
    },
    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgsDelta {
        item_id: String,
        delta: String,
    },
    #[serde(rename = "response.output_item.done")]
    OutputItemDone {
        #[serde(default)] output_index: u32,
        item: CodexOutputItem,
    },
    #[serde(rename = "response.completed")]
    Completed { response: CodexResponseMeta },
    #[serde(rename = "response.error")]
    Error { error: CodexErrorPayload },
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum CodexOutputItem {
    #[serde(rename = "message")]
    Message {
        #[serde(default)] id: String,
    },
    #[serde(rename = "function_call")]
    FunctionCall {
        #[serde(default)] id: String,
        #[serde(default)] call_id: String,
        #[serde(default)] name: String,
        #[serde(default)] arguments: String,
    },
    #[serde(rename = "reasoning")]
    Reasoning {
        #[serde(default)] id: String,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CodexResponseMeta {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub usage: Option<CodexUsage>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CodexUsage {
    #[serde(default)]
    pub input_tokens: u32,
    #[serde(default)]
    pub output_tokens: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CodexErrorPayload {
    pub message: String,
    #[serde(default)]
    pub code: Option<String>,
}

// ─── config ───

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    #[serde(default = "default_listen")]
    pub listen: String,
}

fn default_listen() -> String { "127.0.0.1:4000".to_string() }

impl Default for Config {
    fn default() -> Self {
        Self {
            listen: default_listen(),
        }
    }
}

/// Claude → ChatGPT model slugs the proxy is aware of. The actual mapping is
/// resolved at request time by [`ChatGptTier::map_model`]; this list exists
/// only so `/v1/models` can advertise the Anthropic-side names.
pub const KNOWN_CLAUDE_MODELS: &[&str] = &[
    "claude-opus-4-7",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
    "claude-haiku-4-5",
];

impl Config {
    pub fn load_or_default() -> Self {
        let path = match directories::ProjectDirs::from("", "", "claudeg") {
            Some(d) => d.config_dir().join("config.toml"),
            None => return Self::default(),
        };
        match std::fs::read_to_string(&path) {
            Ok(s) => Self::parse_with_legacy_warnings(&s).unwrap_or_else(|e| {
                eprintln!("warning: {} parse error: {}; using defaults", path.display(), e);
                Self::default()
            }),
            Err(_) => Self::default(),
        }
    }

    /// Parse a config.toml string. Emits one `tracing::warn!` per legacy key
    /// (`default_model`, `[models]`) — they're no longer honored as of v0.2.0
    /// since tier auto-mapping is the only path.
    pub fn parse_with_legacy_warnings(s: &str) -> Result<Self, toml::de::Error> {
        if let Ok(raw) = toml::from_str::<toml::Value>(s) {
            if raw.get("default_model").is_some() {
                tracing::warn!(
                    "claudeg: config.toml field `default_model` is no longer supported \
                     (v0.2.0+: tier auto-mapping). It will be ignored."
                );
            }
            if raw.get("models").is_some() {
                tracing::warn!(
                    "claudeg: config.toml field `[models]` is no longer supported \
                     (v0.2.0+: tier auto-mapping). It will be ignored."
                );
            }
        }
        toml::from_str::<Config>(s)
    }
}

// ─── ChatGPT tier (auto-detected from JWT `chatgpt_plan_type` claim) ───

/// ChatGPT subscription tier as reported by the JWT `chatgpt_plan_type` claim.
/// `Unknown` covers both "claim missing" and "claim value we don't recognize"
/// — both are treated like `Plus` by the mapper, but kept distinct so the
/// `whoami` UI can be explicit about the situation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ChatGptTier {
    Free,
    Go,
    Plus,
    Edu,
    Pro,
    Business,
    Enterprise,
    #[default]
    Unknown,
}

impl ChatGptTier {
    /// Case-insensitive parse of the JWT `chatgpt_plan_type` claim value.
    /// Anything not in the known set → `Unknown`.
    pub fn from_jwt_claim(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "free"       => Self::Free,
            "go"         => Self::Go,
            "plus"       => Self::Plus,
            "edu"        => Self::Edu,
            "pro"        => Self::Pro,
            "business"   => Self::Business,
            "enterprise" => Self::Enterprise,
            _            => Self::Unknown,
        }
    }

    /// Map a Claude model slug to the ChatGPT slug appropriate for this tier.
    /// Falls back to [`Self::default_fallback_model`] for unknown Claude
    /// models. **Panics** on `Tier::Free` — callers must short-circuit Free
    /// before any model mapping (it's rejected at handler entry).
    pub fn map_model(self, claude_model: &str) -> &'static str {
        match self {
            Self::Free => panic!(
                "ChatGptTier::map_model called on Free — callers must reject Free \
                 with HTTP 403 before mapping"
            ),
            Self::Go => match claude_model {
                "claude-opus-4-7"            => "gpt-5.5",
                "claude-sonnet-4-6"          => "gpt-5.5",
                "claude-haiku-4-5-20251001"  => "gpt-5.4-mini",
                "claude-haiku-4-5"           => "gpt-5.4-mini",
                _ => self.default_fallback_model(),
            },
            // Plus / Edu / Unknown share the same mapping (Unknown is treated as Plus).
            Self::Plus | Self::Edu | Self::Unknown => match claude_model {
                "claude-opus-4-7"            => "gpt-5.5",
                "claude-sonnet-4-6"          => "gpt-5.3-codex",
                "claude-haiku-4-5-20251001"  => "gpt-5.4-mini",
                "claude-haiku-4-5"           => "gpt-5.4-mini",
                _ => self.default_fallback_model(),
            },
            Self::Pro | Self::Business | Self::Enterprise => match claude_model {
                "claude-opus-4-7"            => "gpt-5.5-pro",
                "claude-sonnet-4-6"          => "gpt-5.3-codex-spark",
                "claude-haiku-4-5-20251001"  => "gpt-5.4-mini",
                "claude-haiku-4-5"           => "gpt-5.4-mini",
                _ => self.default_fallback_model(),
            },
        }
    }

    /// ChatGPT slug to use when the requested Claude model isn't in our table.
    pub fn default_fallback_model(self) -> &'static str {
        match self {
            Self::Free => "gpt-5.3-codex", // unreachable in practice; Free is rejected at the door
            Self::Go => "gpt-5.5",
            Self::Plus | Self::Edu | Self::Unknown => "gpt-5.3-codex",
            Self::Pro | Self::Business | Self::Enterprise => "gpt-5.3-codex-spark",
        }
    }

    /// Title-cased name for human-readable output (`whoami`, error messages).
    pub fn display_name(self) -> &'static str {
        match self {
            Self::Free       => "Free",
            Self::Go         => "Go",
            Self::Plus       => "Plus",
            Self::Edu        => "Edu",
            Self::Pro        => "Pro",
            Self::Business   => "Business",
            Self::Enterprise => "Enterprise",
            Self::Unknown    => "Unknown",
        }
    }

    /// Lowercase identifier, used in the per-request log line `tier=<name>`.
    pub fn as_log_str(self) -> &'static str {
        match self {
            Self::Free       => "free",
            Self::Go         => "go",
            Self::Plus       => "plus",
            Self::Edu        => "edu",
            Self::Pro        => "pro",
            Self::Business   => "business",
            Self::Enterprise => "enterprise",
            Self::Unknown    => "unknown",
        }
    }
}

// ─── translate: anthropic → codex ───

pub const DEFAULT_INSTRUCTIONS: &str = "You are a helpful coding assistant.";

/// Translate an Anthropic request to a Codex request. The caller supplies the
/// already-resolved ChatGPT model slug (via [`ChatGptTier::map_model`]) so
/// this function stays free of subscription-tier logic.
pub fn to_codex(req: &AnthropicReq, model_out: &str) -> CodexReq {
    let instructions = Some(req.system.as_ref().map(|s| match s {
        SystemField::Text(t) => t.clone(),
        SystemField::Blocks(bs) => bs
            .iter()
            .map(|SystemBlock::Text { text }| text.clone())
            .collect::<Vec<_>>()
            .join("\n"),
    }).unwrap_or_else(|| DEFAULT_INSTRUCTIONS.to_string()));

    let mut input: Vec<CodexItem> = Vec::new();
    for msg in &req.messages {
        let blocks = match &msg.content {
            MessageContent::Text(t) => vec![ContentBlock::Text { text: t.clone() }],
            MessageContent::Blocks(bs) => bs.clone(),
        };
        flatten_blocks_into(&msg.role, &blocks, &mut input);
    }

    let tools = req.tools.as_ref().map(|ts| {
        ts.iter()
            .map(|t| CodexTool::Function {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: t.input_schema.clone(),
            })
            .collect()
    });

    CodexReq {
        model: model_out.to_string(),
        instructions,
        stream: Some(true),
        input,
        tools,
        temperature: req.temperature,
        top_p: req.top_p,
        // ChatGPT-account Codex doesn't accept max_output_tokens; drop it.
        max_output_tokens: None,
        store: Some(false),
    }
}

fn flatten_blocks_into(role: &str, blocks: &[ContentBlock], out: &mut Vec<CodexItem>) {
    let mut message_content: Vec<CodexContent> = Vec::new();
    for b in blocks {
        match b {
            ContentBlock::Text { text } => {
                let c = if role == "assistant" {
                    CodexContent::OutputText { text: text.clone() }
                } else {
                    CodexContent::InputText { text: text.clone() }
                };
                message_content.push(c);
            }
            ContentBlock::Image { source } => {
                let url = match source {
                    ImageSource::Base64 { media_type, data } => {
                        format!("data:{media_type};base64,{data}")
                    }
                    ImageSource::Url { url } => url.clone(),
                };
                message_content.push(CodexContent::InputImage { image_url: url });
            }
            ContentBlock::ToolUse { id, name, input } => {
                // Flush pending message content, then emit a function_call item.
                if !message_content.is_empty() {
                    out.push(CodexItem::Message {
                        role: role.to_string(),
                        content: std::mem::take(&mut message_content),
                    });
                }
                out.push(CodexItem::FunctionCall {
                    call_id: id.clone(),
                    name: name.clone(),
                    arguments: serde_json::to_string(input).unwrap_or_else(|_| "{}".into()),
                });
            }
            ContentBlock::ToolResult { tool_use_id, content, .. } => {
                if !message_content.is_empty() {
                    out.push(CodexItem::Message {
                        role: role.to_string(),
                        content: std::mem::take(&mut message_content),
                    });
                }
                let output = match content {
                    Value::String(s) => serde_json::to_string(s).unwrap_or_else(|_| s.clone()),
                    Value::Null => "\"\"".into(),
                    other => other.to_string(),
                };
                out.push(CodexItem::FunctionCallOutput {
                    call_id: tool_use_id.clone(),
                    output,
                });
            }
        }
    }
    if !message_content.is_empty() {
        out.push(CodexItem::Message {
            role: role.to_string(),
            content: message_content,
        });
    }
}

// ─── translate: codex event → anthropic event ───
use uuid::Uuid;

#[derive(Debug, Clone, Copy)]
enum BlockKind { Text, ToolUse }

#[derive(Debug, Clone, Copy)]
struct OpenBlock { index: usize, kind: BlockKind }

pub struct Translator {
    message_id: String,
    model: String,
    next_index: usize,
    /// Tracks open content blocks keyed by upstream `item_id`. The Codex
    /// Responses API streams events grouped by `item_id` rather than by
    /// position, so we use it to correlate deltas with the block they belong to.
    open: HashMap<String, OpenBlock>,
    started: bool,
    done: bool,
    stop_reason: Option<&'static str>,
    output_tokens: u32,
    input_tokens: u32,
}

impl Translator {
    pub fn new(model: String) -> Self {
        Self {
            message_id: format!("msg_{}", Uuid::new_v4().simple()),
            model,
            next_index: 0,
            open: HashMap::new(),
            started: false,
            done: false,
            stop_reason: None,
            output_tokens: 0,
            input_tokens: 0,
        }
    }

    /// Returns Anthropic SSE events as `(event_name, json_data)` pairs.
    pub fn next(&mut self, ev: CodexEvent) -> Vec<(String, Value)> {
        let mut out = Vec::new();
        if self.done {
            return out;
        }
        match ev {
            CodexEvent::Created { response } => {
                if !self.started {
                    self.started = true;
                    let m = response.model.unwrap_or_else(|| self.model.clone());
                    self.model = m.clone();
                    out.push((
                        "message_start".into(),
                        serde_json::json!({
                            "type": "message_start",
                            "message": {
                                "id": self.message_id,
                                "type": "message",
                                "role": "assistant",
                                "model": m,
                                "content": [],
                                "stop_reason": Value::Null,
                                "stop_sequence": Value::Null,
                                "usage": { "input_tokens": 0, "output_tokens": 0 }
                            }
                        }),
                    ));
                }
            }
            CodexEvent::OutputItemAdded { item, .. } => {
                if let CodexOutputItem::FunctionCall { id, call_id, name, .. } = item {
                    let block_id = if !call_id.is_empty() { call_id } else { id.clone() };
                    let idx = self.alloc_index();
                    out.push((
                        "content_block_start".into(),
                        serde_json::json!({
                            "type": "content_block_start",
                            "index": idx,
                            "content_block": {
                                "type": "tool_use",
                                "id": block_id,
                                "name": name,
                                "input": {}
                            }
                        }),
                    ));
                    self.open.insert(id, OpenBlock { index: idx, kind: BlockKind::ToolUse });
                    self.stop_reason = Some("tool_use");
                }
                // Message and Reasoning items: no block opens here. Text blocks
                // are opened lazily on the first OutputTextDelta. Reasoning is
                // not surfaced as Anthropic content.
            }
            CodexEvent::OutputTextDelta { item_id, delta } => {
                let idx = match self.open.get(&item_id) {
                    Some(b) => b.index,
                    None => {
                        let idx = self.alloc_index();
                        out.push((
                            "content_block_start".into(),
                            serde_json::json!({
                                "type": "content_block_start",
                                "index": idx,
                                "content_block": { "type": "text", "text": "" }
                            }),
                        ));
                        self.open.insert(item_id.clone(), OpenBlock { index: idx, kind: BlockKind::Text });
                        idx
                    }
                };
                out.push((
                    "content_block_delta".into(),
                    serde_json::json!({
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": { "type": "text_delta", "text": delta }
                    }),
                ));
            }
            CodexEvent::FunctionCallArgsDelta { item_id, delta } => {
                if let Some(b) = self.open.get(&item_id).copied() {
                    out.push((
                        "content_block_delta".into(),
                        serde_json::json!({
                            "type": "content_block_delta",
                            "index": b.index,
                            "delta": { "type": "input_json_delta", "partial_json": delta }
                        }),
                    ));
                }
                // If we somehow never saw the OutputItemAdded for this item_id
                // (out-of-order events) we just drop the delta — the alternative
                // is opening a tool_use with no name, which Claude Code rejects.
            }
            CodexEvent::OutputItemDone { item, .. } => {
                let item_id = match &item {
                    CodexOutputItem::FunctionCall { id, .. } => id.clone(),
                    CodexOutputItem::Message { id, .. } => id.clone(),
                    CodexOutputItem::Reasoning { id, .. } => id.clone(),
                    CodexOutputItem::Other => String::new(),
                };
                if let Some(b) = self.open.remove(&item_id) {
                    out.push((
                        "content_block_stop".into(),
                        serde_json::json!({ "type": "content_block_stop", "index": b.index }),
                    ));
                    let _ = b.kind;
                }
            }
            CodexEvent::Completed { response } => {
                // Close any blocks still open (defensive; Codex *should* have
                // emitted OutputItemDone for each).
                let mut leftover: Vec<OpenBlock> = self.open.drain().map(|(_, b)| b).collect();
                leftover.sort_by_key(|b| b.index);
                for b in leftover {
                    out.push((
                        "content_block_stop".into(),
                        serde_json::json!({ "type": "content_block_stop", "index": b.index }),
                    ));
                }
                if let Some(u) = &response.usage {
                    self.input_tokens = u.input_tokens;
                    self.output_tokens = u.output_tokens;
                }
                let stop = self.stop_reason.unwrap_or("end_turn");
                out.push((
                    "message_delta".into(),
                    serde_json::json!({
                        "type": "message_delta",
                        "delta": { "stop_reason": stop, "stop_sequence": Value::Null },
                        "usage": { "input_tokens": self.input_tokens, "output_tokens": self.output_tokens }
                    }),
                ));
                out.push((
                    "message_stop".into(),
                    serde_json::json!({ "type": "message_stop" }),
                ));
                self.done = true;
            }
            CodexEvent::Error { error } => {
                out.push((
                    "error".into(),
                    serde_json::json!({
                        "type": "error",
                        "error": { "type": "api_error", "message": error.message }
                    }),
                ));
                self.done = true;
            }
            CodexEvent::Other => {}
        }
        out
    }

    fn alloc_index(&mut self) -> usize {
        let i = self.next_index;
        self.next_index += 1;
        i
    }
}

// ─── translate: anthropic ⇄ gemini ───
//
// The Gemini path targets Google's native streaming surface at
// `generativelanguage.googleapis.com/v1beta/models/<model>:streamGenerateContent`.
// We hit it with `?alt=sse&key=<API_KEY>` so it streams real SSE frames (the
// default JSON-array streaming format would force us to buffer the whole body).
//
// Wire shape on the way out (request → upstream):
//   { contents: [{role, parts: [{text|functionCall|functionResponse}]}],
//     systemInstruction: { parts: [{ text }] },
//     tools: [{ functionDeclarations: [{name, description, parameters}] }],
//     toolConfig: { functionCallingConfig: { mode, allowedFunctionNames } },
//     generationConfig: { temperature, topP, topK, stopSequences, maxOutputTokens } }
//
// Wire shape on the way back (upstream → translator):
//   data: { candidates: [{ content: { parts: [...] , role: "model" },
//                          finishReason, safetyRatings, ... }],
//           usageMetadata: { promptTokenCount, candidatesTokenCount, totalTokenCount } }

#[derive(Debug, Clone, Serialize)]
pub struct GeminiReq {
    pub contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "systemInstruction")]
    pub system_instruction: Option<GeminiSystemInstruction>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<GeminiTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "toolConfig")]
    pub tool_config: Option<GeminiToolConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "generationConfig")]
    pub generation_config: Option<GeminiGenerationConfig>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GeminiContent {
    pub role: String, // "user" or "model"
    pub parts: Vec<GeminiPart>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GeminiSystemInstruction {
    pub parts: Vec<GeminiPart>,
}

/// A content part. Gemini expresses each part as an object with exactly one of
/// `text`/`functionCall`/`functionResponse` set (no `type` discriminator), so
/// we model it as an untagged enum: serde flattens the variant's fields into
/// the parent object.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum GeminiPart {
    Text {
        text: String,
    },
    FunctionCall {
        #[serde(rename = "functionCall")]
        function_call: GeminiFunctionCall,
    },
    FunctionResponse {
        #[serde(rename = "functionResponse")]
        function_response: GeminiFunctionResponse,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiFunctionCall {
    pub name: String,
    #[serde(default)]
    pub args: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiFunctionResponse {
    pub name: String,
    pub response: Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct GeminiTool {
    #[serde(rename = "functionDeclarations")]
    pub function_declarations: Vec<GeminiFunctionDecl>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GeminiFunctionDecl {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct GeminiToolConfig {
    #[serde(rename = "functionCallingConfig")]
    pub function_calling_config: GeminiFunctionCallingConfig,
}

#[derive(Debug, Clone, Serialize)]
pub struct GeminiFunctionCallingConfig {
    /// "AUTO" | "ANY" | "NONE" — uppercase enum per Gemini's API.
    pub mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "allowedFunctionNames")]
    pub allowed_function_names: Option<Vec<String>>,
}

#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
}

/// Anthropic-side `tool_use_id` → tool name. Used while building the Gemini
/// request to fill the `name` on each `functionResponse` block (Gemini has no
/// notion of a tool-use ID; it correlates by name).
type ToolNameMap = std::collections::HashMap<String, String>;

/// Translate an Anthropic Messages request into a Gemini `generateContent`
/// request body. The caller supplies the resolved model slug separately when
/// building the upstream URL — this function only fills the body.
///
/// Image content blocks are not supported on the Gemini path in this iteration;
/// callers should reject them at the handler level. Tool-use IDs are mapped
/// through `name`, which mirrors how Claude Code uses tools (each tool has a
/// unique name; calls are serial in the agent loop).
pub fn to_gemini(req: &AnthropicReq) -> Result<GeminiReq, String> {
    let system_instruction = req.system.as_ref().map(|s| {
        let text = match s {
            SystemField::Text(t) => t.clone(),
            SystemField::Blocks(bs) => bs.iter()
                .map(|SystemBlock::Text { text }| text.clone())
                .collect::<Vec<_>>()
                .join("\n"),
        };
        GeminiSystemInstruction { parts: vec![GeminiPart::Text { text }] }
    });

    // First pass: walk all messages to learn tool_use_id → name. We need this
    // when emitting functionResponse for ToolResult blocks in the second pass.
    let mut id_to_name: ToolNameMap = Default::default();
    for msg in &req.messages {
        if let MessageContent::Blocks(bs) = &msg.content {
            for b in bs {
                if let ContentBlock::ToolUse { id, name, .. } = b {
                    id_to_name.insert(id.clone(), name.clone());
                }
            }
        }
    }

    let mut contents: Vec<GeminiContent> = Vec::new();
    for msg in &req.messages {
        let blocks: Vec<ContentBlock> = match &msg.content {
            MessageContent::Text(t) => vec![ContentBlock::Text { text: t.clone() }],
            MessageContent::Blocks(bs) => bs.clone(),
        };
        flatten_blocks_to_gemini(&msg.role, &blocks, &id_to_name, &mut contents)?;
    }

    let tools = req.tools.as_ref().map(|ts| {
        let decls = ts.iter().map(|t| GeminiFunctionDecl {
            name: t.name.clone(),
            description: t.description.clone(),
            parameters: t.input_schema.clone(),
        }).collect();
        vec![GeminiTool { function_declarations: decls }]
    });

    let generation_config = {
        let mut cfg = GeminiGenerationConfig::default();
        cfg.temperature = req.temperature;
        cfg.top_p = req.top_p;
        cfg.max_output_tokens = req.max_tokens;
        cfg.stop_sequences = req.stop_sequences.clone();
        if cfg.temperature.is_some()
            || cfg.top_p.is_some()
            || cfg.max_output_tokens.is_some()
            || cfg.stop_sequences.is_some()
        {
            Some(cfg)
        } else { None }
    };

    Ok(GeminiReq {
        contents,
        system_instruction,
        tools,
        // Anthropic's default tool_choice is "auto" → matches Gemini's default
        // (AUTO when toolConfig is omitted). We never need to set ANY/NONE
        // because Claude Code doesn't send `tool_choice`.
        tool_config: None,
        generation_config,
    })
}

fn flatten_blocks_to_gemini(
    role: &str,
    blocks: &[ContentBlock],
    id_to_name: &ToolNameMap,
    out: &mut Vec<GeminiContent>,
) -> Result<(), String> {
    // Gemini's `role` is "user" or "model" (no "assistant"). Function-response
    // parts must also live under role="user" per Google's spec.
    let mut parts: Vec<GeminiPart> = Vec::new();
    let mut current_role = anthropic_role_to_gemini(role);

    let flush = |parts: &mut Vec<GeminiPart>, role: &str, out: &mut Vec<GeminiContent>| {
        if !parts.is_empty() {
            out.push(GeminiContent {
                role: role.to_string(),
                parts: std::mem::take(parts),
            });
        }
    };

    for b in blocks {
        match b {
            ContentBlock::Text { text } => {
                if !text.is_empty() {
                    parts.push(GeminiPart::Text { text: text.clone() });
                }
            }
            ContentBlock::ToolUse { name, input, .. } => {
                // Anthropic puts tool_use blocks under role="assistant";
                // Gemini puts functionCall under role="model".
                parts.push(GeminiPart::FunctionCall {
                    function_call: GeminiFunctionCall {
                        name: name.clone(),
                        args: input.clone(),
                    },
                });
            }
            ContentBlock::ToolResult { tool_use_id, content, .. } => {
                // Tool results in Anthropic land under role="user". Gemini
                // expects them under role="user" too, but as functionResponse
                // parts. Flush any pending text parts first to keep grouping
                // clean.
                if !parts.is_empty() && current_role != "user" {
                    flush(&mut parts, &current_role, out);
                    current_role = "user".into();
                }
                let name = id_to_name.get(tool_use_id).cloned().unwrap_or_else(|| {
                    // Defensive: if the request didn't include the matching
                    // tool_use block, fall back to the id as the name. Gemini
                    // will probably ignore the part but at least the request
                    // stays well-formed.
                    tool_use_id.clone()
                });
                let response = wrap_tool_response(content);
                parts.push(GeminiPart::FunctionResponse {
                    function_response: GeminiFunctionResponse { name, response },
                });
            }
            ContentBlock::Image { .. } => {
                return Err(
                    "image inputs are not supported on the Gemini backend yet — \
                     switch to the ChatGPT backend or remove the image".into()
                );
            }
        }
    }
    flush(&mut parts, &current_role, out);
    Ok(())
}

fn anthropic_role_to_gemini(role: &str) -> String {
    match role {
        "assistant" => "model".into(),
        _ => "user".into(),
    }
}

/// Gemini's `functionResponse.response` is an arbitrary JSON object that the
/// model interprets as the tool's output. Anthropic's `tool_result.content` can
/// be a string or an array of blocks. Normalise both into a `{ content: ... }`
/// object so the model always sees the same shape.
fn wrap_tool_response(content: &Value) -> Value {
    match content {
        Value::Null => serde_json::json!({ "content": "" }),
        Value::String(s) => serde_json::json!({ "content": s }),
        Value::Array(arr) => {
            // Anthropic blocks → flatten text blocks into a single string for
            // a less noisy prompt; preserve non-text content (e.g. JSON) as-is.
            let mut joined = String::new();
            let mut non_text: Vec<Value> = Vec::new();
            for v in arr {
                if let Some(t) = v.get("type").and_then(|t| t.as_str()) {
                    if t == "text" {
                        if let Some(s) = v.get("text").and_then(|s| s.as_str()) {
                            if !joined.is_empty() { joined.push('\n'); }
                            joined.push_str(s);
                            continue;
                        }
                    }
                }
                non_text.push(v.clone());
            }
            if non_text.is_empty() {
                serde_json::json!({ "content": joined })
            } else {
                serde_json::json!({ "content": joined, "extra": non_text })
            }
        }
        other => serde_json::json!({ "content": other.clone() }),
    }
}

// ─── translate: gemini stream → anthropic SSE ───

/// One SSE `data: { ... }` frame from Gemini's streamGenerateContent. We only
/// model the fields we consume; everything else is ignored. Defaults keep us
/// resilient to changes upstream (e.g. new safety fields, telemetry blocks).
#[derive(Debug, Clone, Deserialize)]
pub struct GeminiStreamFrame {
    #[serde(default)]
    pub candidates: Vec<GeminiCandidate>,
    #[serde(default)]
    #[serde(rename = "usageMetadata")]
    pub usage_metadata: Option<GeminiUsageMetadata>,
    /// Set on error frames the server may emit in-stream.
    #[serde(default)]
    pub error: Option<GeminiErrorPayload>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GeminiCandidate {
    #[serde(default)]
    pub content: Option<GeminiCandidateContent>,
    #[serde(default)]
    #[serde(rename = "finishReason")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GeminiCandidateContent {
    #[serde(default)]
    pub parts: Vec<GeminiInPart>,
    #[serde(default)]
    pub role: Option<String>,
}

/// Inbound part. Gemini chooses exactly one of `text`/`functionCall`/
/// `functionResponse`, so we represent it as a struct of options.
#[derive(Debug, Clone, Deserialize)]
pub struct GeminiInPart {
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default, rename = "functionCall")]
    pub function_call: Option<GeminiFunctionCall>,
    #[serde(default, rename = "functionResponse")]
    pub function_response: Option<GeminiFunctionResponse>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct GeminiUsageMetadata {
    #[serde(default, rename = "promptTokenCount")]
    pub prompt_token_count: u32,
    #[serde(default, rename = "candidatesTokenCount")]
    pub candidates_token_count: u32,
    #[serde(default, rename = "totalTokenCount")]
    pub _total_token_count: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GeminiErrorPayload {
    #[serde(default)]
    pub code: Option<u32>,
    pub message: String,
    #[serde(default)]
    pub _status: Option<String>,
}

/// Translates Gemini's stream frames into Anthropic-shaped SSE pairs.
///
/// Streaming model:
///   - First frame: emit `message_start` (we synthesise the message ID).
///   - For each part in a candidate: if it's `text`, open or append to the
///     current text block; if it's `functionCall`, open a tool_use block, emit
///     the entire `args` object as one `input_json_delta`, and close it.
///   - When `finishReason` shows up: emit `message_delta` (with stop_reason +
///     usage) and `message_stop`.
pub struct GeminiTranslator {
    message_id: String,
    model: String,
    next_index: usize,
    /// Index of the currently-open text block, if any. Tool_use blocks open
    /// and close within a single part so they don't need to persist here.
    open_text_index: Option<usize>,
    started: bool,
    done: bool,
    stop_reason: Option<&'static str>,
    input_tokens: u32,
    output_tokens: u32,
}

impl GeminiTranslator {
    pub fn new(model: String) -> Self {
        Self {
            message_id: format!("msg_{}", Uuid::new_v4().simple()),
            model,
            next_index: 0,
            open_text_index: None,
            started: false,
            done: false,
            stop_reason: None,
            input_tokens: 0,
            output_tokens: 0,
        }
    }

    /// Ingest one Gemini stream frame, return zero or more Anthropic SSE
    /// `(event_name, json_data)` pairs to emit downstream.
    pub fn next(&mut self, frame: GeminiStreamFrame) -> Vec<(String, Value)> {
        let mut out = Vec::new();
        if self.done { return out; }

        // In-stream error frames (rare; Gemini usually sets HTTP status, but
        // safety blocks can surface here).
        if let Some(err) = frame.error {
            out.push((
                "error".into(),
                serde_json::json!({
                    "type": "error",
                    "error": { "type": "api_error", "message": err.message }
                }),
            ));
            self.done = true;
            return out;
        }

        if !self.started {
            self.started = true;
            out.push((
                "message_start".into(),
                serde_json::json!({
                    "type": "message_start",
                    "message": {
                        "id": self.message_id,
                        "type": "message",
                        "role": "assistant",
                        "model": self.model,
                        "content": [],
                        "stop_reason": Value::Null,
                        "stop_sequence": Value::Null,
                        "usage": { "input_tokens": 0, "output_tokens": 0 }
                    }
                }),
            ));
        }

        // Per request we only expect candidates[0]; multi-candidate isn't a
        // mode we use. If a frame somehow ships more, we still consume the
        // first only — Anthropic clients don't have a place to put alternates.
        let candidate = frame.candidates.into_iter().next();

        if let Some(cand) = candidate {
            if let Some(content) = cand.content {
                for part in content.parts {
                    if let Some(text) = part.text {
                        if text.is_empty() { continue; }
                        let idx = match self.open_text_index {
                            Some(i) => i,
                            None => {
                                let i = self.alloc_index();
                                out.push((
                                    "content_block_start".into(),
                                    serde_json::json!({
                                        "type": "content_block_start",
                                        "index": i,
                                        "content_block": { "type": "text", "text": "" }
                                    }),
                                ));
                                self.open_text_index = Some(i);
                                i
                            }
                        };
                        out.push((
                            "content_block_delta".into(),
                            serde_json::json!({
                                "type": "content_block_delta",
                                "index": idx,
                                "delta": { "type": "text_delta", "text": text }
                            }),
                        ));
                    } else if let Some(fc) = part.function_call {
                        // Close any open text block before opening a tool_use.
                        if let Some(i) = self.open_text_index.take() {
                            out.push((
                                "content_block_stop".into(),
                                serde_json::json!({
                                    "type": "content_block_stop", "index": i
                                }),
                            ));
                        }
                        let i = self.alloc_index();
                        let tool_id = format!("toolu_{}", Uuid::new_v4().simple());
                        out.push((
                            "content_block_start".into(),
                            serde_json::json!({
                                "type": "content_block_start",
                                "index": i,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": tool_id,
                                    "name": fc.name,
                                    "input": {}
                                }
                            }),
                        ));
                        // Gemini emits args as a whole object per part — surface
                        // it as one input_json_delta so the client's accumulator
                        // sees the full JSON string.
                        let args_str = serde_json::to_string(&fc.args)
                            .unwrap_or_else(|_| "{}".into());
                        out.push((
                            "content_block_delta".into(),
                            serde_json::json!({
                                "type": "content_block_delta",
                                "index": i,
                                "delta": { "type": "input_json_delta", "partial_json": args_str }
                            }),
                        ));
                        out.push((
                            "content_block_stop".into(),
                            serde_json::json!({
                                "type": "content_block_stop", "index": i
                            }),
                        ));
                        self.stop_reason = Some("tool_use");
                    }
                    // `functionResponse` shouldn't appear in upstream stream
                    // frames (it's a request-side concept), so we ignore it.
                }
            }
            if let Some(fr) = cand.finish_reason {
                // If we already emitted a tool_use block, the model produced a
                // call — that's "tool_use" on the Anthropic side regardless of
                // whether Gemini reports STOP or MAX_TOKENS. Only let MAX_TOKENS
                // override (truncation matters more than the tool reason).
                let mapped = map_finish_reason(&fr).unwrap_or("end_turn");
                self.stop_reason = match (self.stop_reason, mapped) {
                    (Some("tool_use"), "end_turn")   => Some("tool_use"),
                    (Some("tool_use"), "max_tokens") => Some("max_tokens"),
                    (_, other)                        => Some(other),
                };
            }
        }

        if let Some(usage) = frame.usage_metadata {
            self.input_tokens = usage.prompt_token_count;
            self.output_tokens = usage.candidates_token_count;
        }

        out
    }

    /// Emit terminal frames (close any open text block, `message_delta`, and
    /// `message_stop`). Call exactly once after the upstream SSE stream ends.
    pub fn finish(&mut self) -> Vec<(String, Value)> {
        let mut out = Vec::new();
        if self.done { return out; }
        if !self.started {
            // Stream had zero frames — emit a synthetic message_start so
            // the client sees a well-formed envelope before message_stop.
            self.started = true;
            out.push((
                "message_start".into(),
                serde_json::json!({
                    "type": "message_start",
                    "message": {
                        "id": self.message_id,
                        "type": "message",
                        "role": "assistant",
                        "model": self.model,
                        "content": [],
                        "stop_reason": Value::Null,
                        "stop_sequence": Value::Null,
                        "usage": { "input_tokens": 0, "output_tokens": 0 }
                    }
                }),
            ));
        }
        if let Some(i) = self.open_text_index.take() {
            out.push((
                "content_block_stop".into(),
                serde_json::json!({ "type": "content_block_stop", "index": i }),
            ));
        }
        let stop = self.stop_reason.unwrap_or("end_turn");
        out.push((
            "message_delta".into(),
            serde_json::json!({
                "type": "message_delta",
                "delta": { "stop_reason": stop, "stop_sequence": Value::Null },
                "usage": {
                    "input_tokens": self.input_tokens,
                    "output_tokens": self.output_tokens,
                }
            }),
        ));
        out.push((
            "message_stop".into(),
            serde_json::json!({ "type": "message_stop" }),
        ));
        self.done = true;
        out
    }

    fn alloc_index(&mut self) -> usize {
        let i = self.next_index;
        self.next_index += 1;
        i
    }
}

fn map_finish_reason(r: &str) -> Option<&'static str> {
    match r {
        "STOP"     => Some("end_turn"),
        "MAX_TOKENS" => Some("max_tokens"),
        // SAFETY / RECITATION / BLOCKLIST / PROHIBITED_CONTENT / SPII / OTHER
        // → "end_turn" but log the upstream reason so operators can see it.
        _ => {
            tracing::debug!(finish_reason = %r, "gemini finishReason mapped to end_turn");
            Some("end_turn")
        }
    }
}

// ─── error ───

#[derive(Debug, thiserror::Error)]
pub enum AppError {
    #[error("{0}")]
    InvalidRequest(String),
    #[error("not authenticated; run `claudeg login`")]
    NotAuthenticated,
    #[error("claudeg needs ChatGPT Go/Plus/Pro/Business/Enterprise/Edu. Detected: free. Upgrade at https://chatgpt.com/pricing")]
    SubscriptionRequired,
    #[error("upstream rate limit")]
    RateLimit { retry_after: Option<String> },
    #[error("{0}")]
    Upstream(String),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl AppError {
    pub fn status(&self) -> StatusCode {
        match self {
            Self::InvalidRequest(_)     => StatusCode::BAD_REQUEST,
            Self::NotAuthenticated      => StatusCode::UNAUTHORIZED,
            Self::SubscriptionRequired  => StatusCode::FORBIDDEN,
            Self::RateLimit { .. }      => StatusCode::TOO_MANY_REQUESTS,
            Self::Upstream(_)           => StatusCode::BAD_GATEWAY,
            Self::Other(_)              => StatusCode::BAD_GATEWAY,
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Self::InvalidRequest(_)     => "invalid_request_error",
            Self::NotAuthenticated      => "authentication_error",
            Self::SubscriptionRequired  => "subscription_required",
            Self::RateLimit { .. }      => "rate_limit_error",
            Self::Upstream(_)           => "api_error",
            Self::Other(_)              => "api_error",
        }
    }

    pub fn body(&self) -> Value {
        // `subscription_required` uses the simpler `{"error":{...}}` shape
        // specified in the v0.2.0 tier-detection design (no outer `"type":"error"`).
        if matches!(self, Self::SubscriptionRequired) {
            return serde_json::json!({
                "error": { "type": self.kind(), "message": self.to_string() }
            });
        }
        serde_json::json!({
            "type": "error",
            "error": { "type": self.kind(), "message": self.to_string() }
        })
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let mut resp = (self.status(), Json(self.body())).into_response();
        if let AppError::RateLimit { retry_after: Some(s) } = &self {
            if let Ok(v) = axum::http::HeaderValue::from_str(s) {
                resp.headers_mut().insert("retry-after", v);
            }
        }
        resp
    }
}

// ─── extractors ───

/// `axum::Json<T>` wrapper whose rejection emits the Anthropic error envelope
/// instead of axum's default plain-string body. Spec §10 requires every error
/// response (including malformed JSON) to look like:
///   `{"type":"error","error":{"type":"invalid_request_error","message":"…"}}`.
pub struct AnthropicJson<T>(pub T);

impl<T, S> axum::extract::FromRequest<S> for AnthropicJson<T>
where
    T: serde::de::DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = AppError;

    async fn from_request(
        req: axum::extract::Request,
        state: &S,
    ) -> Result<Self, Self::Rejection> {
        match axum::Json::<T>::from_request(req, state).await {
            Ok(axum::Json(v)) => Ok(AnthropicJson(v)),
            Err(rej) => Err(AppError::InvalidRequest(rej.body_text())),
        }
    }
}

// ─── auth ───
use std::path::{Path, PathBuf};
use std::time::SystemTime;

// OAuth config matching OpenAI's Codex CLI (codex-rs/login/src/server.rs).
// Issuer: https://auth.openai.com — PKCE S256 flow.
pub const OAUTH_CLIENT_ID:     &str = "app_EMoamEEZ73f0CkXaXp7hrann";
pub const OAUTH_AUTHORIZE_URL: &str = "https://auth.openai.com/oauth/authorize";
pub const OAUTH_TOKEN_URL:     &str = "https://auth.openai.com/oauth/token";
pub const OAUTH_SCOPES:        &str = "openid profile email offline_access api.connectors.read api.connectors.invoke";

// PKCE callback listener ports — mirror the official Codex CLI
// (codex-rs/login/src/server.rs).
pub const PKCE_PRIMARY_PORT:  u16 = 1455;
pub const PKCE_FALLBACK_PORT: u16 = 1457;
const PKCE_BIND_RETRIES:      usize = 10;
const PKCE_BIND_RETRY_DELAY:  std::time::Duration = std::time::Duration::from_millis(200);
const PKCE_CALLBACK_TIMEOUT:  std::time::Duration = std::time::Duration::from_secs(300);

/// OAuth-backed ChatGPT-subscription auth. This is the original v0.1/v0.2
/// auth payload; it's now the `Chatgpt` variant of [`AuthState`]. The on-disk
/// JSON for this variant carries an extra `"provider": "chatgpt"` discriminator
/// from v0.3 onward; older untagged files still load via the legacy fallback in
/// [`AuthState::load_from`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatgptAuth {
    pub access_token: String,
    pub refresh_token: String,
    #[serde(with = "humantime_serde_systemtime")]
    pub expires_at: SystemTime,
    #[serde(default)]
    pub account_id: Option<String>,
    /// ChatGPT subscription tier as parsed from the JWT's `chatgpt_plan_type`
    /// claim. `Unknown` (the default) covers old auth.json files written by
    /// v0.1.x and JWTs missing the claim.
    #[serde(default)]
    pub tier: ChatGptTier,
}

/// API-key-backed Google Gemini auth, introduced in v0.3.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiAuth {
    pub api_key: String,
    #[serde(default = "default_gemini_model")]
    pub model: String,
}

pub fn default_gemini_model() -> String { "gemini-flash-latest".to_string() }

mod humantime_serde_systemtime {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    pub fn serialize<S: Serializer>(t: &SystemTime, s: S) -> Result<S::Ok, S::Error> {
        let secs = t.duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        secs.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<SystemTime, D::Error> {
        let secs = u64::deserialize(d)?;
        Ok(UNIX_EPOCH + Duration::from_secs(secs))
    }
}

impl ChatgptAuth {
    pub fn is_fresh(&self, slack: std::time::Duration) -> bool {
        match self.expires_at.duration_since(SystemTime::now()) {
            Ok(rem) => rem > slack,
            Err(_) => false,
        }
    }
}

/// Provider-tagged auth payload persisted in `auth.json`. Either a ChatGPT
/// OAuth subscription or a Gemini API key. On disk we write a `"provider"`
/// discriminator; legacy untagged v0.1/v0.2 files are accepted as ChatGPT for
/// backward compatibility (rewritten with the tag on the next save).
#[derive(Debug, Clone)]
pub enum AuthState {
    Chatgpt(ChatgptAuth),
    Gemini(GeminiAuth),
}

impl AuthState {
    pub fn default_path() -> PathBuf {
        directories::ProjectDirs::from("", "", "claudeg")
            .map(|d| d.config_dir().to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."))
            .join("auth.json")
    }

    pub fn load_default() -> Option<Self> {
        Self::load_from(&Self::default_path()).ok()
    }

    pub fn load_from(path: &Path) -> anyhow::Result<Self> {
        let s = std::fs::read_to_string(path)?;
        let mut v: Value = serde_json::from_str(&s)?;
        let provider = v.get("provider").and_then(|p| p.as_str()).map(|s| s.to_string());
        if let Some(obj) = v.as_object_mut() {
            obj.remove("provider");
        }
        match provider.as_deref() {
            Some("gemini") => {
                let inner: GeminiAuth = serde_json::from_value(v)
                    .map_err(|e| anyhow::anyhow!("auth.json (gemini): {e}"))?;
                Ok(AuthState::Gemini(inner))
            }
            // "chatgpt" or untagged legacy: deserialize as ChatgptAuth and
            // re-derive tier/account_id from the JWT so stale caches heal.
            _ => {
                let inner: ChatgptAuth = serde_json::from_value(v)
                    .map_err(|e| anyhow::anyhow!("auth.json (chatgpt): {e}"))?;
                Ok(AuthState::Chatgpt(Self::sanitize_chatgpt(inner)))
            }
        }
    }

    /// Re-derive tier and (if absent) account_id from the cached JWT. Run on
    /// every load so v0.1.x caches without a `tier` field self-heal, and so
    /// stale tier values picked up after a plan upgrade get corrected.
    fn sanitize_chatgpt(mut state: ChatgptAuth) -> ChatgptAuth {
        let (jwt_account_id, jwt_tier) = extract_jwt_claims(&state.access_token);
        state.tier = jwt_tier;
        if state.account_id.is_none() {
            state.account_id = jwt_account_id;
        }
        state
    }

    pub fn save_to(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let json = match self {
            AuthState::Chatgpt(c) => {
                let mut v = serde_json::to_value(c)?;
                if let Some(obj) = v.as_object_mut() {
                    obj.insert("provider".into(), Value::String("chatgpt".into()));
                }
                serde_json::to_string_pretty(&v)?
            }
            AuthState::Gemini(g) => {
                let mut v = serde_json::to_value(g)?;
                if let Some(obj) = v.as_object_mut() {
                    obj.insert("provider".into(), Value::String("gemini".into()));
                }
                serde_json::to_string_pretty(&v)?
            }
        };
        std::fs::write(path, json)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(path)?.permissions();
            perms.set_mode(0o600);
            std::fs::set_permissions(path, perms)?;
        }
        Ok(())
    }

    pub fn as_chatgpt(&self) -> Option<&ChatgptAuth> {
        if let Self::Chatgpt(c) = self { Some(c) } else { None }
    }

    pub fn as_gemini(&self) -> Option<&GeminiAuth> {
        if let Self::Gemini(g) = self { Some(g) } else { None }
    }

    /// Short identifier used in logs and the request-log middleware.
    pub fn provider_name(&self) -> &'static str {
        match self { Self::Chatgpt(_) => "chatgpt", Self::Gemini(_) => "gemini" }
    }
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    #[serde(default)]
    refresh_token: Option<String>,
    expires_in: u64,
    #[serde(default)]
    account_id: Option<String>,
}

/// Extract `chatgpt_account_id` and `chatgpt_plan_type` from a ChatGPT-issued
/// JWT access token in a single decode. JWT format: `header.payload.signature`
/// — we only need the middle segment. The function never panics; missing /
/// malformed fields surface as `None` / `ChatGptTier::Unknown`.
pub fn extract_jwt_claims(access_token: &str) -> (Option<String>, ChatGptTier) {
    use base64::Engine;
    let Some(payload_b64) = access_token.split('.').nth(1) else {
        return (None, ChatGptTier::Unknown);
    };
    let Ok(bytes) = base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(payload_b64) else {
        return (None, ChatGptTier::Unknown);
    };
    let Ok(v) = serde_json::from_slice::<Value>(&bytes) else {
        return (None, ChatGptTier::Unknown);
    };
    let auth = v.get("https://api.openai.com/auth");
    let account_id = auth
        .and_then(|a| a.get("chatgpt_account_id"))
        .and_then(|v| v.as_str())
        .map(String::from);
    let tier = auth
        .and_then(|a| a.get("chatgpt_plan_type"))
        .and_then(|v| v.as_str())
        .map(ChatGptTier::from_jwt_claim)
        .unwrap_or(ChatGptTier::Unknown);
    (account_id, tier)
}

/// Backwards-compatible shim — same as `extract_jwt_claims(t).0`.
pub fn extract_account_id_from_jwt(access_token: &str) -> Option<String> {
    extract_jwt_claims(access_token).0
}

pub async fn refresh_at(
    client: &reqwest::Client,
    base: &str,
    path: &str,
    client_id: &str,
    refresh_token: &str,
) -> Result<ChatgptAuth, AppError> {
    let url = format!("{base}{path}");
    let resp = client
        .post(&url)
        .form(&[
            ("grant_type",    "refresh_token"),
            ("client_id",     client_id),
            ("refresh_token", refresh_token),
        ])
        .send()
        .await
        .map_err(|e| AppError::Upstream(e.to_string()))?;
    let status = resp.status();
    if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::BAD_REQUEST {
        return Err(AppError::NotAuthenticated);
    }
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(AppError::Upstream(format!("refresh failed {status}: {body}")));
    }
    let tr: TokenResponse = resp.json().await.map_err(|e| AppError::Upstream(e.to_string()))?;
    let (jwt_account_id, tier) = extract_jwt_claims(&tr.access_token);
    let account_id = tr.account_id.or(jwt_account_id);
    Ok(ChatgptAuth {
        access_token: tr.access_token,
        refresh_token: tr.refresh_token.unwrap_or_else(|| refresh_token.to_string()),
        expires_at: SystemTime::now() + std::time::Duration::from_secs(tr.expires_in),
        account_id,
        tier,
    })
}

pub async fn refresh(state: &ChatgptAuth) -> Result<ChatgptAuth, AppError> {
    refresh_with_origin(state, &default_auth_origin()?).await
}

/// Default `scheme://host` for the OAuth issuer, derived from
/// `OAUTH_TOKEN_URL` so a single constant remains the source of truth.
pub fn default_auth_origin() -> Result<String, AppError> {
    let parsed = url::Url::parse(OAUTH_TOKEN_URL).map_err(|e| AppError::Other(e.into()))?;
    Ok(format!(
        "{}://{}",
        parsed.scheme(),
        parsed.host_str().unwrap_or("auth.openai.com"),
    ))
}

/// Same as `refresh`, but the OAuth token endpoint origin can be overridden
/// (the path comes from `OAUTH_TOKEN_URL`). Used by tests + AppState.
pub async fn refresh_with_origin(state: &ChatgptAuth, origin: &str) -> Result<ChatgptAuth, AppError> {
    let client = reqwest::Client::new();
    let parsed = url::Url::parse(OAUTH_TOKEN_URL).map_err(|e| AppError::Other(e.into()))?;
    let path = parsed.path().to_string();
    refresh_at(&client, origin, &path, OAUTH_CLIENT_ID, &state.refresh_token).await
}

/// 64 random bytes URL-safe-base64-encoded as the PKCE `code_verifier`, paired
/// with the SHA-256 `code_challenge` derived from it. Mirrors the official
/// Codex CLI's `codex-rs/login/src/pkce.rs`.
pub fn generate_pkce() -> (String, String) {
    use base64::Engine;
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use sha2::{Digest, Sha256};

    use rand::RngCore;
    let mut bytes = [0u8; 64];
    rand::thread_rng().fill_bytes(&mut bytes);
    let verifier = URL_SAFE_NO_PAD.encode(bytes);

    let digest = Sha256::digest(verifier.as_bytes());
    let challenge = URL_SAFE_NO_PAD.encode(digest);

    (verifier, challenge)
}

/// 32 random bytes URL-safe-base64-encoded — used as the OAuth `state` CSRF
/// token.
pub fn generate_state() -> String {
    use base64::Engine;
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;

    use rand::RngCore;
    let mut bytes = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut bytes);
    URL_SAFE_NO_PAD.encode(bytes)
}

/// Build the OAuth authorize URL with the standard PKCE + redirect params.
pub fn build_authorize_url(
    base: &str,
    authorize_path: &str,
    client_id: &str,
    scope: &str,
    redirect_uri: &str,
    state: &str,
    code_challenge: &str,
) -> String {
    let mut u = url::Url::parse(&format!("{base}{authorize_path}"))
        .expect("authorize URL is statically well-formed");
    u.query_pairs_mut()
        .append_pair("response_type",         "code")
        .append_pair("client_id",             client_id)
        .append_pair("redirect_uri",          redirect_uri)
        .append_pair("scope",                 scope)
        .append_pair("state",                 state)
        .append_pair("code_challenge",        code_challenge)
        .append_pair("code_challenge_method", "S256");
    u.into()
}

/// Bind the PKCE callback listener on `127.0.0.1:1455`, retrying briefly to
/// ride out stale sockets, then falling back to `:1457`. Mirrors
/// `codex-rs/login/src/server.rs`.
async fn bind_pkce_listener() -> Result<tokio::net::TcpListener, AppError> {
    use tokio::net::TcpListener;

    let primary = std::net::SocketAddr::from(([127, 0, 0, 1], PKCE_PRIMARY_PORT));
    for _ in 0..PKCE_BIND_RETRIES {
        if let Ok(l) = TcpListener::bind(primary).await {
            return Ok(l);
        }
        tokio::time::sleep(PKCE_BIND_RETRY_DELAY).await;
    }

    let fallback = std::net::SocketAddr::from(([127, 0, 0, 1], PKCE_FALLBACK_PORT));
    if let Ok(l) = TcpListener::bind(fallback).await {
        return Ok(l);
    }

    Err(AppError::Other(anyhow::anyhow!(
        "could not bind callback listener on ports {PKCE_PRIMARY_PORT} or {PKCE_FALLBACK_PORT}"
    )))
}

/// Bind the PKCE callback listener at a specific port (used by tests so they
/// can pick an unused high port instead of racing 1455 with real users).
async fn bind_pkce_listener_on(port: u16) -> Result<tokio::net::TcpListener, AppError> {
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| AppError::Other(anyhow::anyhow!(
            "could not bind callback listener on port {port}: {e}"
        )))
}

/// Result of the browser callback: the OAuth `code` (or an error message
/// describing why we couldn't extract one — e.g. CSRF state mismatch).
type CallbackResult = Result<String, String>;

#[derive(Deserialize)]
struct CallbackQuery {
    code: Option<String>,
    state: Option<String>,
    error: Option<String>,
    error_description: Option<String>,
}

/// Core PKCE flow. Generic over the OAuth issuer base URL so tests can point
/// it at a wiremock server. `listener_port_override = Some(p)` forces the
/// callback listener onto port `p` (test-only); `None` uses the standard
/// 1455-with-fallback-1457 behavior.
pub async fn pkce_login_at(
    base: &str,
    authorize_path: &str,
    token_path: &str,
    client_id: &str,
    scope: &str,
    on_authorize_url: impl Fn(&str),
    listener_port_override: Option<u16>,
) -> Result<ChatgptAuth, AppError> {
    use axum::extract::Query;
    use axum::response::Html;
    use axum::routing::get;
    use axum::Router;

    // 1. bind callback listener.
    let listener = match listener_port_override {
        Some(p) => bind_pkce_listener_on(p).await?,
        None    => bind_pkce_listener().await?,
    };
    let local_addr = listener
        .local_addr()
        .map_err(|e| AppError::Other(anyhow::anyhow!(e)))?;
    let port = local_addr.port();
    let redirect_uri = format!("http://localhost:{port}/auth/callback");

    // 2. generate PKCE + state.
    let (verifier, challenge) = generate_pkce();
    let state = generate_state();

    // 3. build authorize URL & let caller drive the browser.
    let authorize_url = build_authorize_url(
        base, authorize_path, client_id, scope,
        &redirect_uri, &state, &challenge,
    );
    on_authorize_url(&authorize_url);

    // 4. spin up the callback server.
    let (tx, rx) = tokio::sync::oneshot::channel::<CallbackResult>();
    let tx = Arc::new(tokio::sync::Mutex::new(Some(tx)));
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let shutdown_tx = Arc::new(tokio::sync::Mutex::new(Some(shutdown_tx)));

    let expected_state = state.clone();
    let tx_for_handler = tx.clone();
    let shutdown_for_handler = shutdown_tx.clone();
    let app = Router::new().route(
        "/auth/callback",
        get(move |Query(q): Query<CallbackQuery>| {
            let tx_for_handler = tx_for_handler.clone();
            let shutdown_for_handler = shutdown_for_handler.clone();
            let expected_state = expected_state.clone();
            async move {
                let result: CallbackResult = if let Some(err) = q.error {
                    let detail = q.error_description.unwrap_or_default();
                    Err(format!("authorization error: {err} {detail}"))
                } else if q.state.as_deref() != Some(expected_state.as_str()) {
                    Err("authorization state mismatch (possible CSRF)".to_string())
                } else if let Some(code) = q.code {
                    Ok(code)
                } else {
                    Err("authorization callback missing `code`".to_string())
                };

                let ok = result.is_ok();
                if let Some(sender) = tx_for_handler.lock().await.take() {
                    let _ = sender.send(result);
                }
                // Schedule shutdown after we hand the response back.
                let shutdown = shutdown_for_handler.clone();
                tokio::spawn(async move {
                    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                    if let Some(s) = shutdown.lock().await.take() {
                        let _ = s.send(());
                    }
                });

                let body = if ok {
                    "<html><body><h2>Login complete.</h2>\
                        <p>You can close this tab and return to the terminal.</p>\
                        </body></html>"
                } else {
                    "<html><body><h2>Login failed.</h2>\
                        <p>Check the terminal for details.</p>\
                        </body></html>"
                };
                Html(body)
            }
        }),
    );

    let serve = tokio::spawn(async move {
        let _ = axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = shutdown_rx.await;
            })
            .await;
    });

    // 5. wait for the callback (or time out).
    let code = match tokio::time::timeout(PKCE_CALLBACK_TIMEOUT, rx).await {
        Ok(Ok(Ok(code))) => code,
        Ok(Ok(Err(msg))) => {
            // Cause server to shut down too.
            if let Some(s) = shutdown_tx.lock().await.take() {
                let _ = s.send(());
            }
            let _ = serve.await;
            return Err(AppError::Upstream(msg));
        }
        Ok(Err(_)) => {
            if let Some(s) = shutdown_tx.lock().await.take() {
                let _ = s.send(());
            }
            let _ = serve.await;
            return Err(AppError::Upstream(
                "browser callback channel closed unexpectedly".into(),
            ));
        }
        Err(_) => {
            if let Some(s) = shutdown_tx.lock().await.take() {
                let _ = s.send(());
            }
            let _ = serve.await;
            return Err(AppError::Upstream(
                "timed out waiting for browser callback".into(),
            ));
        }
    };
    // ensure the server shuts down before we keep going.
    let _ = serve.await;

    // 6. exchange code for tokens.
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base}{token_path}"))
        .form(&[
            ("grant_type",    "authorization_code"),
            ("code",          code.as_str()),
            ("redirect_uri",  redirect_uri.as_str()),
            ("client_id",     client_id),
            ("code_verifier", verifier.as_str()),
        ])
        .send()
        .await
        .map_err(|e| AppError::Upstream(e.to_string()))?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(AppError::Upstream(format!(
            "token exchange failed {status}: {body}"
        )));
    }
    let tr: TokenResponse = resp
        .json()
        .await
        .map_err(|e| AppError::Upstream(e.to_string()))?;
    let (jwt_account_id, tier) = extract_jwt_claims(&tr.access_token);
    let account_id = tr.account_id.or(jwt_account_id);
    Ok(ChatgptAuth {
        access_token: tr.access_token,
        refresh_token: tr.refresh_token.unwrap_or_default(),
        expires_at: SystemTime::now() + std::time::Duration::from_secs(tr.expires_in),
        account_id,
        tier,
    })
}

pub async fn pkce_login_interactive() -> Result<ChatgptAuth, AppError> {
    let parsed = url::Url::parse(OAUTH_AUTHORIZE_URL)
        .map_err(|e| AppError::Other(e.into()))?;
    let base = format!(
        "{}://{}",
        parsed.scheme(),
        parsed.host_str().unwrap_or("auth.openai.com"),
    );
    let authorize_path = parsed.path().to_string();
    let token_path = url::Url::parse(OAUTH_TOKEN_URL)
        .map(|u| u.path().to_string())
        .unwrap_or_else(|_| "/oauth/token".into());

    pkce_login_at(
        &base,
        &authorize_path,
        &token_path,
        OAUTH_CLIENT_ID,
        OAUTH_SCOPES,
        |url| {
            println!(
                "\nopen this URL in your browser to log in:\n  {url}\n\n\
                 (we will catch the callback on 127.0.0.1)\n",
            );
            let _ = open_browser(url);
        },
        None,
    )
    .await
}

fn open_browser(url: &str) -> std::io::Result<()> {
    let cmd = if cfg!(target_os = "macos") { "open" }
        else if cfg!(target_os = "windows") { "start" }
        else { "xdg-open" };
    std::process::Command::new(cmd).arg(url).spawn().map(|_| ())
}

// ─── handlers ───

#[derive(Clone)]
pub struct AppState {
    pub cfg: Config,
    pub http: reqwest::Client,
    pub upstream_base: String, // ChatGPT, e.g. "https://chatgpt.com"
    /// Base URL of the Gemini upstream (overridden in tests).
    pub gemini_base: String,   // e.g. "https://generativelanguage.googleapis.com"
    pub auth: Arc<tokio::sync::Mutex<Option<AuthState>>>,
    pub auth_path: PathBuf,
    /// Origin (scheme + host) of the OAuth token endpoint. Defaults to the
    /// production issuer; tests point this at a wiremock server.
    pub auth_origin: String,
}

/// Shared slot a handler can write into mid-request to surface model
/// translation info (Anthropic → Codex) to the request-logging middleware.
#[derive(Clone, Default)]
pub struct RequestModels(pub Arc<std::sync::Mutex<ModelPair>>);

#[derive(Default, Debug, Clone)]
pub struct ModelPair {
    pub model_in: String,
    pub model_out: String,
    /// Lowercase tier name (e.g. `"plus"`, `"pro"`, `"unknown"`). Logged
    /// per-request so operators can see which mapping table was applied.
    /// On the Gemini path this is always `"free"`.
    pub tier: String,
    /// Active backend (`"chatgpt"` or `"gemini"`). Empty on requests that
    /// short-circuited before backend dispatch (e.g. /health).
    pub provider: String,
}

pub fn build_app(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/messages", post(messages))
        .route("/v1/messages/count_tokens", post(count_tokens))
        .layer(axum::middleware::from_fn(request_log_middleware))
        .with_state(state)
}

/// Emit one structured log line per request, per spec §10:
/// `method`, `path`, `model_in`, `model_out`, `status`, `duration_ms`, `bytes_out`.
/// `bytes_out` is best-effort — Content-Length if the response set one,
/// otherwise 0 (SSE responses are streamed and we don't instrument the body).
async fn request_log_middleware(
    mut req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    let start = std::time::Instant::now();
    let method = req.method().clone();
    let path = req.uri().path().to_string();

    // Inject a slot for the handler to record model_in/model_out.
    let models = RequestModels::default();
    req.extensions_mut().insert(models.clone());

    let response = next.run(req).await;
    let status = response.status().as_u16();
    let duration_ms = start.elapsed().as_millis() as u64;
    let bytes_out = response
        .headers()
        .get(axum::http::header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);

    let pair = models.0.lock().ok().map(|g| g.clone()).unwrap_or_default();
    tracing::info!(
        method = %method,
        path = %path,
        provider = %pair.provider,
        model_in = %pair.model_in,
        model_out = %pair.model_out,
        tier = %pair.tier,
        status = status,
        duration_ms = duration_ms,
        bytes_out = bytes_out,
        "request",
    );

    response
}

async fn health() -> &'static str { "ok" }

async fn models(State(_state): State<AppState>) -> Json<Value> {
    let now = SystemTime::now().duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64).unwrap_or(0);
    let data: Vec<Value> = KNOWN_CLAUDE_MODELS.iter().map(|name| {
        serde_json::json!({
            "id": name, "type": "model",
            "display_name": name, "created_at": now
        })
    }).collect();
    Json(serde_json::json!({ "data": data, "has_more": false, "first_id": null, "last_id": null }))
}

async fn count_tokens(
    State(_state): State<AppState>,
    AnthropicJson(req): AnthropicJson<AnthropicReq>,
) -> Result<Json<Value>, AppError> {
    let mut chars: usize = req.system.as_ref().map(|s| match s {
        SystemField::Text(t) => t.chars().count(),
        SystemField::Blocks(bs) => bs.iter()
            .map(|SystemBlock::Text { text }| text.chars().count()).sum(),
    }).unwrap_or(0);

    for m in &req.messages {
        match &m.content {
            MessageContent::Text(t) => chars += t.chars().count(),
            MessageContent::Blocks(bs) => for b in bs {
                match b {
                    ContentBlock::Text { text } => chars += text.chars().count(),
                    ContentBlock::ToolUse { input, .. } => chars += input.to_string().chars().count(),
                    ContentBlock::ToolResult { content, .. } => chars += content.to_string().chars().count(),
                    ContentBlock::Image { .. } => chars += 1024, // heuristic per image
                }
            }
        }
    }

    let tokens = ((chars + 3) / 4) as u64; // ~4 chars/token heuristic
    Ok(Json(serde_json::json!({ "input_tokens": tokens })))
}

use axum::response::sse::{Event, KeepAlive, Sse};
use eventsource_stream::Eventsource;
use futures::{Stream, StreamExt, TryStreamExt};
use std::convert::Infallible;

const UPSTREAM_PATH: &str = "/backend-api/codex/responses";

async fn messages(
    State(state): State<AppState>,
    axum::Extension(models): axum::Extension<RequestModels>,
    AnthropicJson(req): AnthropicJson<AnthropicReq>,
) -> Result<Response, AppError> {
    tracing::debug!(
        model = %req.model,
        stream = ?req.stream,
        msgs = req.messages.len(),
        tools = req.tools.as_ref().map(|t| t.len()).unwrap_or(0),
        "incoming /v1/messages"
    );
    // Snapshot which backend is cached and dispatch. We hold the lock only long
    // enough to clone; HTTP I/O happens without it.
    let auth_snapshot = {
        let guard = state.auth.lock().await;
        guard.as_ref().cloned().ok_or(AppError::NotAuthenticated)?
    };

    match auth_snapshot {
        AuthState::Chatgpt(chatgpt) => {
            messages_chatgpt(state, models, req, chatgpt).await
        }
        AuthState::Gemini(gemini) => {
            messages_gemini(state, models, req, gemini).await
        }
    }
}

/// ChatGPT-subscription path: tier-aware Anthropic→Codex translation.
async fn messages_chatgpt(
    state: AppState,
    models: RequestModels,
    req: AnthropicReq,
    chatgpt: ChatgptAuth,
) -> Result<Response, AppError> {
    if chatgpt.tier == ChatGptTier::Free {
        if let Ok(mut g) = models.0.lock() {
            g.model_in = req.model.clone();
            g.tier = chatgpt.tier.as_log_str().to_string();
            g.provider = "chatgpt".to_string();
        }
        return Err(AppError::SubscriptionRequired);
    }
    let tier = chatgpt.tier;
    let model_out = tier.map_model(&req.model).to_string();
    let codex = to_codex(&req, &model_out);
    if let Ok(mut g) = models.0.lock() {
        g.model_in = req.model.clone();
        g.model_out = codex.model.clone();
        g.tier = tier.as_log_str().to_string();
        g.provider = "chatgpt".to_string();
    }
    let pairs = forward_translated_pairs(state, codex, req.model.clone()).await?;
    finalize_response(pairs, req.stream).await
}

/// Gemini path: free Google AI Studio API key, native Gemini wire format.
async fn messages_gemini(
    state: AppState,
    models: RequestModels,
    req: AnthropicReq,
    gemini: GeminiAuth,
) -> Result<Response, AppError> {
    let model_out = gemini.model.clone();
    if let Ok(mut g) = models.0.lock() {
        g.model_in = req.model.clone();
        g.model_out = model_out.clone();
        g.tier = "free".to_string();
        g.provider = "gemini".to_string();
    }
    let pairs = forward_gemini_pairs(
        gemini,
        req.clone(),
        model_out,
        state.gemini_base.clone(),
    ).await?;
    finalize_response(pairs, req.stream).await
}

/// Shared tail: either pin-and-return SSE, or accumulate into a single JSON
/// Message for non-streaming clients. Used by both the ChatGPT and Gemini paths.
async fn finalize_response<S>(pairs: S, stream: Option<bool>) -> Result<Response, AppError>
where
    S: Stream<Item = (String, Value)> + Send + 'static,
{
    if stream != Some(true) {
        let message = collect_message_from_pairs(pairs).await?;
        return Ok(axum::Json(message).into_response());
    }
    // Spec §8: emit a real `event: ping` frame every 15s as keep-alive (the
    // default `.text(...)` produces a `: ping` comment line, which some
    // Anthropic SDK readers reject).
    let events = pairs.map(|(name, data)| {
        Ok::<Event, Infallible>(Event::default().event(name).data(data.to_string()))
    });
    let sse = Sse::new(events).keep_alive(
        KeepAlive::new()
            .interval(std::time::Duration::from_secs(15))
            .event(Event::default().event("ping").data("{}")),
    );
    Ok(sse.into_response())
}

/// Consume a stream of translated Anthropic SSE pairs into a single Message
/// JSON, matching Anthropic's non-streamed /v1/messages response shape.
async fn collect_message_from_pairs<S>(stream: S) -> Result<Value, AppError>
where
    S: Stream<Item = (String, Value)>,
{
    use futures::StreamExt;
    futures::pin_mut!(stream);

    let mut message = serde_json::json!({
        "id": "",
        "type": "message",
        "role": "assistant",
        "model": "",
        "content": Value::Array(Vec::new()),
        "stop_reason": Value::Null,
        "stop_sequence": Value::Null,
        "usage": { "input_tokens": 0, "output_tokens": 0 }
    });
    let mut blocks: Vec<Value> = Vec::new();
    let mut text_buf: Vec<String> = Vec::new();
    let mut json_buf: Vec<String> = Vec::new();

    while let Some((name, data)) = stream.next().await {
        match name.as_str() {
            "message_start" => {
                if let Some(m) = data.get("message") {
                    message["id"] = m.get("id").cloned().unwrap_or_default();
                    message["model"] = m.get("model").cloned().unwrap_or_default();
                    if let Some(role) = m.get("role") {
                        message["role"] = role.clone();
                    }
                }
            }
            "content_block_start" => {
                let block = data.get("content_block").cloned().unwrap_or(Value::Null);
                blocks.push(block);
                text_buf.push(String::new());
                json_buf.push(String::new());
            }
            "content_block_delta" => {
                if blocks.is_empty() { continue; }
                let idx = blocks.len() - 1;
                if let Some(t) = data.pointer("/delta/text").and_then(|v| v.as_str()) {
                    text_buf[idx].push_str(t);
                }
                if let Some(j) = data.pointer("/delta/partial_json").and_then(|v| v.as_str()) {
                    json_buf[idx].push_str(j);
                }
            }
            "content_block_stop" => {
                if blocks.is_empty() { continue; }
                let idx = blocks.len() - 1;
                let block = &mut blocks[idx];
                let bt = block.get("type").and_then(|v| v.as_str()).unwrap_or("").to_string();
                if bt == "text" {
                    block["text"] = Value::String(text_buf[idx].clone());
                } else if bt == "tool_use" {
                    let parsed = serde_json::from_str::<Value>(&json_buf[idx])
                        .unwrap_or(Value::Object(Default::default()));
                    block["input"] = parsed;
                }
            }
            "message_delta" => {
                if let Some(d) = data.get("delta") {
                    if let Some(sr) = d.get("stop_reason") { message["stop_reason"] = sr.clone(); }
                    if let Some(ss) = d.get("stop_sequence") { message["stop_sequence"] = ss.clone(); }
                }
                if let Some(u) = data.get("usage") {
                    message["usage"] = u.clone();
                }
            }
            "message_stop" => break,
            "error" => {
                let msg = data.pointer("/error/message").and_then(|v| v.as_str())
                    .unwrap_or("upstream error").to_string();
                return Err(AppError::Upstream(msg));
            }
            _ => {}
        }
    }

    message["content"] = Value::Array(blocks);
    Ok(message)
}

async fn current_access_token(state: &AppState) -> Result<String, AppError> {
    // Snapshot under the lock, drop the guard, then refresh outside the lock
    // so concurrent requests don't serialize on a Mutex held across HTTP I/O.
    // Only the ChatGPT path needs a Bearer token — Gemini uses an api-key
    // query parameter and never reaches this function.
    let snapshot = {
        let guard = state.auth.lock().await;
        match guard.as_ref().ok_or(AppError::NotAuthenticated)? {
            AuthState::Chatgpt(c) => c.clone(),
            AuthState::Gemini(_) => return Err(AppError::Other(anyhow::anyhow!(
                "internal: ChatGPT token requested while Gemini backend is active",
            ))),
        }
    };
    if snapshot.is_fresh(std::time::Duration::from_secs(60)) {
        return Ok(snapshot.access_token);
    }
    let refreshed = refresh_with_origin(&snapshot, &state.auth_origin).await?;
    AuthState::Chatgpt(refreshed.clone()).save_to(&state.auth_path).ok();
    let mut guard = state.auth.lock().await;
    *guard = Some(AuthState::Chatgpt(refreshed.clone()));
    Ok(refreshed.access_token)
}

async fn forward_translated_pairs(
    state: AppState,
    codex: CodexReq,
    model_in: String,
) -> Result<impl Stream<Item = (String, Value)>, AppError> {
    let mut attempt = 0u8;
    let resp = loop {
        attempt += 1;
        let token = current_access_token(&state).await?;
        // Account-id header. Required by chatgpt.com/backend-api/codex/responses.
        // Prefer the AuthState cache; fall back to decoding the JWT in-flight.
        let account_id = {
            let guard = state.auth.lock().await;
            guard.as_ref()
                .and_then(|a| a.as_chatgpt())
                .and_then(|c| c.account_id.clone())
                .or_else(|| extract_account_id_from_jwt(&token))
                .unwrap_or_default()
        };
        let mut req = state.http
            .post(format!("{}{UPSTREAM_PATH}", state.upstream_base))
            .header("authorization", format!("Bearer {token}"))
            .header("accept", "text/event-stream")
            .header("originator", "codex_cli_rs")
            .header("openai-beta", "responses=experimental")
            .json(&codex);
        if !account_id.is_empty() {
            req = req.header("chatgpt-account-id", account_id);
        }
        let r = req.send()
            .await
            .map_err(|e| AppError::Upstream(e.to_string()))?;
        match r.status() {
            s if s.is_success() => break r,
            reqwest::StatusCode::UNAUTHORIZED if attempt == 1 => {
                // Force refresh, then retry. Snapshot+drop the guard before
                // awaiting refresh so we don't hold the Mutex across HTTP I/O.
                let snapshot = {
                    let guard = state.auth.lock().await;
                    guard.as_ref().and_then(|a| a.as_chatgpt()).cloned()
                };
                if let Some(c) = snapshot {
                    let new = refresh_with_origin(&c, &state.auth_origin).await?;
                    let wrapped = AuthState::Chatgpt(new);
                    wrapped.save_to(&state.auth_path).ok();
                    let mut guard = state.auth.lock().await;
                    *guard = Some(wrapped);
                }
                continue;
            }
            reqwest::StatusCode::TOO_MANY_REQUESTS => {
                let retry_after = r.headers().get("retry-after")
                    .and_then(|v| v.to_str().ok()).map(String::from);
                return Err(AppError::RateLimit { retry_after });
            }
            s => {
                let body = r.text().await.unwrap_or_default();
                return Err(AppError::Upstream(format!("upstream {s}: {body}")));
            }
        }
    };

    // model_in/model_out flow to the request-log middleware via RequestModels.
    let _ = model_in; // parameter kept for call-site clarity
    let bytes = resp.bytes_stream()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e));

    let translator = Translator::new(codex.model.clone());
    let events = bytes.eventsource().scan(
        (translator, /*done=*/ false),
        |(translator, done), item| {
            // Once we've emitted a terminal error frame we stop producing events.
            if *done {
                return std::future::ready(None);
            }
            let pairs: Vec<(String, Value)> = match item {
                Ok(raw) => {
                    tracing::trace!(event = %raw.event, "codex SSE");
                    match serde_json::from_str::<CodexEvent>(&raw.data) {
                        Ok(CodexEvent::Other) => Vec::new(),
                        Ok(ev) => translator.next(ev),
                        Err(e) => {
                            tracing::error!(err = %e, data = %raw.data, "codex SSE parse failure");
                            *done = true;
                            upstream_error_frames(format!("upstream SSE parse error: {e}"))
                        }
                    }
                }
                Err(e) => {
                    tracing::error!(err = %e, "codex SSE stream failure");
                    *done = true;
                    upstream_error_frames(format!("upstream SSE stream error: {e}"))
                }
            };
            std::future::ready(Some(pairs))
        },
    )
    .flat_map(|pairs| futures::stream::iter(pairs.into_iter()));

    Ok(events)
}

/// Stream a Gemini `streamGenerateContent` SSE response and emit Anthropic
/// SSE pairs. Mirrors `forward_translated_pairs` for the Gemini path.
///
/// The translator runs on a spawned task that owns the upstream stream and
/// pushes pairs into an mpsc channel; the returned Stream is the channel's
/// Receiver. This gives us a single owner for the GeminiTranslator (so we can
/// call `next()` on each frame and `finish()` once on EOF) without needing
/// `async-stream`.
async fn forward_gemini_pairs(
    gemini: GeminiAuth,
    req: AnthropicReq,
    model_out: String,
    base: String,
) -> Result<impl Stream<Item = (String, Value)>, AppError> {
    let body = to_gemini(&req).map_err(AppError::InvalidRequest)?;

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .map_err(|e| AppError::Other(anyhow::anyhow!(e)))?;

    let url = format!(
        "{base}/v1beta/models/{model}:streamGenerateContent",
        model = model_out
    );

    let resp = client
        .post(&url)
        .query(&[("alt", "sse"), ("key", gemini.api_key.as_str())])
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| AppError::Upstream(e.to_string()))?;

    match resp.status() {
        s if s.is_success() => {}
        reqwest::StatusCode::UNAUTHORIZED | reqwest::StatusCode::FORBIDDEN => {
            let body = resp.text().await.unwrap_or_default();
            return Err(AppError::Upstream(format!("gemini auth rejected: {body}")));
        }
        reqwest::StatusCode::TOO_MANY_REQUESTS => {
            let retry_after = resp.headers().get("retry-after")
                .and_then(|v| v.to_str().ok()).map(String::from);
            return Err(AppError::RateLimit { retry_after });
        }
        s => {
            let body = resp.text().await.unwrap_or_default();
            return Err(AppError::Upstream(format!("gemini upstream {s}: {body}")));
        }
    }

    let (mut tx, rx) = futures::channel::mpsc::unbounded::<(String, Value)>();

    tokio::spawn(async move {
        use futures::SinkExt;
        let bytes = resp.bytes_stream()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e));
        let mut sse = bytes.eventsource();
        let mut translator = GeminiTranslator::new(model_out);
        let mut errored = false;

        while let Some(item) = sse.next().await {
            let pairs: Vec<(String, Value)> = match item {
                Ok(raw) => {
                    tracing::trace!(event = %raw.event, "gemini SSE");
                    if raw.data.trim() == "[DONE]" { break; }
                    match serde_json::from_str::<GeminiStreamFrame>(&raw.data) {
                        Ok(frame) => translator.next(frame),
                        Err(e) => {
                            tracing::error!(err = %e, data = %raw.data,
                                "gemini SSE parse failure");
                            errored = true;
                            upstream_error_frames(format!("gemini SSE parse error: {e}"))
                        }
                    }
                }
                Err(e) => {
                    tracing::error!(err = %e, "gemini SSE stream failure");
                    errored = true;
                    upstream_error_frames(format!("gemini SSE stream error: {e}"))
                }
            };
            for p in pairs {
                if tx.send(p).await.is_err() { return; }
            }
            if errored { return; }
        }

        // Normal EOF — emit closing message_delta + message_stop.
        for p in translator.finish() {
            if tx.send(p).await.is_err() { return; }
        }
    });

    Ok(rx)
}

/// Produce a terminal pair of Anthropic SSE frames for an in-stream failure:
/// an `error` event matching spec §10, followed by `message_stop` so clients
/// close their reader cleanly instead of waiting on a half-open stream.
fn upstream_error_frames(message: String) -> Vec<(String, Value)> {
    vec![
        (
            "error".into(),
            serde_json::json!({
                "type": "error",
                "error": { "type": "api_error", "message": message }
            }),
        ),
        (
            "message_stop".into(),
            serde_json::json!({ "type": "message_stop" }),
        ),
    ]
}

// ─── service: per-OS autostart ───

#[derive(Debug, Clone, Copy)]
pub enum ServiceAction { Install, Uninstall }

pub fn service_apply(binary: &Path, port: u16, action: ServiceAction) -> anyhow::Result<()> {
    #[cfg(target_os = "macos")]
    { return service_apply_macos(binary, port, action); }
    #[cfg(target_os = "linux")]
    { return service_apply_linux(binary, port, action); }
    #[cfg(target_os = "windows")]
    { return service_apply_windows(binary, port, action); }
    #[allow(unreachable_code)]
    { anyhow::bail!("unsupported platform"); }
}

// ── macOS (launchd) ──

#[cfg(target_os = "macos")]
fn plist_path() -> anyhow::Result<PathBuf> {
    directories::BaseDirs::new()
        .map(|d| d.home_dir().join("Library/LaunchAgents/com.claudeg.proxy.plist"))
        .ok_or_else(|| anyhow::anyhow!("could not resolve user home directory"))
}

pub fn build_launchd_plist(binary: &str, port: u16) -> String {
    format!(r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.claudeg.proxy</string>
    <key>ProgramArguments</key>
    <array>
        <string>{binary}</string>
        <string>serve</string>
        <string>--port</string>
        <string>{port}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/claudeg.out.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/claudeg.err.log</string>
</dict>
</plist>
"#)
}

#[cfg(target_os = "macos")]
fn service_apply_macos(binary: &Path, port: u16, action: ServiceAction) -> anyhow::Result<()> {
    let plist = plist_path()?;
    match action {
        ServiceAction::Install => {
            let body = build_launchd_plist(&binary.display().to_string(), port);
            if let Some(p) = plist.parent() { std::fs::create_dir_all(p)?; }
            std::fs::write(&plist, body)?;
            let plist_str = plist.to_str().ok_or_else(|| {
                anyhow::anyhow!("plist path contains non-UTF8 bytes: {}", plist.display())
            })?;
            let _ = std::process::Command::new("launchctl")
                .args(["bootout", &format!("gui/{}/com.claudeg.proxy", uid_string())])
                .output();
            let out = std::process::Command::new("launchctl")
                .args(["bootstrap", &format!("gui/{}", uid_string()), plist_str])
                .output()?;
            if !out.status.success() {
                let stderr = String::from_utf8_lossy(&out.stderr);
                anyhow::bail!("launchctl bootstrap failed: {stderr}");
            }
        }
        ServiceAction::Uninstall => {
            let _ = std::process::Command::new("launchctl")
                .args(["bootout", &format!("gui/{}/com.claudeg.proxy", uid_string())])
                .output();
            let _ = std::fs::remove_file(&plist);
        }
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn uid_string() -> String {
    // Best effort. `id -u` is universally present.
    std::process::Command::new("id").arg("-u")
        .output().ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "501".into())
}

// ── Linux (systemd --user) ──

pub fn build_systemd_unit(binary: &str, port: u16) -> String {
    format!(r#"[Unit]
Description=claudeg Anthropic→Codex proxy
After=network-online.target

[Service]
ExecStart={binary} serve --port {port}
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
"#)
}

#[cfg(target_os = "linux")]
fn systemd_unit_path() -> anyhow::Result<PathBuf> {
    directories::BaseDirs::new()
        .map(|d| d.home_dir().join(".config/systemd/user/claudeg.service"))
        .ok_or_else(|| anyhow::anyhow!("could not resolve user home directory"))
}

#[cfg(target_os = "linux")]
fn service_apply_linux(binary: &Path, port: u16, action: ServiceAction) -> anyhow::Result<()> {
    let unit_path = systemd_unit_path()?;
    match action {
        ServiceAction::Install => {
            let body = build_systemd_unit(&binary.display().to_string(), port);
            if let Some(p) = unit_path.parent() { std::fs::create_dir_all(p)?; }
            std::fs::write(&unit_path, body)?;
            let _ = std::process::Command::new("systemctl")
                .args(["--user", "daemon-reload"]).status();
            let st = std::process::Command::new("systemctl")
                .args(["--user", "enable", "--now", "claudeg.service"]).status();
            if !st.map(|s| s.success()).unwrap_or(false) {
                anyhow::bail!("systemctl --user enable --now claudeg.service failed. \
                    If you have no systemd user session, run `{} serve` manually.",
                    binary.display());
            }
        }
        ServiceAction::Uninstall => {
            let _ = std::process::Command::new("systemctl")
                .args(["--user", "disable", "--now", "claudeg.service"]).status();
            let _ = std::fs::remove_file(&unit_path);
            let _ = std::process::Command::new("systemctl")
                .args(["--user", "daemon-reload"]).status();
        }
    }
    Ok(())
}

// ── Windows (Startup folder) ──

pub fn build_windows_startup_cmd(binary: &str, port: u16) -> String {
    // /B avoids opening a console window.
    format!("@echo off\r\nstart \"\" /B \"{binary}\" serve --port {port}\r\n")
}

#[cfg(target_os = "windows")]
fn startup_cmd_path() -> PathBuf {
    let appdata = std::env::var_os("APPDATA").unwrap_or_default();
    PathBuf::from(appdata).join("Microsoft/Windows/Start Menu/Programs/Startup/claudeg.cmd")
}

#[cfg(target_os = "windows")]
fn service_apply_windows(binary: &Path, port: u16, action: ServiceAction) -> anyhow::Result<()> {
    let cmd_path = startup_cmd_path();
    match action {
        ServiceAction::Install => {
            let body = build_windows_startup_cmd(&binary.display().to_string(), port);
            if let Some(p) = cmd_path.parent() { std::fs::create_dir_all(p)?; }
            std::fs::write(&cmd_path, body)?;
            let binary_str = binary.to_str().ok_or_else(|| {
                anyhow::anyhow!("binary path contains non-UTF8 bytes: {}", binary.display())
            })?;
            // Start it for the current session too.
            std::process::Command::new("cmd")
                .args(["/C", "start", "", "/B", binary_str,
                       "serve", "--port", &port.to_string()])
                .spawn()?;
        }
        ServiceAction::Uninstall => {
            let _ = std::fs::remove_file(&cmd_path);
        }
    }
    Ok(())
}

// ─── claude code settings ───

pub fn claude_settings_path() -> PathBuf {
    directories::BaseDirs::new()
        .map(|d| d.home_dir().join(".claude").join("settings.json"))
        .unwrap_or_else(|| PathBuf::from(".claude/settings.json"))
}

/// Merge our env keys into Claude Code's settings.json.
/// Returns Some(backup_path) if a pre-existing file was backed up.
pub fn configure_claude_settings(path: &Path, port: u16) -> anyhow::Result<Option<PathBuf>> {
    let backup = if path.exists() {
        let bak = path.with_extension("json.claudeg-bak");
        std::fs::copy(path, &bak)?;
        Some(bak)
    } else {
        None
    };

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut root: Value = if path.exists() {
        let s = std::fs::read_to_string(path)?;
        serde_json::from_str(&s).unwrap_or_else(|_| Value::Object(Default::default()))
    } else {
        Value::Object(Default::default())
    };

    if !root.is_object() {
        root = Value::Object(Default::default());
    }

    let env = root.as_object_mut().unwrap()
        .entry("env".to_string())
        .or_insert_with(|| Value::Object(Default::default()));
    if !env.is_object() {
        *env = Value::Object(Default::default());
    }
    let env = env.as_object_mut().unwrap();
    env.insert("ANTHROPIC_BASE_URL".into(), Value::String(format!("http://127.0.0.1:{port}")));
    env.insert("ANTHROPIC_API_KEY".into(),  Value::String("_".into()));

    let pretty = serde_json::to_string_pretty(&root)?;
    std::fs::write(path, pretty)?;
    Ok(backup)
}

pub fn unconfigure_claude_settings(path: &Path) -> anyhow::Result<()> {
    if !path.exists() { return Ok(()); }
    let s = std::fs::read_to_string(path)?;
    let mut root: Value = match serde_json::from_str(&s) {
        Ok(v) => v,
        Err(_) => return Ok(()),
    };
    if let Some(env) = root.get_mut("env").and_then(|v| v.as_object_mut()) {
        env.remove("ANTHROPIC_BASE_URL");
        env.remove("ANTHROPIC_API_KEY");
    }
    let pretty = serde_json::to_string_pretty(&root)?;
    std::fs::write(path, pretty)?;
    Ok(())
}

// ─── default-action: run `claude` through the proxy ───

/// Sentinel `ANTHROPIC_API_KEY` value the `Run` default-action sets when
/// launching `claude`. The proxy ignores its content; the value's last 20
/// characters (`claudeg-routing-proxy`) are pre-approved in `~/.claude.json`
/// by `claudeg setup` so Claude Code doesn't prompt on every invocation.
// Claude Code matches API keys by their last 20 characters, so the sentinel
// and the tail must agree precisely. "claudeg-routing-prox" is 20 chars; the
// dropped "y" off "proxy" keeps the tail readable while satisfying the length.
pub const SENTINEL_API_KEY: &str = "sk-ant-claudeg-routing-prox";
pub const WRAPPER_API_KEY_TAIL: &str = "claudeg-routing-prox";

pub fn claude_json_path() -> PathBuf {
    directories::BaseDirs::new()
        .map(|d| d.home_dir().join(".claude.json"))
        .unwrap_or_else(|| PathBuf::from(".claude.json"))
}

/// Add our sentinel tail to `customApiKeyResponses.approved` in `~/.claude.json`
/// so Claude Code doesn't prompt on every `claudeg "..."` invocation.
/// Idempotent: a no-op if the entry is already present.
pub fn approve_wrapper_api_key(claude_json: &Path) -> anyhow::Result<bool> {
    let mut root: Value = if claude_json.exists() {
        let s = std::fs::read_to_string(claude_json)?;
        serde_json::from_str(&s).unwrap_or_else(|_| Value::Object(Default::default()))
    } else {
        Value::Object(Default::default())
    };
    if !root.is_object() { root = Value::Object(Default::default()); }

    let sec = root.as_object_mut().unwrap()
        .entry("customApiKeyResponses".to_string())
        .or_insert_with(|| serde_json::json!({ "approved": [], "rejected": [] }));
    if !sec.is_object() { *sec = serde_json::json!({ "approved": [], "rejected": [] }); }

    let approved = sec.as_object_mut().unwrap()
        .entry("approved".to_string())
        .or_insert_with(|| Value::Array(Vec::new()));
    if !approved.is_array() { *approved = Value::Array(Vec::new()); }

    let arr = approved.as_array_mut().unwrap();
    let already = arr.iter().any(|v| v.as_str() == Some(WRAPPER_API_KEY_TAIL));
    if already { return Ok(false); }
    arr.push(Value::String(WRAPPER_API_KEY_TAIL.to_string()));

    if let Some(parent) = claude_json.parent() { std::fs::create_dir_all(parent)?; }
    let pretty = serde_json::to_string_pretty(&root)?;
    std::fs::write(claude_json, pretty)?;
    Ok(true)
}

/// Ensure the proxy is running on `port`, then exec `claude` with `args`,
/// inheriting the current terminal. This is the default action invoked when
/// the user runs `claudeg "<prompt>"` (or any args clap doesn't recognize as
/// a subcommand). Returns Err if `claude` itself is not on PATH.
pub async fn run_proxied_claude(args: &[String], port: u16) -> anyhow::Result<()> {
    let base = format!("http://127.0.0.1:{port}");
    let client = reqwest::Client::new();

    async fn ping(c: &reqwest::Client, url: &str) -> bool {
        c.get(url).send().await.map(|r| r.status().is_success()).unwrap_or(false)
    }

    if !ping(&client, &format!("{base}/health")).await {
        // Autostart proxy in a detached background process.
        let bin = std::env::current_exe()?;
        std::process::Command::new(&bin)
            .arg("serve")
            .arg("--port").arg(port.to_string())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .stdin(std::process::Stdio::null())
            .spawn()?;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            if ping(&client, &format!("{base}/health")).await { break; }
            if std::time::Instant::now() > deadline {
                anyhow::bail!("proxy did not start within 5s on {base}");
            }
            tokio::time::sleep(std::time::Duration::from_millis(150)).await;
        }
    }

    let mut cmd = std::process::Command::new("claude");
    cmd.env("ANTHROPIC_BASE_URL", &base);
    cmd.env("ANTHROPIC_API_KEY", SENTINEL_API_KEY);
    cmd.args(args);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        let err = cmd.exec();
        anyhow::bail!("could not exec `claude`: {err}. Is Claude Code installed and on PATH?");
    }
    #[cfg(not(unix))]
    {
        let status = cmd.status()
            .map_err(|e| anyhow::anyhow!("could not spawn `claude`: {e}. Is Claude Code installed and on PATH?"))?;
        std::process::exit(status.code().unwrap_or(1));
    }
}

// ─── main + CLI ───
use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(
    name = "claudeg",
    version,
    about = "Run Claude Code through a ChatGPT subscription or a free Gemini 3 Flash key.",
    long_about = "Run `claudeg \"<prompt>\"` to launch Claude Code routed through the local proxy.\n\
                  Management subcommands: login, logout, whoami, setup, serve, uninstall."
)]
pub struct Cli {
    #[command(subcommand)]
    pub cmd: Option<Cmd>,
}

#[derive(Subcommand, Debug)]
pub enum Cmd {
    /// One-step: login + pre-approve sentinel + register background service.
    Setup {
        /// Skip the login step (re-run config + service only).
        #[arg(long)] skip_login: bool,
        /// Listen port for the proxy.
        #[arg(long, default_value_t = 4000)] port: u16,
        /// Force the ChatGPT subscription backend (skip the picker).
        #[arg(long, conflicts_with_all = ["gemini", "gemini_key"])]
        chatgpt: bool,
        /// Force the Gemini 3 Flash backend (skip the picker).
        #[arg(long, conflicts_with = "chatgpt")]
        gemini: bool,
        /// Non-interactive: use this Gemini API key directly (implies --gemini).
        /// Useful for CI and scripts.
        #[arg(long, value_name = "KEY", conflicts_with = "chatgpt")]
        gemini_key: Option<String>,
    },
    /// Authenticate against a backend. With no flag, shows an interactive picker
    /// (ChatGPT subscription vs. Gemini 3 Flash). Use --chatgpt or --gemini to
    /// skip the picker.
    Login {
        /// Force the ChatGPT subscription backend (browser OAuth).
        #[arg(long, conflicts_with_all = ["gemini", "gemini_key"])]
        chatgpt: bool,
        /// Force the Gemini 3 Flash backend (prompt for API key).
        #[arg(long, conflicts_with = "chatgpt")]
        gemini: bool,
        /// Non-interactive: use this Gemini API key directly (implies --gemini).
        #[arg(long, value_name = "KEY", conflicts_with = "chatgpt")]
        gemini_key: Option<String>,
    },
    /// Forget the cached credentials and stop any running proxy.
    /// Useful for swapping backends or accounts.
    Logout,
    /// Show login status (which backend is active + model mapping).
    Whoami,
    /// Run the proxy in the foreground (advanced; the default action autostarts it).
    Serve {
        #[arg(long)] port: Option<u16>,
    },
    /// Remove service, undo Claude Code config edits.
    Uninstall {
        /// Also delete the cached credentials.
        #[arg(long)] purge_auth: bool,
    },
    /// Default action: forward all subsequent args to `claude`, autostarting
    /// the proxy first. Captured by clap as an external subcommand so
    /// `claudeg "list files"` and `claudeg --help-claude` both flow here.
    #[command(external_subcommand)]
    Run(Vec<String>),
}

/// User's backend choice as collected from CLI flags or the interactive picker.
#[derive(Debug, Clone)]
enum BackendChoice {
    Chatgpt,
    /// `Some(key)` skips the prompt; `None` triggers the hidden-input prompt.
    Gemini(Option<String>),
}

impl BackendChoice {
    /// Resolve flags into a choice, falling back to the interactive picker if
    /// no flag was supplied. Used by both `Login` and `Setup`.
    fn from_flags_or_prompt(
        chatgpt: bool,
        gemini: bool,
        gemini_key: Option<String>,
    ) -> anyhow::Result<Self> {
        if chatgpt { return Ok(Self::Chatgpt); }
        if gemini_key.is_some() || gemini { return Ok(Self::Gemini(gemini_key)); }
        Ok(prompt_backend_choice()?)
    }
}

/// Interactive backend picker. Reads a one-character choice from stdin.
fn prompt_backend_choice() -> anyhow::Result<BackendChoice> {
    use std::io::Write;
    println!();
    println!("How do you want to back claudeg?");
    println!();
    println!("  1) ChatGPT subscription   Plus/Pro/Business/Enterprise · browser OAuth");
    println!("  2) Gemini 3 Flash         Free · paste a Google AI Studio API key");
    println!();
    for _ in 0..3 {
        print!("Choose [1-2] (default 1): ");
        std::io::stdout().flush().ok();
        let mut line = String::new();
        std::io::stdin().read_line(&mut line)?;
        match line.trim() {
            "" | "1" | "chatgpt" | "ChatGPT" => return Ok(BackendChoice::Chatgpt),
            "2" | "gemini" | "Gemini" => return Ok(BackendChoice::Gemini(None)),
            other => eprintln!("(unrecognized choice `{other}` — type 1 or 2)"),
        }
    }
    anyhow::bail!("no valid backend chosen after 3 attempts")
}

/// Acquire a Gemini API key — either the one passed via `--gemini-key`/CLI,
/// or by prompting the user. Validates against the upstream `models` endpoint
/// before returning. On a network error the key is accepted with a warning
/// so offline setups still work.
async fn acquire_gemini_key(initial: Option<String>) -> anyhow::Result<GeminiAuth> {
    let key = match initial {
        Some(k) => {
            let trimmed = k.trim().to_string();
            if trimmed.is_empty() {
                anyhow::bail!("--gemini-key was empty");
            }
            trimmed
        }
        None => {
            println!();
            println!("Get a free Gemini API key at https://aistudio.google.com/apikey");
            println!("(Free tier: ~5–15 requests/minute, 100–1000 requests/day)");
            println!();
            let entered = rpassword::prompt_password("Paste API key (input hidden): ")?;
            let trimmed = entered.trim().to_string();
            if trimmed.is_empty() {
                anyhow::bail!("no API key entered");
            }
            trimmed
        }
    };

    // Validate. A 200 OK confirms the key is alive; 401/403 reject; anything
    // else (network error, 5xx) we treat as inconclusive and accept with a
    // warning so offline first-run works.
    let model = default_gemini_model();
    let url = format!("https://generativelanguage.googleapis.com/v1beta/models/{model}");
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()?;
    match client.get(&url).query(&[("key", key.as_str())]).send().await {
        Ok(r) if r.status().is_success() => {
            println!("✓ Gemini API key validated against {model}");
        }
        Ok(r) if r.status() == reqwest::StatusCode::UNAUTHORIZED
              || r.status() == reqwest::StatusCode::FORBIDDEN => {
            anyhow::bail!(
                "Gemini API rejected the key ({}). Double-check at https://aistudio.google.com/apikey",
                r.status()
            );
        }
        Ok(r) => {
            eprintln!(
                "warning: Gemini validation returned HTTP {} — saving anyway",
                r.status()
            );
        }
        Err(e) => {
            eprintln!("warning: could not reach Gemini ({e}) — saving the key anyway");
        }
    }

    Ok(GeminiAuth { api_key: key, model })
}

/// Drive the user through the chosen login flow, persist to `auth.json`, and
/// return the saved AuthState. Shared by `Cmd::Login` and `Cmd::Setup`.
async fn run_login_flow(choice: BackendChoice) -> anyhow::Result<AuthState> {
    let state = match choice {
        BackendChoice::Chatgpt => {
            let c = pkce_login_interactive().await
                .map_err(|e| anyhow::anyhow!(e.to_string()))?;
            AuthState::Chatgpt(c)
        }
        BackendChoice::Gemini(initial) => {
            let g = acquire_gemini_key(initial).await?;
            AuthState::Gemini(g)
        }
    };
    state.save_to(&AuthState::default_path())?;
    Ok(state)
}

/// Best-effort: kill any background `claudeg serve` process on this machine.
/// Used by `logout` and `uninstall` so users don't have to manually pkill.
fn stop_running_proxy_processes() {
    #[cfg(unix)]
    {
        let _ = std::process::Command::new("pkill")
            .args(["-f", "claudeg serve"])
            .output();
    }
    #[cfg(windows)]
    {
        let script = "Get-CimInstance Win32_Process -Filter \"Name='claudeg.exe'\" | \
                      Where-Object { $_.CommandLine -match 'serve' } | \
                      ForEach-Object { Stop-Process -Id $_.ProcessId -Force }";
        let _ = std::process::Command::new("powershell")
            .args(["-NoProfile", "-Command", script])
            .output();
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "claudeg=info".into()),
        )
        .init();

    let cli = Cli::parse();
    let cmd = cli.cmd.unwrap_or_else(|| Cmd::Run(Vec::new()));
    match cmd {
        Cmd::Setup { skip_login, port, chatgpt, gemini, gemini_key } => {
            // 1. login (unless --skip-login)
            let active = if !skip_login {
                let choice = BackendChoice::from_flags_or_prompt(chatgpt, gemini, gemini_key)?;
                let state = run_login_flow(choice).await?;
                println!("✓ logged in ({})", state.provider_name());
                Some(state)
            } else {
                println!("(skipped login)");
                AuthState::load_default()
            };

            let bin = std::env::current_exe()?;

            // 2. pre-approve our sentinel API key so Claude Code doesn't prompt
            let cj = claude_json_path();
            match approve_wrapper_api_key(&cj) {
                Ok(true)  => println!("✓ approved wrapper sentinel in {}", cj.display()),
                Ok(false) => println!("✓ wrapper sentinel already approved in {}", cj.display()),
                Err(e)    => eprintln!("warning: could not patch {}: {e}", cj.display()),
            }

            // 3. defensively clean up any legacy global redirect from earlier versions
            let settings = claude_settings_path();
            let _ = unconfigure_claude_settings(&settings);

            // 4. install background service
            service_apply(&bin, port, ServiceAction::Install)?;
            println!("✓ service installed (auto-starts at login)");

            // 5. health probe
            let url = format!("http://127.0.0.1:{port}/health");
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            let client = reqwest::Client::new();
            loop {
                if let Ok(r) = client.get(&url).send().await {
                    if r.status().is_success() {
                        println!("✓ proxy responding at {url}");
                        break;
                    }
                }
                if std::time::Instant::now() > deadline {
                    eprintln!("warning: proxy not responding at {url} yet \
                        (service may still be starting)");
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(300)).await;
            }

            println!("\nAll set:");
            println!("  claude              → real Anthropic API (unchanged)");
            let backend_label = match active.as_ref().map(|s| s.provider_name()) {
                Some("gemini") => "the Gemini 3 Flash API",
                Some("chatgpt") => "your ChatGPT subscription",
                _ => "the configured backend",
            };
            println!("  claudeg \"prompt\"    → routed through {backend_label}");
        }
        Cmd::Logout => {
            stop_running_proxy_processes();
            let auth = AuthState::default_path();
            // Note before removal which backend was active, so the follow-up
            // hint is specific to what the user just disconnected.
            let prior = AuthState::load_default();
            match std::fs::remove_file(&auth) {
                Ok(_)  => println!("✓ removed cached credentials at {}", auth.display()),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound =>
                    println!("(already logged out — no credentials cached)"),
                Err(e) => eprintln!("warning: could not remove {}: {e}", auth.display()),
            }
            match prior.as_ref().map(|s| s.provider_name()) {
                Some("gemini") => {
                    println!("\nTo finish switching:");
                    println!("  1. (Optional) Revoke the old key at https://aistudio.google.com/apikey");
                    println!("  2. Run `claudeg login` to choose a backend and authenticate");
                }
                _ => {
                    println!("\nTo finish switching:");
                    println!("  1. Sign out of chatgpt.com in your browser (or use a private window)");
                    println!("  2. Run `claudeg login` to choose a backend and authenticate");
                }
            }
        }
        Cmd::Uninstall { purge_auth } => {
            let bin = std::env::current_exe()?;
            stop_running_proxy_processes();
            let _ = service_apply(&bin, 4000, ServiceAction::Uninstall);
            println!("✓ service removed");

            // Also undo any legacy global redirect that older versions installed.
            let settings = claude_settings_path();
            let _ = unconfigure_claude_settings(&settings);

            if purge_auth {
                let _ = std::fs::remove_file(AuthState::default_path());
                println!("✓ removed cached credentials");
            }

            println!("\nThe `claudeg` binary itself is still on disk — remove it manually if you like.");
        }
        Cmd::Run(args) => {
            run_proxied_claude(&args, 4000).await?;
        }
        Cmd::Login { chatgpt, gemini, gemini_key } => {
            let choice = BackendChoice::from_flags_or_prompt(chatgpt, gemini, gemini_key)?;
            let state = run_login_flow(choice).await?;
            println!(
                "✓ logged in ({}). credentials cached at {}",
                state.provider_name(),
                AuthState::default_path().display()
            );
        }
        Cmd::Whoami => match AuthState::load_default() {
            Some(s) => print_whoami(&s),
            None => println!("not logged in. run `claudeg login`."),
        },
        Cmd::Serve { port } => {
            let mut cfg = Config::load_or_default();
            if let Some(p) = port {
                cfg.listen = format!("127.0.0.1:{p}");
            }
            let listen = cfg.listen.clone();
            let auth = AuthState::load_default();
            if auth.is_none() {
                eprintln!("warning: not logged in. run `claudeg login` first.");
            }
            let state = AppState {
                cfg,
                http: reqwest::Client::builder()
                    .timeout(std::time::Duration::from_secs(600))
                    .build()?,
                upstream_base: "https://chatgpt.com".into(),
                gemini_base: "https://generativelanguage.googleapis.com".into(),
                auth: Arc::new(tokio::sync::Mutex::new(auth)),
                auth_path: AuthState::default_path(),
                auth_origin: default_auth_origin()
                    .unwrap_or_else(|_| "https://auth.openai.com".into()),
            };
            let app = build_app(state);
            let listener = tokio::net::TcpListener::bind(&listen).await?;
            tracing::info!(%listen, "claudeg listening");
            axum::serve(listener, app).await?;
        }
    }
    Ok(())
}

/// Render the `claudeg whoami` output for a loaded `AuthState`.
fn print_whoami(s: &AuthState) {
    match s {
        AuthState::Chatgpt(c) => print_whoami_chatgpt(c),
        AuthState::Gemini(g)  => print_whoami_gemini(g),
    }
}

fn print_whoami_chatgpt(c: &ChatgptAuth) {
    let rem = c.expires_at.duration_since(SystemTime::now()).ok();
    let expires = match rem {
        Some(d) => format!("expires in {}s", d.as_secs()),
        None    => "token expired — refresh on next call".to_string(),
    };

    println!("backend: ChatGPT subscription");
    match c.tier {
        ChatGptTier::Free => {
            println!(
                "logged in as Free — claudeg requires Go/Plus/Pro tier or higher. \
                 Upgrade at https://chatgpt.com/pricing"
            );
            println!("({expires})");
        }
        ChatGptTier::Unknown => {
            println!(
                "logged in (tier unknown — JWT missing chatgpt_plan_type, treating as Plus) \
                 ({expires})"
            );
            print_chatgpt_mapping_table(ChatGptTier::Unknown);
        }
        tier => {
            println!("logged in as {} ({})", tier.display_name(), expires);
            print_chatgpt_mapping_table(tier);
        }
    }
}

fn print_chatgpt_mapping_table(tier: ChatGptTier) {
    println!();
    println!("auto-mapping (detected from chatgpt_plan_type):");
    println!("  claude-opus-4-7      → {}", tier.map_model("claude-opus-4-7"));
    println!("  claude-sonnet-4-6    → {}", tier.map_model("claude-sonnet-4-6"));
    println!("  claude-haiku-4-5     → {}", tier.map_model("claude-haiku-4-5"));
    println!("  (unmapped Claude model → {})", tier.default_fallback_model());
}

fn print_whoami_gemini(g: &GeminiAuth) {
    println!("backend: Gemini 3 Flash (free)");
    println!("key cached at {} (mode 0600)", AuthState::default_path().display());
    println!();
    println!("mapping (all Claude models route to the same free model):");
    println!("  claude-opus-4-7      → {}", g.model);
    println!("  claude-sonnet-4-6    → {}", g.model);
    println!("  claude-haiku-4-5     → {}", g.model);
    println!();
    println!("Free tier limits apply: ~5–15 requests/minute, 100–1000 requests/day.");
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn anthropic_req_deserializes_text_fixture() {
        let raw = std::fs::read_to_string("tests/fixtures/anthropic_text.json").unwrap();
        let req: AnthropicReq = serde_json::from_str(&raw).unwrap();
        assert_eq!(req.model, "claude-sonnet-4-6");
        assert_eq!(req.max_tokens, Some(1024));
        assert_eq!(req.stream, Some(true));
        assert_eq!(req.messages.len(), 2);
        // First message: string content
        assert!(matches!(req.messages[0].content, MessageContent::Text(_)));
        // Second message: array of blocks
        assert!(matches!(req.messages[1].content, MessageContent::Blocks(_)));
    }

    #[test]
    fn codex_req_deserializes_text_fixture() {
        let raw = std::fs::read_to_string("tests/fixtures/codex_text.json").unwrap();
        let req: CodexReq = serde_json::from_str(&raw).unwrap();
        assert_eq!(req.model, "gpt-5.3-codex");
        assert_eq!(req.instructions.as_deref(), Some("You are a helpful assistant."));
        assert_eq!(req.input.len(), 2);
    }

    #[test]
    fn config_default_has_listen() {
        let cfg = Config::default();
        assert_eq!(cfg.listen, "127.0.0.1:4000");
    }

    #[test]
    fn config_ignores_legacy_model_keys() {
        // v0.2.0 dropped `default_model` and `[models]` — they should parse
        // without error (so old configs don't break startup) but contribute
        // nothing to the resulting Config struct.
        let toml = r#"
            listen = "127.0.0.1:5555"
            default_model = "gpt-5.4"

            [models]
            "claude-haiku-4-5-20251001" = "gpt-5.3-codex-spark"
        "#;
        let cfg = Config::parse_with_legacy_warnings(toml).unwrap();
        assert_eq!(cfg.listen, "127.0.0.1:5555");
    }

    fn load_pair(a: &str, c: &str) -> (AnthropicReq, Value) {
        let a: AnthropicReq = serde_json::from_str(
            &std::fs::read_to_string(format!("tests/fixtures/{a}")).unwrap()
        ).unwrap();
        let c: Value = serde_json::from_str(
            &std::fs::read_to_string(format!("tests/fixtures/{c}")).unwrap()
        ).unwrap();
        (a, c)
    }

    #[test]
    fn to_codex_translates_text() {
        let (a, expected) = load_pair("anthropic_text.json", "codex_text.json");
        // Fixture targets the Plus mapping (claude-sonnet-4-6 → gpt-5.3-codex).
        let model_out = ChatGptTier::Plus.map_model(&a.model);
        let got = to_codex(&a, model_out);
        assert_eq!(serde_json::to_value(&got).unwrap(), expected);
    }

    #[test]
    fn to_codex_translates_tools() {
        let (a, expected) = load_pair("anthropic_tool.json", "codex_tool.json");
        let model_out = ChatGptTier::Plus.map_model(&a.model);
        let got = to_codex(&a, model_out);
        assert_eq!(serde_json::to_value(&got).unwrap(), expected);
    }

    #[test]
    fn to_codex_translates_image() {
        let (a, expected) = load_pair("anthropic_image.json", "codex_image.json");
        let model_out = ChatGptTier::Plus.map_model(&a.model);
        let got = to_codex(&a, model_out);
        assert_eq!(serde_json::to_value(&got).unwrap(), expected);
    }

    // ── Gemini translation ───────────────────────────────────────────────

    #[test]
    fn to_gemini_translates_text() {
        let (a, expected) = load_pair("anthropic_text.json", "gemini_text.json");
        let got = to_gemini(&a).unwrap();
        assert_eq!(serde_json::to_value(&got).unwrap(), expected);
    }

    #[test]
    fn to_gemini_translates_tools_and_tool_results() {
        let (a, expected) = load_pair("anthropic_tool.json", "gemini_tool.json");
        let got = to_gemini(&a).unwrap();
        assert_eq!(serde_json::to_value(&got).unwrap(), expected);
    }

    #[test]
    fn to_gemini_rejects_image_inputs() {
        let req = AnthropicReq {
            model: "claude-haiku-4-5".into(),
            max_tokens: Some(64),
            stream: Some(true),
            system: None,
            messages: vec![Message {
                role: "user".into(),
                content: MessageContent::Blocks(vec![ContentBlock::Image {
                    source: ImageSource::Url { url: "data:image/png;base64,XYZ".into() },
                }]),
            }],
            tools: None,
            temperature: None,
            top_p: None,
            stop_sequences: None,
        };
        let err = to_gemini(&req).expect_err("image input must error on Gemini path");
        assert!(err.contains("not supported"), "msg: {err}");
    }

    #[test]
    fn gemini_translator_text_stream() {
        let mut t = GeminiTranslator::new("gemini-flash-latest".into());

        let frame1: GeminiStreamFrame = serde_json::from_value(serde_json::json!({
            "candidates": [{
                "content": { "parts": [{ "text": "Hello " }], "role": "model" }
            }]
        })).unwrap();
        let frame2: GeminiStreamFrame = serde_json::from_value(serde_json::json!({
            "candidates": [{
                "content": { "parts": [{ "text": "world" }], "role": "model" },
                "finishReason": "STOP"
            }],
            "usageMetadata": { "promptTokenCount": 4, "candidatesTokenCount": 2 }
        })).unwrap();

        let mut out: Vec<(String, Value)> = Vec::new();
        out.extend(t.next(frame1));
        out.extend(t.next(frame2));
        out.extend(t.finish());

        let names: Vec<&str> = out.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        assert_eq!(out[2].1["delta"]["text"], "Hello ");
        assert_eq!(out[3].1["delta"]["text"], "world");
        let md = out.iter().find(|(n, _)| n == "message_delta").unwrap();
        assert_eq!(md.1["delta"]["stop_reason"], "end_turn");
        assert_eq!(md.1["usage"]["input_tokens"], 4);
        assert_eq!(md.1["usage"]["output_tokens"], 2);
    }

    #[test]
    fn gemini_translator_tool_call_stream() {
        let mut t = GeminiTranslator::new("gemini-flash-latest".into());
        let frame: GeminiStreamFrame = serde_json::from_value(serde_json::json!({
            "candidates": [{
                "content": {
                    "parts": [
                        { "functionCall": { "name": "Read", "args": { "path": "foo.txt" } } }
                    ],
                    "role": "model"
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": { "promptTokenCount": 12, "candidatesTokenCount": 5 }
        })).unwrap();
        let mut out: Vec<(String, Value)> = Vec::new();
        out.extend(t.next(frame));
        out.extend(t.finish());

        let names: Vec<&str> = out.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        let cbs = &out[1].1;
        assert_eq!(cbs["content_block"]["type"], "tool_use");
        assert_eq!(cbs["content_block"]["name"], "Read");
        // Tool-use id is synthesised; just check the prefix.
        assert!(cbs["content_block"]["id"].as_str().unwrap().starts_with("toolu_"));
        let cbd = &out[2].1;
        assert_eq!(cbd["delta"]["type"], "input_json_delta");
        assert_eq!(cbd["delta"]["partial_json"], "{\"path\":\"foo.txt\"}");
        let md = out.iter().find(|(n, _)| n == "message_delta").unwrap();
        assert_eq!(md.1["delta"]["stop_reason"], "tool_use");
    }

    #[test]
    fn gemini_translator_maps_max_tokens_finish_reason() {
        let mut t = GeminiTranslator::new("gemini-flash-latest".into());
        let frame: GeminiStreamFrame = serde_json::from_value(serde_json::json!({
            "candidates": [{
                "content": { "parts": [{ "text": "hi" }], "role": "model" },
                "finishReason": "MAX_TOKENS"
            }]
        })).unwrap();
        let mut out = t.next(frame);
        out.extend(t.finish());
        let md = out.iter().find(|(n, _)| n == "message_delta").unwrap();
        assert_eq!(md.1["delta"]["stop_reason"], "max_tokens");
    }

    fn drain(t: &mut Translator, ev: CodexEvent) -> Vec<(String, Value)> {
        t.next(ev)
            .into_iter()
            .map(|(name, data)| (name, data))
            .collect()
    }

    #[test]
    fn translator_text_only_stream() {
        let mut t = Translator::new("gpt-5.4".into());
        let mut out = Vec::new();

        out.extend(drain(&mut t, CodexEvent::Created {
            response: CodexResponseMeta {
                id: Some("resp_1".into()),
                model: Some("gpt-5.4".into()),
                status: Some("in_progress".into()),
                usage: None,
            },
        }));
        out.extend(drain(&mut t, CodexEvent::OutputItemAdded {
            output_index: 0,
            item: CodexOutputItem::Message { id: "msg_1".into() },
        }));
        out.extend(drain(&mut t, CodexEvent::OutputTextDelta {
            item_id: "msg_1".into(), delta: "Hi ".into(),
        }));
        out.extend(drain(&mut t, CodexEvent::OutputTextDelta {
            item_id: "msg_1".into(), delta: "there.".into(),
        }));
        out.extend(drain(&mut t, CodexEvent::OutputItemDone {
            output_index: 0,
            item: CodexOutputItem::Message { id: "msg_1".into() },
        }));
        out.extend(drain(&mut t, CodexEvent::Completed {
            response: CodexResponseMeta {
                id: Some("resp_1".into()),
                model: Some("gpt-5.4".into()),
                status: Some("completed".into()),
                usage: Some(CodexUsage { input_tokens: 5, output_tokens: 3 }),
            },
        }));

        let names: Vec<&str> = out.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        assert_eq!(out[2].1["delta"]["type"], "text_delta");
        assert_eq!(out[2].1["delta"]["text"], "Hi ");
    }

    #[test]
    fn translator_tool_use_stream() {
        let mut t = Translator::new("gpt-5.4".into());
        let mut out = Vec::new();
        out.extend(drain(&mut t, CodexEvent::Created {
            response: CodexResponseMeta { id: Some("r".into()), model: Some("gpt-5.4".into()), status: None, usage: None },
        }));
        out.extend(drain(&mut t, CodexEvent::OutputItemAdded {
            output_index: 0,
            item: CodexOutputItem::FunctionCall {
                id: "fc_1".into(),
                call_id: "tc_1".into(),
                name: "Read".into(),
                arguments: String::new(),
            },
        }));
        out.extend(drain(&mut t, CodexEvent::FunctionCallArgsDelta {
            item_id: "fc_1".into(), delta: "{\"path\":\"foo".into(),
        }));
        out.extend(drain(&mut t, CodexEvent::FunctionCallArgsDelta {
            item_id: "fc_1".into(), delta: ".txt\"}".into(),
        }));
        out.extend(drain(&mut t, CodexEvent::OutputItemDone {
            output_index: 0,
            item: CodexOutputItem::FunctionCall {
                id: "fc_1".into(),
                call_id: "tc_1".into(),
                name: "Read".into(),
                arguments: "{\"path\":\"foo.txt\"}".into(),
            },
        }));
        out.extend(drain(&mut t, CodexEvent::Completed {
            response: CodexResponseMeta { id: None, model: None, status: Some("completed".into()), usage: None },
        }));

        let names: Vec<&str> = out.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        assert_eq!(out[1].1["content_block"]["type"], "tool_use");
        assert_eq!(out[1].1["content_block"]["name"], "Read");
        assert_eq!(out[1].1["content_block"]["id"], "tc_1");
        assert_eq!(out[2].1["delta"]["type"], "input_json_delta");
        // message_delta stop_reason should be tool_use since the only output was a function call.
        let md = out.iter().find(|(n, _)| n == "message_delta").unwrap();
        assert_eq!(md.1["delta"]["stop_reason"], "tool_use");
    }

    #[test]
    fn translator_interleaves_text_and_tool_by_item_id() {
        // Codex multiplexes events for multiple output items by item_id.
        // The translator must route deltas to the correct content block.
        let mut t = Translator::new("gpt-5.4".into());
        let mut out = Vec::new();
        out.extend(drain(&mut t, CodexEvent::Created {
            response: CodexResponseMeta { id: None, model: Some("gpt-5.4".into()), status: None, usage: None },
        }));
        // Open a message item and a function_call item.
        out.extend(drain(&mut t, CodexEvent::OutputItemAdded {
            output_index: 0,
            item: CodexOutputItem::Message { id: "msg_1".into() },
        }));
        out.extend(drain(&mut t, CodexEvent::OutputItemAdded {
            output_index: 1,
            item: CodexOutputItem::FunctionCall {
                id: "fc_1".into(), call_id: "tc_1".into(),
                name: "Read".into(), arguments: String::new(),
            },
        }));
        // Interleave deltas.
        out.extend(drain(&mut t, CodexEvent::OutputTextDelta {
            item_id: "msg_1".into(), delta: "hello".into(),
        }));
        out.extend(drain(&mut t, CodexEvent::FunctionCallArgsDelta {
            item_id: "fc_1".into(), delta: "{\"a\":1}".into(),
        }));
        out.extend(drain(&mut t, CodexEvent::OutputItemDone {
            output_index: 0,
            item: CodexOutputItem::Message { id: "msg_1".into() },
        }));
        out.extend(drain(&mut t, CodexEvent::OutputItemDone {
            output_index: 1,
            item: CodexOutputItem::FunctionCall {
                id: "fc_1".into(), call_id: "tc_1".into(),
                name: "Read".into(), arguments: "{\"a\":1}".into(),
            },
        }));
        out.extend(drain(&mut t, CodexEvent::Completed {
            response: CodexResponseMeta { id: None, model: None, status: None, usage: None },
        }));

        // Find the two content_block_start frames and their indices.
        let text_block_idx = out.iter().find_map(|(n, v)| {
            (n == "content_block_start" && v["content_block"]["type"] == "text").then(|| v["index"].clone())
        }).expect("text block start");
        let tool_block_idx = out.iter().find_map(|(n, v)| {
            (n == "content_block_start" && v["content_block"]["type"] == "tool_use").then(|| v["index"].clone())
        }).expect("tool block start");
        assert_ne!(text_block_idx, tool_block_idx);

        // The text delta lands at text_block_idx, the input_json_delta at tool_block_idx.
        let text_delta = out.iter().find(|(n, v)| {
            n == "content_block_delta" && v["index"] == text_block_idx && v["delta"]["type"] == "text_delta"
        });
        let tool_delta = out.iter().find(|(n, v)| {
            n == "content_block_delta" && v["index"] == tool_block_idx && v["delta"]["type"] == "input_json_delta"
        });
        assert!(text_delta.is_some(), "text delta must land on text block");
        assert!(tool_delta.is_some(), "input_json delta must land on tool block");
    }

    #[test]
    fn app_error_serializes_anthropic_shape() {
        let err = AppError::InvalidRequest("bad json".into());
        let body = serde_json::to_value(&err.body()).unwrap();
        assert_eq!(body["type"], "error");
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["message"], "bad json");
        assert_eq!(err.status().as_u16(), 400);
    }

    #[test]
    fn auth_state_roundtrips_to_file() {
        use std::time::{Duration, SystemTime};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("auth.json");

        // `load_from` treats the JWT as the source of truth for tier and (when
        // the on-disk struct lacks one) account_id, so the access_token must
        // carry the claims we expect to read back.
        let jwt = tier_tests::make_test_jwt("acct_42", "plus");
        let state = AuthState::Chatgpt(ChatgptAuth {
            access_token: jwt.clone(),
            refresh_token: "RT".into(),
            expires_at: SystemTime::now() + Duration::from_secs(3600),
            account_id: Some("acct_42".into()),
            tier: ChatGptTier::Plus,
        });
        state.save_to(&path).unwrap();

        let loaded = AuthState::load_from(&path).unwrap();
        let c = loaded.as_chatgpt().expect("expected chatgpt variant");
        assert_eq!(c.access_token, jwt);
        assert_eq!(c.refresh_token, "RT");
        assert_eq!(c.account_id.as_deref(), Some("acct_42"));
        assert_eq!(c.tier, ChatGptTier::Plus);

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600);
        }
    }

    #[test]
    fn auth_state_roundtrips_gemini_variant() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("auth.json");
        let state = AuthState::Gemini(GeminiAuth {
            api_key: "AIza_test_key".into(),
            model:   default_gemini_model(),
        });
        state.save_to(&path).unwrap();

        let loaded = AuthState::load_from(&path).unwrap();
        let g = loaded.as_gemini().expect("expected gemini variant");
        assert_eq!(g.api_key, "AIza_test_key");
        assert_eq!(g.model, "gemini-flash-latest");

        // The on-disk JSON must carry the provider discriminator.
        let raw = std::fs::read_to_string(&path).unwrap();
        let v: Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(v["provider"], "gemini");
    }

    #[test]
    fn auth_state_loads_legacy_untagged_as_chatgpt() {
        // Pre-v0.3 auth.json files have no `provider` field. They must still
        // load and round-trip via the Chatgpt variant.
        let dir  = tempfile::tempdir().unwrap();
        let path = dir.path().join("auth.json");
        std::fs::write(&path, r#"{
            "access_token":  "AT",
            "refresh_token": "RT",
            "expires_at":    9999999999,
            "account_id":    "acct"
        }"#).unwrap();
        let loaded = AuthState::load_from(&path).unwrap();
        let c = loaded.as_chatgpt().expect("legacy file must load as Chatgpt");
        assert_eq!(c.access_token, "AT");
        assert_eq!(c.refresh_token, "RT");
        assert_eq!(c.account_id.as_deref(), Some("acct"));
    }

    #[tokio::test]
    async fn auth_refresh_against_mock() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/oauth/token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "access_token": "new_AT",
                "refresh_token": "new_RT",
                "expires_in": 3600,
                "token_type": "Bearer"
            })))
            .mount(&server)
            .await;

        let new_state = refresh_at(
            &reqwest::Client::new(),
            &server.uri(),
            "/oauth/token",
            OAUTH_CLIENT_ID,
            "old_RT",
        )
        .await
        .unwrap();
        assert_eq!(new_state.access_token, "new_AT");
        assert_eq!(new_state.refresh_token, "new_RT");
        assert!(new_state.is_fresh(std::time::Duration::from_secs(60)));
    }

    #[tokio::test]
    async fn pkce_login_against_mock() {
        use wiremock::matchers::{body_string_contains, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;

        // Token endpoint: succeed when we see an authorization-code grant.
        Mock::given(method("POST"))
            .and(path("/oauth/token"))
            .and(body_string_contains("grant_type=authorization_code"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "access_token":  "AT",
                "refresh_token": "RT",
                "expires_in":    3600,
                "account_id":    "acct",
            })))
            .mount(&server)
            .await;

        // Pick a free localhost port for the callback listener. We bind+drop
        // to ask the kernel for an unused port; brief race window, fine here.
        let port = {
            let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let p = l.local_addr().unwrap().port();
            drop(l);
            p
        };

        let server_uri = server.uri();
        let login = tokio::spawn(async move {
            pkce_login_at(
                &server_uri,
                "/oauth/authorize",
                "/oauth/token",
                OAUTH_CLIENT_ID,
                OAUTH_SCOPES,
                |authorize_url| {
                    // Simulate the browser: parse the authorize URL,
                    // extract state + redirect_uri, then GET the callback
                    // with a fake `code` and the right `state`.
                    let parsed = url::Url::parse(authorize_url).unwrap();
                    let mut state = String::new();
                    let mut redirect = String::new();
                    for (k, v) in parsed.query_pairs() {
                        match k.as_ref() {
                            "state"        => state = v.into_owned(),
                            "redirect_uri" => redirect = v.into_owned(),
                            _ => {}
                        }
                    }
                    assert!(!state.is_empty(), "authorize URL missing state");
                    assert!(!redirect.is_empty(), "authorize URL missing redirect_uri");
                    let cb = format!("{redirect}?code=AUTHCODE&state={state}");
                    tokio::spawn(async move {
                        // Tiny delay so the callback server is definitely up.
                        tokio::time::sleep(
                            std::time::Duration::from_millis(50)
                        ).await;
                        let _ = reqwest::Client::new().get(&cb).send().await;
                    });
                },
                Some(port),
            )
            .await
        });

        let state = login.await.unwrap().unwrap();
        assert_eq!(state.access_token,  "AT");
        assert_eq!(state.refresh_token, "RT");
        assert_eq!(state.account_id.as_deref(), Some("acct"));
        assert!(state.is_fresh(std::time::Duration::from_secs(60)));
    }

    #[test]
    fn pkce_helpers_produce_expected_shapes() {
        // verifier: 64 random bytes → URL-safe-no-pad base64 → 86 chars.
        let (verifier, challenge) = generate_pkce();
        assert_eq!(verifier.len(), 86);
        assert!(verifier.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'));
        // challenge: SHA-256 (32 bytes) → URL-safe-no-pad base64 → 43 chars.
        assert_eq!(challenge.len(), 43);

        let s = generate_state();
        assert_eq!(s.len(), 43); // 32 bytes → 43 chars unpadded base64.
    }

    #[test]
    fn build_authorize_url_emits_all_required_params() {
        let url = build_authorize_url(
            "https://auth.example.com",
            "/oauth/authorize",
            "client123",
            "openid profile",
            "http://localhost:1455/auth/callback",
            "STATE",
            "CHALLENGE",
        );
        let parsed = url::Url::parse(&url).unwrap();
        let q: std::collections::HashMap<_, _> = parsed.query_pairs().into_owned().collect();
        assert_eq!(q.get("response_type").map(String::as_str), Some("code"));
        assert_eq!(q.get("client_id").map(String::as_str), Some("client123"));
        assert_eq!(q.get("scope").map(String::as_str), Some("openid profile"));
        assert_eq!(q.get("state").map(String::as_str), Some("STATE"));
        assert_eq!(q.get("code_challenge").map(String::as_str), Some("CHALLENGE"));
        assert_eq!(q.get("code_challenge_method").map(String::as_str), Some("S256"));
        assert_eq!(
            q.get("redirect_uri").map(String::as_str),
            Some("http://localhost:1455/auth/callback"),
        );
    }

    #[tokio::test]
    async fn malformed_json_returns_anthropic_error_envelope() {
        let app = build_app(test_state());
        let server = axum_test::TestServer::new(app);
        for endpoint in ["/v1/messages", "/v1/messages/count_tokens"] {
            let resp = server.post(endpoint)
                .content_type("application/json")
                .text("not-json")
                .await;
            assert_eq!(resp.status_code(), 400, "{endpoint} should reject malformed JSON");
            let json: Value = resp.json();
            assert_eq!(json["type"], "error", "{endpoint} body shape");
            assert_eq!(json["error"]["type"], "invalid_request_error", "{endpoint} error type");
            assert!(
                json["error"]["message"].as_str().is_some_and(|s| !s.is_empty()),
                "{endpoint} should include a non-empty message",
            );
        }
    }

    #[tokio::test]
    async fn count_tokens_returns_heuristic() {
        let app = build_app(test_state());
        let server = axum_test::TestServer::new(app);
        let body = serde_json::json!({
            "model": "claude-sonnet-4-6",
            "messages": [
                { "role": "user", "content": "Hello world, this is a test." }
            ]
        });
        let resp = server.post("/v1/messages/count_tokens").json(&body).await;
        resp.assert_status_ok();
        let json: Value = resp.json();
        assert!(json["input_tokens"].as_u64().unwrap() > 0);
    }

    fn test_state() -> AppState {
        AppState {
            cfg: Config::default(),
            http: reqwest::Client::new(),
            upstream_base: "http://127.0.0.1:0".into(),
            gemini_base:   "http://127.0.0.1:0".into(),
            auth: std::sync::Arc::new(tokio::sync::Mutex::new(None)),
            auth_path: std::path::PathBuf::from("/dev/null"),
            auth_origin: "http://127.0.0.1:0".into(),
        }
    }

    #[tokio::test]
    async fn messages_streams_translated_response() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};
        use std::time::Duration;

        let upstream = MockServer::start().await;

        // Build a fake SSE stream of Codex events
        let sse_body = concat!(
            "event: response.created\n",
            "data: {\"type\":\"response.created\",\"response\":{\"id\":\"r1\",\"model\":\"gpt-5.4\",\"status\":\"in_progress\"}}\n\n",
            "event: response.output_text.delta\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hello\"}\n\n",
            "event: response.output_text.delta\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\" world\"}\n\n",
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"r1\",\"status\":\"completed\",\"usage\":{\"input_tokens\":3,\"output_tokens\":2}}}\n\n",
        );

        Mock::given(method("POST"))
            .and(path("/backend-api/codex/responses"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse_body),
            )
            .mount(&upstream)
            .await;

        let mut state = test_state();
        state.upstream_base = upstream.uri();
        *state.auth.lock().await = Some(AuthState::Chatgpt(ChatgptAuth {
            access_token: "AT".into(),
            refresh_token: "RT".into(),
            expires_at: SystemTime::now() + Duration::from_secs(3600),
            account_id: None,
            tier: ChatGptTier::Plus,
        }));

        let app = build_app(state);
        let server = axum_test::TestServer::new(app);
        let resp = server.post("/v1/messages")
            .json(&serde_json::json!({
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "stream": true,
                "messages": [{ "role": "user", "content": "hi" }]
            }))
            .await;

        resp.assert_status_ok();
        let body = resp.text();
        assert!(body.contains("event: message_start"));
        assert!(body.contains("event: content_block_delta"));
        assert!(body.contains("\"text\":\"Hello\""));
        assert!(body.contains("\"text\":\" world\""));
        assert!(body.contains("event: message_stop"));
    }

    #[tokio::test]
    async fn messages_refreshes_token_on_401_then_succeeds() {
        use wiremock::matchers::{method, path, header};
        use wiremock::{Mock, MockServer, ResponseTemplate};
        use std::time::Duration;

        let upstream = MockServer::start().await;

        // First /backend-api/codex/responses with the stale token: 401.
        Mock::given(method("POST"))
            .and(path("/backend-api/codex/responses"))
            .and(header("authorization", "Bearer stale_AT"))
            .respond_with(ResponseTemplate::new(401))
            .up_to_n_times(1)
            .mount(&upstream)
            .await;

        // OAuth refresh: hand back a fresh access token.
        Mock::given(method("POST"))
            .and(path("/oauth/token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "access_token": "fresh_AT",
                "refresh_token": "new_RT",
                "expires_in": 3600,
                "token_type": "Bearer"
            })))
            .mount(&upstream)
            .await;

        // Retry with the fresh token: succeed and return a valid SSE body.
        let sse_body = concat!(
            "event: response.created\n",
            "data: {\"type\":\"response.created\",\"response\":{\"id\":\"r1\",\"model\":\"gpt-5.4\",\"status\":\"in_progress\"}}\n\n",
            "event: response.output_text.delta\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hi after refresh\"}\n\n",
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"r1\",\"status\":\"completed\"}}\n\n",
        );
        Mock::given(method("POST"))
            .and(path("/backend-api/codex/responses"))
            .and(header("authorization", "Bearer fresh_AT"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse_body),
            )
            .mount(&upstream)
            .await;

        let mut state = test_state();
        state.upstream_base = upstream.uri();
        state.auth_origin = upstream.uri();
        // An auth state that is still inside its "fresh" window but stale on
        // the wire — so we exercise the 401-retry path rather than the
        // proactive refresh in `current_access_token`.
        *state.auth.lock().await = Some(AuthState::Chatgpt(ChatgptAuth {
            access_token: "stale_AT".into(),
            refresh_token: "stale_RT".into(),
            expires_at: SystemTime::now() + Duration::from_secs(3600),
            account_id: None,
            tier: ChatGptTier::Plus,
        }));

        let app = build_app(state.clone());
        let server = axum_test::TestServer::new(app);
        let resp = server.post("/v1/messages")
            .json(&serde_json::json!({
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "stream": true,
                "messages": [{ "role": "user", "content": "ping" }]
            }))
            .await;

        resp.assert_status_ok();
        let body = resp.text();
        assert!(body.contains("\"text\":\"hi after refresh\""), "body: {body}");
        // The cached token must have been replaced.
        let cached = state.auth.lock().await;
        let c = cached.as_ref().and_then(|a| a.as_chatgpt()).expect("chatgpt");
        assert_eq!(c.access_token, "fresh_AT");
        assert_eq!(c.refresh_token, "new_RT");
    }

    #[tokio::test]
    async fn messages_emits_error_frame_on_bad_upstream_json() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};
        use std::time::Duration;

        let upstream = MockServer::start().await;
        // The first event parses; the second has unparseable JSON in `data`.
        // The proxy should emit a normal `message_start`, then an `error` and
        // `message_stop` frame instead of silently dropping the bad event.
        let sse_body = concat!(
            "event: response.created\n",
            "data: {\"type\":\"response.created\",\"response\":{\"id\":\"r1\",\"model\":\"gpt-5.4\"}}\n\n",
            "event: response.output_text.delta\n",
            "data: {not json\n\n",
        );
        Mock::given(method("POST"))
            .and(path("/backend-api/codex/responses"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse_body),
            )
            .mount(&upstream)
            .await;

        let mut state = test_state();
        state.upstream_base = upstream.uri();
        *state.auth.lock().await = Some(AuthState::Chatgpt(ChatgptAuth {
            access_token: "AT".into(),
            refresh_token: "RT".into(),
            expires_at: SystemTime::now() + Duration::from_secs(3600),
            account_id: None,
            tier: ChatGptTier::Plus,
        }));

        let app = build_app(state);
        let server = axum_test::TestServer::new(app);
        let resp = server.post("/v1/messages")
            .json(&serde_json::json!({
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "stream": true,
                "messages": [{ "role": "user", "content": "hi" }]
            }))
            .await;

        resp.assert_status_ok();
        let body = resp.text();
        assert!(body.contains("event: message_start"));
        assert!(body.contains("event: error"));
        assert!(body.contains("\"type\":\"api_error\""));
        assert!(body.contains("event: message_stop"));
    }

    #[tokio::test]
    async fn messages_streams_translated_response_via_gemini() {
        use wiremock::matchers::{method, path, query_param};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let upstream = MockServer::start().await;

        let sse_body = concat!(
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hello\"}],\"role\":\"model\"}}]}\n\n",
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\" world\"}],\"role\":\"model\"},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":3,\"candidatesTokenCount\":2}}\n\n",
        );

        Mock::given(method("POST"))
            .and(path("/v1beta/models/gemini-flash-latest:streamGenerateContent"))
            .and(query_param("alt", "sse"))
            .and(query_param("key", "AIza_test"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse_body),
            )
            .mount(&upstream)
            .await;

        let mut state = test_state();
        state.gemini_base = upstream.uri();
        *state.auth.lock().await = Some(AuthState::Gemini(GeminiAuth {
            api_key: "AIza_test".into(),
            model:   "gemini-flash-latest".into(),
        }));

        let app = build_app(state);
        let server = axum_test::TestServer::new(app);
        let resp = server.post("/v1/messages")
            .json(&serde_json::json!({
                "model": "claude-haiku-4-5",
                "max_tokens": 64,
                "stream": true,
                "messages": [{ "role": "user", "content": "hi" }]
            }))
            .await;

        resp.assert_status_ok();
        let body = resp.text();
        assert!(body.contains("event: message_start"));
        assert!(body.contains("\"text\":\"Hello\""), "body: {body}");
        assert!(body.contains("\"text\":\" world\""), "body: {body}");
        assert!(body.contains("event: message_stop"));
        assert!(body.contains("\"stop_reason\":\"end_turn\""));
    }

    #[tokio::test]
    async fn messages_returns_401_when_no_credentials() {
        let state = test_state();
        // auth stays None.
        let app = build_app(state);
        let server = axum_test::TestServer::new(app);
        let resp = server.post("/v1/messages")
            .json(&serde_json::json!({
                "model": "claude-haiku-4-5",
                "messages": [{ "role": "user", "content": "hi" }]
            }))
            .await;
        assert_eq!(resp.status_code(), 401);
        let body: Value = resp.json();
        assert_eq!(body["error"]["type"], "authentication_error");
    }

    #[test]
    fn cli_parses_login_provider_flags() {
        use clap::Parser;

        let cli = Cli::parse_from(["claudeg", "login", "--gemini"]);
        match cli.cmd {
            Some(Cmd::Login { chatgpt, gemini, gemini_key }) => {
                assert!(!chatgpt);
                assert!(gemini);
                assert!(gemini_key.is_none());
            }
            _ => panic!("expected Login"),
        }

        let cli = Cli::parse_from(["claudeg", "login", "--gemini-key", "AIza_x"]);
        match cli.cmd {
            Some(Cmd::Login { chatgpt, gemini, gemini_key }) => {
                assert!(!chatgpt);
                assert!(!gemini);
                assert_eq!(gemini_key.as_deref(), Some("AIza_x"));
            }
            _ => panic!("expected Login"),
        }

        let cli = Cli::parse_from(["claudeg", "login", "--chatgpt"]);
        match cli.cmd {
            Some(Cmd::Login { chatgpt, gemini, gemini_key }) => {
                assert!(chatgpt);
                assert!(!gemini);
                assert!(gemini_key.is_none());
            }
            _ => panic!("expected Login"),
        }

        // --chatgpt + --gemini must be rejected as a clap conflict.
        let err = Cli::try_parse_from(["claudeg", "login", "--chatgpt", "--gemini"]);
        assert!(err.is_err(), "conflicting flags must error");
    }

    #[test]
    fn cli_parses_subcommands() {
        use clap::Parser;
        let cli = Cli::parse_from(["claudeg", "login"]);
        assert!(matches!(cli.cmd, Some(Cmd::Login { .. })));

        let cli = Cli::parse_from(["claudeg", "serve", "--port", "5555"]);
        match cli.cmd {
            Some(Cmd::Serve { port }) => assert_eq!(port, Some(5555)),
            _ => panic!("expected serve"),
        }
    }

    #[test]
    fn cli_parses_setup_and_uninstall() {
        use clap::Parser;
        let cli = Cli::parse_from(["claudeg", "setup"]);
        assert!(matches!(cli.cmd, Some(Cmd::Setup { .. })));

        let cli = Cli::parse_from(["claudeg", "uninstall"]);
        assert!(matches!(cli.cmd, Some(Cmd::Uninstall { .. })));

        let cli = Cli::parse_from(["claudeg", "logout"]);
        assert!(matches!(cli.cmd, Some(Cmd::Logout)));
    }

    #[test]
    fn cli_with_no_args_yields_no_subcommand() {
        // `claudeg` alone parses to `cmd == None`; main() turns this into
        // `Cmd::Run(vec![])` so the user gets an interactive `claude` session
        // through the proxy (matching plain `claude`'s no-args behavior).
        use clap::Parser;
        let cli = Cli::parse_from(["claudeg"]);
        assert!(cli.cmd.is_none(), "no args should produce None, got {:?}", cli.cmd);
    }

    #[test]
    fn cli_treats_unknown_args_as_run_passthrough() {
        // `claudeg "list rust files"` and `claudeg --some-claude-flag foo` should
        // both be captured by the external `Run` subcommand so the args flow
        // through to the underlying `claude` binary unchanged.
        use clap::Parser;
        let cli = Cli::parse_from(["claudeg", "list rust files"]);
        match cli.cmd {
            Some(Cmd::Run(args)) => assert_eq!(args, vec!["list rust files".to_string()]),
            _ => panic!("expected Run, got {:?}", cli.cmd),
        }
        let cli = Cli::parse_from(["claudeg", "summarize", "the", "spec"]);
        match cli.cmd {
            Some(Cmd::Run(args)) => assert_eq!(args, vec!["summarize", "the", "spec"]),
            _ => panic!("expected Run, got {:?}", cli.cmd),
        }
    }

    #[test]
    fn settings_merge_creates_when_missing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        let bak = configure_claude_settings(&path, 4000).unwrap();
        assert!(bak.is_none(), "no backup when file did not exist");
        let v: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(v["env"]["ANTHROPIC_BASE_URL"], "http://127.0.0.1:4000");
        assert_eq!(v["env"]["ANTHROPIC_API_KEY"], "_");
    }

    #[test]
    fn settings_merge_preserves_existing_keys() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        std::fs::write(&path, r#"{
            "theme": "dark",
            "env": { "FOO": "bar" }
        }"#).unwrap();
        let bak = configure_claude_settings(&path, 5555).unwrap();
        assert!(bak.is_some(), "backup written when file existed");
        let v: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(v["theme"], "dark");
        assert_eq!(v["env"]["FOO"], "bar");
        assert_eq!(v["env"]["ANTHROPIC_BASE_URL"], "http://127.0.0.1:5555");
        assert_eq!(v["env"]["ANTHROPIC_API_KEY"], "_");
    }

    #[test]
    fn settings_unconfigure_removes_only_our_keys() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.json");
        std::fs::write(&path, r#"{
            "theme": "dark",
            "env": {
                "FOO": "bar",
                "ANTHROPIC_BASE_URL": "http://127.0.0.1:4000",
                "ANTHROPIC_API_KEY": "_"
            }
        }"#).unwrap();
        unconfigure_claude_settings(&path).unwrap();
        let v: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(v["env"]["FOO"], "bar");
        assert!(v["env"].get("ANTHROPIC_BASE_URL").is_none());
        assert!(v["env"].get("ANTHROPIC_API_KEY").is_none());
        assert_eq!(v["theme"], "dark");
    }

    #[test]
    fn approve_wrapper_api_key_is_idempotent_and_preserves_other_keys() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".claude.json");
        std::fs::write(&path, r#"{
            "anonymousId": "xyz",
            "customApiKeyResponses": {
                "approved": ["existing-approved-tail"],
                "rejected": ["existing-rejected-tail"]
            }
        }"#).unwrap();

        let added = approve_wrapper_api_key(&path).unwrap();
        assert!(added, "first call should add the sentinel");

        let v: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(v["anonymousId"], "xyz");
        let approved = v["customApiKeyResponses"]["approved"].as_array().unwrap();
        assert!(approved.iter().any(|x| x == "existing-approved-tail"));
        assert!(approved.iter().any(|x| x == WRAPPER_API_KEY_TAIL));
        // Rejected list untouched.
        assert_eq!(v["customApiKeyResponses"]["rejected"][0], "existing-rejected-tail");

        // Second call should be a no-op.
        let added2 = approve_wrapper_api_key(&path).unwrap();
        assert!(!added2, "second call should be a no-op");
    }

    #[test]
    fn approve_wrapper_api_key_creates_file_when_missing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".claude.json");
        assert!(!path.exists());
        let added = approve_wrapper_api_key(&path).unwrap();
        assert!(added);
        let v: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(v["customApiKeyResponses"]["approved"][0], WRAPPER_API_KEY_TAIL);
    }

    #[test]
    fn sentinel_api_key_tail_matches_constant() {
        // The sentinel value's last 20 chars must equal WRAPPER_API_KEY_TAIL,
        // because that's what `claudeg setup` writes into ~/.claude.json so
        // Claude Code stops prompting on every `claudeg "..."` invocation.
        let len = SENTINEL_API_KEY.len();
        assert!(len >= 20);
        let tail = &SENTINEL_API_KEY[len - 20..];
        assert_eq!(tail, WRAPPER_API_KEY_TAIL);
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn launchd_plist_content_is_well_formed() {
        let plist = build_launchd_plist("/usr/local/bin/claudeg", 4000);
        assert!(plist.contains("<key>Label</key>"));
        assert!(plist.contains("<string>com.claudeg.proxy</string>"));
        assert!(plist.contains("<key>RunAtLoad</key>"));
        assert!(plist.contains("<key>KeepAlive</key>"));
        assert!(plist.contains("<string>/usr/local/bin/claudeg</string>"));
        assert!(plist.contains("<string>serve</string>"));
        assert!(plist.contains("<string>--port</string>"));
        assert!(plist.contains("<string>4000</string>"));
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn systemd_unit_content_is_well_formed() {
        let unit = build_systemd_unit("/home/user/.local/bin/claudeg", 4000);
        assert!(unit.contains("[Service]"));
        assert!(unit.contains("ExecStart=/home/user/.local/bin/claudeg serve --port 4000"));
        assert!(unit.contains("Restart=on-failure"));
    }

    #[test]
    #[cfg(target_os = "windows")]
    fn windows_startup_cmd_is_well_formed() {
        let cmd = build_windows_startup_cmd("C:\\Users\\u\\claudeg.exe", 4000);
        assert!(cmd.contains("start \"\" /B"));
        assert!(cmd.contains("\"C:\\Users\\u\\claudeg.exe\" serve --port 4000"));
    }

    // ─── ChatGPT-tier auto-mapping (v0.2.0) ───
    mod tier_tests {
        use super::super::*;
        use pretty_assertions::assert_eq;

        #[test]
        fn from_jwt_claim_recognises_known_tiers_case_insensitive() {
            // Canonical lowercase strings.
            assert_eq!(ChatGptTier::from_jwt_claim("free"),       ChatGptTier::Free);
            assert_eq!(ChatGptTier::from_jwt_claim("go"),         ChatGptTier::Go);
            assert_eq!(ChatGptTier::from_jwt_claim("plus"),       ChatGptTier::Plus);
            assert_eq!(ChatGptTier::from_jwt_claim("edu"),        ChatGptTier::Edu);
            assert_eq!(ChatGptTier::from_jwt_claim("pro"),        ChatGptTier::Pro);
            assert_eq!(ChatGptTier::from_jwt_claim("business"),   ChatGptTier::Business);
            assert_eq!(ChatGptTier::from_jwt_claim("enterprise"), ChatGptTier::Enterprise);

            // Case-insensitive — the spec says any casing must match.
            assert_eq!(ChatGptTier::from_jwt_claim("Plus"),       ChatGptTier::Plus);
            assert_eq!(ChatGptTier::from_jwt_claim("PLUS"),       ChatGptTier::Plus);
            assert_eq!(ChatGptTier::from_jwt_claim("Pro"),        ChatGptTier::Pro);
            assert_eq!(ChatGptTier::from_jwt_claim("BUSINESS"),   ChatGptTier::Business);
            assert_eq!(ChatGptTier::from_jwt_claim("Enterprise"), ChatGptTier::Enterprise);

            // Anything outside the known set falls through to Unknown.
            assert_eq!(ChatGptTier::from_jwt_claim(""),           ChatGptTier::Unknown);
            assert_eq!(ChatGptTier::from_jwt_claim("team"),       ChatGptTier::Unknown);
            assert_eq!(ChatGptTier::from_jwt_claim("plus_v2"),    ChatGptTier::Unknown);
        }

        #[test]
        fn map_model_go_tier() {
            let t = ChatGptTier::Go;
            assert_eq!(t.map_model("claude-opus-4-7"),            "gpt-5.5");
            assert_eq!(t.map_model("claude-sonnet-4-6"),          "gpt-5.5");
            assert_eq!(t.map_model("claude-haiku-4-5-20251001"),  "gpt-5.4-mini");
            assert_eq!(t.map_model("claude-haiku-4-5"),           "gpt-5.4-mini");
        }

        #[test]
        fn map_model_plus_edu_unknown_share_the_same_table() {
            for t in [ChatGptTier::Plus, ChatGptTier::Edu, ChatGptTier::Unknown] {
                assert_eq!(t.map_model("claude-opus-4-7"),           "gpt-5.5",       "{t:?}");
                assert_eq!(t.map_model("claude-sonnet-4-6"),         "gpt-5.3-codex", "{t:?}");
                assert_eq!(t.map_model("claude-haiku-4-5-20251001"), "gpt-5.4-mini",  "{t:?}");
                assert_eq!(t.map_model("claude-haiku-4-5"),          "gpt-5.4-mini",  "{t:?}");
            }
        }

        #[test]
        fn map_model_pro_business_enterprise_share_the_same_table() {
            for t in [ChatGptTier::Pro, ChatGptTier::Business, ChatGptTier::Enterprise] {
                assert_eq!(t.map_model("claude-opus-4-7"),           "gpt-5.5-pro",        "{t:?}");
                assert_eq!(t.map_model("claude-sonnet-4-6"),         "gpt-5.3-codex-spark","{t:?}");
                assert_eq!(t.map_model("claude-haiku-4-5-20251001"), "gpt-5.4-mini",       "{t:?}");
                assert_eq!(t.map_model("claude-haiku-4-5"),          "gpt-5.4-mini",       "{t:?}");
            }
        }

        #[test]
        fn default_fallback_per_tier() {
            // Each tier has a fallback for "Claude model not in our table".
            assert_eq!(ChatGptTier::Go.default_fallback_model(),         "gpt-5.5");
            assert_eq!(ChatGptTier::Plus.default_fallback_model(),       "gpt-5.3-codex");
            assert_eq!(ChatGptTier::Edu.default_fallback_model(),        "gpt-5.3-codex");
            assert_eq!(ChatGptTier::Unknown.default_fallback_model(),    "gpt-5.3-codex");
            assert_eq!(ChatGptTier::Pro.default_fallback_model(),        "gpt-5.3-codex-spark");
            assert_eq!(ChatGptTier::Business.default_fallback_model(),   "gpt-5.3-codex-spark");
            assert_eq!(ChatGptTier::Enterprise.default_fallback_model(), "gpt-5.3-codex-spark");

            // And map_model falls back to it for unknown Claude slugs.
            assert_eq!(ChatGptTier::Plus.map_model("claude-future-model"), "gpt-5.3-codex");
            assert_eq!(ChatGptTier::Pro.map_model("claude-future-model"),  "gpt-5.3-codex-spark");
            assert_eq!(ChatGptTier::Go.map_model("claude-future-model"),   "gpt-5.5");
        }

        #[test]
        #[should_panic(expected = "ChatGptTier::map_model called on Free")]
        fn map_model_panics_on_free_tier() {
            // Free is rejected at the handler — `map_model` should never run.
            let _ = ChatGptTier::Free.map_model("claude-opus-4-7");
        }

        #[test]
        fn default_is_unknown() {
            assert_eq!(ChatGptTier::default(), ChatGptTier::Unknown);
        }

        #[test]
        fn extract_jwt_claims_reads_both_account_id_and_plan_type() {
            use base64::Engine;
            let payload = serde_json::json!({
                "https://api.openai.com/auth": {
                    "chatgpt_account_id": "acct_xyz",
                    "chatgpt_plan_type": "pro"
                }
            });
            let b64 = base64::engine::general_purpose::URL_SAFE_NO_PAD
                .encode(serde_json::to_vec(&payload).unwrap());
            let jwt = format!("header.{b64}.signature");
            let (acct, tier) = extract_jwt_claims(&jwt);
            assert_eq!(acct.as_deref(), Some("acct_xyz"));
            assert_eq!(tier, ChatGptTier::Pro);
        }

        #[test]
        fn extract_jwt_claims_missing_plan_type_yields_unknown() {
            use base64::Engine;
            let payload = serde_json::json!({
                "https://api.openai.com/auth": { "chatgpt_account_id": "acct" }
            });
            let b64 = base64::engine::general_purpose::URL_SAFE_NO_PAD
                .encode(serde_json::to_vec(&payload).unwrap());
            let jwt = format!("h.{b64}.s");
            let (acct, tier) = extract_jwt_claims(&jwt);
            assert_eq!(acct.as_deref(), Some("acct"));
            assert_eq!(tier, ChatGptTier::Unknown);
        }

        #[test]
        fn extract_jwt_claims_malformed_token_yields_none_unknown() {
            assert_eq!(extract_jwt_claims("not-a-jwt"),  (None, ChatGptTier::Unknown));
            assert_eq!(extract_jwt_claims(""),           (None, ChatGptTier::Unknown));
            assert_eq!(extract_jwt_claims("a.b.c"),      (None, ChatGptTier::Unknown));
        }

        #[test]
        fn auth_state_deserializes_without_tier_field() {
            // Old auth.json files (v0.1.x) had no `tier` field. They must still
            // load, defaulting to Unknown.
            let json = r#"{
                "access_token":  "AT",
                "refresh_token": "RT",
                "expires_at":    9999999999,
                "account_id":    "acct"
            }"#;
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("auth.json");
            std::fs::write(&path, json).unwrap();
            let s = AuthState::load_from(&path).unwrap();
            let c = s.as_chatgpt().expect("legacy file → Chatgpt");
            assert_eq!(c.tier, ChatGptTier::Unknown);
        }

        /// Build a JWT whose payload carries the given (account_id, plan_type)
        /// under the `https://api.openai.com/auth` namespace. Only the payload
        /// segment is meaningful; the header/signature are placeholders.
        pub(super) fn make_test_jwt(account_id: &str, plan_type: &str) -> String {
            use base64::Engine;
            let payload = serde_json::json!({
                "https://api.openai.com/auth": {
                    "chatgpt_account_id": account_id,
                    "chatgpt_plan_type":  plan_type,
                }
            });
            let b64 = base64::engine::general_purpose::URL_SAFE_NO_PAD
                .encode(serde_json::to_vec(&payload).unwrap());
            format!("header.{b64}.signature")
        }

        #[test]
        fn load_from_derives_tier_from_jwt_when_disk_missing_tier() {
            // Regression: a v0.1.x auth.json has no `tier` field, but the cached
            // access_token JWT carries `chatgpt_plan_type`. `load_from` must
            // treat the JWT as the source of truth instead of defaulting to
            // Unknown, so old caches self-heal on first run under v0.2.x.
            let jwt = make_test_jwt("acct_jwt", "plus");
            let json = format!(
                r#"{{
                    "access_token":  "{jwt}",
                    "refresh_token": "RT",
                    "expires_at":    9999999999
                }}"#
            );
            let dir  = tempfile::tempdir().unwrap();
            let path = dir.path().join("auth.json");
            std::fs::write(&path, json).unwrap();

            let s = AuthState::load_from(&path).unwrap();
            let c = s.as_chatgpt().expect("chatgpt variant");
            assert_eq!(c.tier, ChatGptTier::Plus);
            assert_eq!(c.account_id.as_deref(), Some("acct_jwt"));
        }

        #[tokio::test]
        async fn handler_rejects_free_tier_with_403_subscription_required() {
            use std::time::Duration;

            let state = super::test_state();
            // No upstream needed — the handler must short-circuit on Free.
            *state.auth.lock().await = Some(AuthState::Chatgpt(ChatgptAuth {
                access_token: "AT".into(),
                refresh_token: "RT".into(),
                expires_at: SystemTime::now() + Duration::from_secs(3600),
                account_id: None,
                tier: ChatGptTier::Free,
            }));

            let app = build_app(state);
            let server = axum_test::TestServer::new(app);
            let resp = server.post("/v1/messages")
                .json(&serde_json::json!({
                    "model": "claude-opus-4-7",
                    "messages": [{ "role": "user", "content": "hi" }],
                }))
                .await;
            assert_eq!(resp.status_code(), 403);
            let body: Value = resp.json();
            assert_eq!(body["error"]["type"], "subscription_required");
            assert!(
                body["error"]["message"].as_str().is_some_and(|s| s.contains("Detected: free")),
                "message should mention detected tier: {body}"
            );
        }
    }
}
