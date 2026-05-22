//! REPL Module - Interactive Read-Eval-Print Loop
//!
//! Beautiful REPL interface matching the original AgentRust aesthetic

use crate::api::{ApiClient, ChatMessage, ToolDefinition, ToolCall};
use crate::cli::ui;
use crate::metacognition::MetaAction;
use crate::state::AppState;
use crate::mcp::ToolRegistry;
use colored::Colorize;
use std::io::{self, BufRead, Write};
use std::sync::Arc;

pub struct Repl {
    state: AppState,
    conversation_history: Vec<ChatMessage>,
    tool_registry: Arc<ToolRegistry>,
}

impl Repl {
    pub fn new(state: AppState) -> Self {
        ui::init_terminal();
        let tool_registry = Arc::new(ToolRegistry::new());

        // 注册内置工具（使用 tokio::task::block_in_place）
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                tool_registry.register_builtin_tools().await;
            });
        });

        Self {
            state,
            conversation_history: Vec::new(),
            tool_registry,
        }
    }

    pub fn start(&mut self, initial_prompt: Option<String>) -> anyhow::Result<()> {
        ui::print_welcome();

        if let Some(prompt) = initial_prompt {
            self.process_input(&prompt)?;
        }

        let stdin = io::stdin();
        let mut stdout = io::stdout();

        loop {
            ui::print_prompt();
            stdout.flush()?;

            let mut input = String::new();
            stdin.lock().read_line(&mut input)?;
            let input = input.trim();

            if input.is_empty() {
                continue;
            }

            match input {
                "exit" | "quit" | ".exit" | ":q" => {
                    println!("\n  {} {}\n",
                        "👋".yellow(),
                        "Goodbye!".truecolor(255, 140, 66).bold()
                    );
                    break;
                }
                "help" | ".help" | ":h" => ui::print_help(),
                "status" | ".status" => self.print_status(),
                "clear" | ".clear" | ":c" => ui::clear_screen(),
                "history" | ".history" => self.print_history(),
                "reset" | ".reset" => self.reset_conversation(),
                "config" | ".config" => self.print_config(),
                _ => self.process_input(input)?,
            }
        }

        Ok(())
    }

    fn process_input(&mut self, input: &str) -> anyhow::Result<()> {
        // Show user message with styling
        ui::print_user_message(input);

        let client = ApiClient::new(self.state.settings.clone());

        let api_key = match client.get_api_key() {
            Some(key) => key,
            None => {
                ui::print_error("API key not configured\n\nSet it with:\n  agentrust config set api_key \"your-api-key\"");
                return Ok(());
            }
        };

        self.conversation_history.push(ChatMessage::user(input));

        // 获取工具定义
        let tools = self.get_tool_definitions();

        // MERIT: notify the engine that a new turn is starting, and reset
        // per-turn buffers.
        let metacog_enabled = self.state.settings.metacog.enabled;
        if metacog_enabled {
            self.metacog_call(|e| e.on_turn_start(input));
        }

        let mut iter_index: i32 = 0;
        let mut pending_injection: Option<String> = None;

        // 工具调用循环
        loop {
            // MERIT: ask the engine what to do this iteration.
            if metacog_enabled {
                let decision = self.metacog_call(|e| e.before_next_iteration(iter_index));
                if matches!(decision.action, MetaAction::Abort) {
                    ui::print_error(&format!(
                        "[metacognition] aborting turn: {}",
                        decision.reason
                    ));
                    break;
                }
                if !decision.injection.is_empty() {
                    pending_injection = Some(decision.injection);
                }
                iter_index += 1;
            }

            // Show typing indicator
            ui::print_typing_indicator();

            // Build the message list to send. If MERIT asked for an
            // injection, prepend it as a system message just for this turn
            // (we don't store it in `conversation_history`).
            let mut messages = self.conversation_history.clone();
            if let Some(inj) = pending_injection.take() {
                let mut prepended = Vec::with_capacity(messages.len() + 1);
                prepended.push(ChatMessage::system(inj));
                prepended.extend(messages.into_iter());
                messages = prepended;
            }
            // MERIT Layer 3 self-knowledge → system prompt.
            if metacog_enabled {
                let section = self.metacog_section();
                if !section.is_empty() {
                    let mut prepended = Vec::with_capacity(messages.len() + 1);
                    prepended.push(ChatMessage::system(section));
                    prepended.extend(messages.into_iter());
                    messages = prepended;
                }
            }
            let base_url = client.get_base_url();
            let model = client.get_model().to_string();
            let max_tokens = self.state.settings.api.max_tokens;

            let mut request_body = serde_json::json!({
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "stream": false,
                "temperature": 0.7
            });

            // 注入工具定义
            if !tools.is_empty() {
                request_body["tools"] = serde_json::to_value(&tools)?;
            }

            let http_client = reqwest::blocking::Client::new();
            let url = format!("{}/v1/chat/completions", base_url);

            let resp = match http_client
                .post(&url)
                .header("Authorization", format!("Bearer {}", api_key))
                .header("Content-Type", "application/json")
                .json(&request_body)
                .send()
            {
                Ok(r) => r,
                Err(e) => {
                    ui::print_error(&format!("Request failed: {}", e));
                    return Ok(());
                }
            };

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().unwrap_or_default();
                ui::print_error(&format!("API error ({}): {}", status, body));
                return Ok(());
            }

            let json: serde_json::Value = resp.json().unwrap_or(serde_json::json!({}));

            if let Some(choices) = json.get("choices").and_then(|c| c.as_array()) {
                if let Some(choice) = choices.first() {
                    let message = choice.get("message");

                    // 检查是否有工具调用
                    let tool_calls = message
                        .and_then(|m| m.get("tool_calls"))
                        .and_then(|tc| tc.as_array())
                        .cloned();

                    if let Some(calls) = tool_calls {
                        if !calls.is_empty() {
                            // 打印工具调用信息
                            println!();
                            for call in &calls {
                                if let Some(func) = call.get("function") {
                                    let tool_name = func.get("name")
                                        .and_then(|n| n.as_str())
                                        .unwrap_or("unknown");
                                    println!("  {} Executing tool: {}",
                                        "🔧".truecolor(255, 200, 100),
                                        tool_name.cyan().bold()
                                    );
                                }
                            }
                            println!();

                            // 添加 assistant 消息（带 tool_calls）
                            let tool_calls_parsed: Vec<ToolCall> = calls.iter().filter_map(|call| {
                                let id = call.get("id")?.as_str()?.to_string();
                                let r#type = call.get("type")?.as_str()?.to_string();
                                let func = call.get("function")?;
                                let name = func.get("name")?.as_str()?.to_string();
                                let arguments = func.get("arguments")?.as_str()?.to_string();
                                Some(ToolCall {
                                    id,
                                    r#type,
                                    function: crate::api::ToolCallFunction {
                                        name,
                                        arguments,
                                    },
                                })
                            }).collect();

                            let assistant_msg = ChatMessage {
                                role: "assistant".to_string(),
                                content: message.and_then(|m| m.get("content")).and_then(|c| c.as_str()).map(|s| s.to_string()),
                                tool_calls: Some(tool_calls_parsed),
                                tool_call_id: None,
                                images: Vec::new(),
                            };
                            self.conversation_history.push(assistant_msg);

                            // 执行每个工具调用并添加结果
                            for call in &calls {
                                if let (Some(id), Some(func)) = (
                                    call.get("id").and_then(|i| i.as_str()),
                                    call.get("function")
                                ) {
                                    let tool_name = func.get("name")
                                        .and_then(|n| n.as_str())
                                        .unwrap_or("unknown");
                                    let args_str = func.get("arguments")
                                        .and_then(|a| a.as_str())
                                        .unwrap_or("{}");

                                    let args: serde_json::Value = serde_json::from_str(args_str)
                                        .unwrap_or(serde_json::json!({}));

                                    // MERIT: notify before tool execution.
                                    if metacog_enabled {
                                        let canonical = args.to_string();
                                        let tn = tool_name.to_string();
                                        self.metacog_call(move |e| {
                                            e.on_tool_use(&tn, &canonical)
                                        });
                                    }

                                    // 执行工具
                                    let result = self.execute_tool(tool_name, args);

                                    // MERIT: notify of result (success/error).
                                    if metacog_enabled {
                                        let is_error = result.contains("\"error\"")
                                            || result.contains("\"success\":false");
                                        let tn = tool_name.to_string();
                                        self.metacog_call(move |e| {
                                            e.on_tool_result(&tn, is_error)
                                        });
                                    }

                                    // 添加工具结果消息
                                    let tool_result_msg = ChatMessage::tool(id, result);
                                    self.conversation_history.push(tool_result_msg);
                                }
                            }

                            // 继续循环，让 AI 处理工具结果
                            continue;
                        }
                    }

                    // 没有工具调用，处理普通响应
                    if let Some(content) = message
                        .and_then(|m| m.get("content"))
                        .and_then(|c| c.as_str())
                    {
                        // MERIT: feed the assistant's text into the CoT
                        // monitor as one complete block.
                        if metacog_enabled {
                            let chunk = content.to_string();
                            self.metacog_call(move |e| {
                                e.on_cot_delta(&chunk);
                                e.on_cot_end();
                            });
                        }
                        ui::print_claude_message(content);
                        self.conversation_history.push(ChatMessage::assistant(content.to_string()));

                        // Print token usage if available
                        if let Some(usage) = json.get("usage") {
                            if let (Some(prompt), Some(completion)) = (
                                usage.get("prompt_tokens").and_then(|t| t.as_u64()),
                                usage.get("completion_tokens").and_then(|t| t.as_u64()),
                            ) {
                                let total = prompt + completion;
                                println!("  {} {} prompt · {} generated · {} total",
                                    "◦".truecolor(100, 100, 100),
                                    prompt.to_string().truecolor(150, 150, 150),
                                    completion.to_string().truecolor(150, 150, 150),
                                    total.to_string().truecolor(180, 180, 180)
                                );
                            }
                        }
                    }
                }
            }

            // 退出循环
            break;
        }

        // MERIT: end of turn — flush Layer 3 persistence and trigger
        // Layer 4 schema revision if the cadence is due.
        if metacog_enabled {
            self.metacog_call(|e| e.on_turn_end());
        }

        Ok(())
    }

    /// Synchronously call into the MERIT engine from the blocking REPL.
    ///
    /// The engine lives behind a `tokio::sync::RwLock` on `AppState`, so we
    /// hop into the current Tokio runtime via `block_in_place` + `block_on`.
    fn metacog_call<R>(&self, f: impl FnOnce(&mut crate::metacognition::MetacognitionEngine) -> R) -> R {
        let engine = self.state.metacog.clone();
        tokio::task::block_in_place(move || {
            tokio::runtime::Handle::current().block_on(async move {
                let mut guard = engine.write().await;
                f(&mut *guard)
            })
        })
    }

    /// Borrow the engine to read out the current self-model prompt
    /// section. Empty when Layer 3 has nothing relevant.
    fn metacog_section(&self) -> String {
        let engine = self.state.metacog.clone();
        tokio::task::block_in_place(move || {
            tokio::runtime::Handle::current().block_on(async move {
                let guard = engine.read().await;
                guard.self_model_prompt_section()
            })
        })
    }

    /// 获取 MCP �
