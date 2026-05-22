//! `AgentRunner` — drives a [`Goal`] to completion via a generic
//! plan-execute-reflect loop.
//!
//! The runner is intentionally headless: it produces a [`RunOutcome`]
//! (and emits structured [`RunStep`]s as it goes) instead of printing.
//! Front-ends (CLI, GUI, Web, WASM) can format the steps however they
//! like.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, warn};

use crate::agent::goal::{Goal, GoalStatus};
use crate::agent::persona::{Persona, PersonaProfile};
use crate::api::{ApiClient, ChatMessage, ChatResponse, ToolDefinition};
use crate::metacognition::{MetaAction, MetacognitionEngine};
use crate::tools::ToolRegistry;

/// Why the runner stopped iterating.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StopReason {
    /// Model produced an assistant message with no tool calls — taken
    /// as a final answer.
    AssistantFinal,
    /// Hit the configured `max_steps` budget.
    MaxStepsReached,
    /// One of the tool calls failed in a way the runner couldn't recover
    /// from.
    ToolError,
    /// API call failed.
    ApiError,
    /// External caller signalled cancellation.
    Cancelled,
}

/// What kind of work happened in a [`RunStep`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StepKind {
    /// The model produced a tool call.
    ToolCall,
    /// The model produced plain text (intermediate thought or final
    /// answer).
    Assistant,
    /// A tool produced output that we fed back to the model.
    ToolResult,
}

/// A single observable step in a run — useful for UIs that want to show
/// the trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunStep {
    pub iteration: usize,
    pub kind: StepKind,
    /// Short label — tool name for tool calls/results, "assistant" for
    /// model text.
    pub label: String,
    /// Payload — JSON-encoded arguments / output, or the assistant text.
    pub payload: String,
}

/// Final result of [`AgentRunner::run`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunOutcome {
    pub goal: Goal,
    pub persona_name: String,
    pub stop_reason: StopReason,
    /// Last assistant text the model produced, if any.
    pub final_answer: Option<String>,
    /// Full trace of steps in order.
    pub steps: Vec<RunStep>,
}

/// Headless agent loop. Construct via [`AgentRunner::new`].
pub struct AgentRunner {
    api: ApiClient,
    tools: Arc<ToolRegistry>,
    persona: Persona,
    profile: PersonaProfile,
    /// Maximum tool-loop iterations before we bail out.
    max_steps: usize,
    /// Optional metacognition engine — when supplied, the runner emits
    /// `on_turn_start` / `before_next_iteration` / `on_tool_use` /
    /// `on_tool_result` / `on_turn_end` events and respects
    /// [`MetaAction::Abort`] decisions.
    metacog: Option<Arc<RwLock<MetacognitionEngine>>>,
}

impl AgentRunner {
    /// Build a runner. `persona.profile()` is resolved once up-front.
    pub fn new(api: ApiClient, tools: Arc<ToolRegistry>, persona: Persona) -> Self {
        let profile = persona.profile();
        Self {
            api,
            tools,
            persona,
            profile,
            max_steps: 12,
            metacog: None,
        }
    }

    /// Builder: cap the loop. Default is 12.
    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps.max(1);
        self
    }

    /// Builder: attach a metacognition engine. The runner will drive
    /// it across the run lifecycle and bail out on `MetaAction::Abort`.
    pub fn with_metacog(mut self, metacog: Arc<RwLock<MetacognitionEngine>>) -> Self {
        self.metacog = Some(metacog);
        self
    }

    /// Drive a goal to completion.
    pub async fn run(&self, mut goal: Goal) -> anyhow::Result<RunOutcome> {
        goal.status = GoalStatus::Running;

        // Notify the metacognition engine that a new task started.
        if let Some(m) = &self.metacog {
            let mut e = m.write().await;
            e.on_turn_start(&goal.objective);
        }

        let system_prompt = self.compose_system_prompt(&goal);
        let mut messages: Vec<ChatMessage> = vec![
            ChatMessage::system(system_prompt),
            ChatMessage::user(format!("Begin. Goal: {}", goal.objective)),
        ];

        let tool_defs = self.collect_tool_definitions();
        let mut steps: Vec<RunStep> = Vec::new();
        let mut final_answer: Option<String> = None;
        let mut stop_reason = StopReason::MaxStepsReached;

        for iter in 0..self.max_steps {
            debug!(iter, persona = %self.persona, "agent iteration");

            // Ask the metacognition engine whether to proceed. An
            // `Abort` decision short-circuits the loop; an `injection`
            // string is prepended as a one-shot system message.
            let mut pending_injection: Option<String> = None;
            if let Some(m) = &self.metacog {
                let mut e = m.write().await;
                let decision = e.before_next_iteration(iter as i32);
                if matches!(decision.action, MetaAction::Abort) {
                    warn!(
                        reason = %decision.reason,
                        "metacognition aborted run"
                    );
                    stop_reason = StopReason::Cancelled;
                    break;
                }
                if !decision.injection.is_empty() {
                    pending_injection = Some(decision.injection);
                }
            }

            // Build the actual request list — clone the running history,
            // then prepend a per-iteration injection plus the persistent
            // self-model section (if any).
            let mut req_messages = messages.clone();
            if let Some(inj) = pending_injection.take() {
                req_messages.insert(0, ChatMessage::system(inj));
            }
            if let Some(m) = &self.metacog {
                let section = {
                    let e = m.read().await;
                    e.self_model_prompt_section()
                };
                if !section.is_empty() {
                    req_messages.insert(0, ChatMessage::system(section));
                }
            }

            let response: ChatResponse = match self
                .api
                .chat(req_messages, tool_defs.clone())
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    warn!(error = %e, "API error in agent loop");
                    stop_reason = StopReason::ApiError;
                    break;
                }
            };

            let choice = match response.choices.into_iter().next() {
                Some(c) => c,
                None => {
                    stop_reason = StopReason::ApiError;
                    break;
                }
            };

            let assistant_msg = choice.message;
            let tool_calls = assistant_msg.tool_calls.clone().unwrap_or_default();
            let text = assistant_msg.content.clone().unwrap_or_default();

            // Always record the assistant turn (text or tool-call
            // intent) so the UI can play it back.
            if !text.is_empty() {
                // Feed the assistant text into the CoT monitor as one
                // complete block — the runner doesn't stream yet, so
                // this is effectively a full chunk per iteration.
                if let Some(m) = &self.metacog {
                    let mut e = m.write().await;
                    e.on_cot_delta(&text);
                    e.on_cot_end();
                }
                steps.push(RunStep {
                    iteration: iter,
                    kind: StepKind::Assistant,
                    label: "assistant".into(),
                    payload: text.clone(),
                });
                final_answer = Some(text);
            }

            // No tool calls → take it as a final answer and stop.
            if tool_calls.is_empty() {
                stop_reason = StopReason::AssistantFinal;
                messages.push(assistant_msg);
                break;
            }

            // Push the assistant tool-call message verbatim so the
            // follow-up `tool` messages have the right correspondences.
            messages.push(assistant_msg);

            // Execute each tool call sequentially. A single failure
            // gets reported back to the model as a tool message — the
            // run continues unless the failure is structural.
            for call in tool_calls {
                let name = call.function.name.clone();
                let args_raw = call.function.arguments.clone();
                let args_json: serde_json::Value = serde_json::from_str(&args_raw)
                    .unwrap_or_else(|_| serde_json::Value::String(args_raw.clone()));

                steps.push(RunStep {
                    iteration: iter,
                    kind: StepKind::ToolCall,
                    label: name.clone(),
                    payload: args_json.to_string(),
                });

                if let Some(m) = &self.metacog {
                    let mut e = m.write().await;
                    e.on_tool_use(&name, &args_json.to_string());
                }

                let exec = self.tools.execute(&name, args_json).await;
                let is_error = exec.is_err();
                let (label, payload) = match exec {
                    Ok(out) => {
                        let serialized = serde_json::to_string(&out)
                            .unwrap_or_else(|_| "<unserialisable>".into());
                        (name.clone(), serialized)
                    }
                    Err(e) => {
                        let msg = format!("tool error: {}", e.message);
                        (name.clone(), msg)
                    }
                };

                if let Some(m) = &self.metacog {
                    let mut e = m.write().await;
                    e.on_tool_result(&name, is_error);
                }

                steps.push(RunStep {
                    iteration: iter,
                    kind: StepKind::ToolResult,
                    label: label.clone(),
                    payload: payload.clone(),
                });

                messages.push(ChatMessage::tool(call.id.clone(), payload));
            }
        }

        // Flush turn-end so Layer 3/4 can persist + schedule revision.
        if let Some(m) = &self.metacog {
            let mut e = m.write().await;
            e.on_turn_end();
        }

        goal.status = match stop_reason {
            StopReason::AssistantFinal => GoalStatus::Succeeded,
            StopReason::MaxStepsReached
            | StopReason::ToolError
            | StopReason::ApiError
            | StopReason::Cancelled => GoalStatus::Failed,
        };

        Ok(RunOutcome {
            goal,
            persona_name: self.profile.name.clone(),
            stop_reason,
            final_answer,
            steps,
        })
    }

    fn compose_system_prompt(&self, goal: &Goal) -> String {
        let mut out = String::new();
        out.push_str(&self.profile.system_prompt);
        out.push_str("\n\n");
        out.push_str(&goal.to_prompt_block());
        out.push_str(
            "\n\n# Loop policy\n\
             - Take one step at a time.\n\
             - Prefer calling a tool over guessing.\n\
             - When the goal is met, reply in plain text with the final \
               answer and no further tool calls.",
        );
        out
    }

    fn collect_tool_definitions(&self) -> Option<Vec<ToolDefinition>> {
        let defs: Vec<ToolDefinition> = self
            .tools
            .list()
            .into_iter()
            .map(|t| {
                ToolDefinition::new(
                    t.name().to_string(),
                    t.description().to_string(),
                    t.input_schema(),
                )
            })
            .collect();
        if defs.is_empty() {
            None
        } else {
            Some(defs)
        }
    }
}
