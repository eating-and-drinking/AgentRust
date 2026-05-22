//! Façade over Layers 1–4 — `MetacognitionEngine`.
//!
//! Single entrypoint the agent loop talks to. Per turn:
//!
//! ```text
//!   on_turn_start(user_input)
//!   for i in 0..max_iters:
//!     dec = before_next_iteration(i)
//!     if dec.action == Abort: break
//!     // splice dec.injection into the next system prompt slot
//!     stream LLM response:
//!       for delta in text: on_cot_delta(delta)
//!       on_cot_end()
//!     for t in tool_uses:
//!       on_tool_use(t.name, canonical(t.input))
//!       run_tool(t)
//!       on_tool_result(t.name, is_error)
//!   on_turn_end()
//! ```
//!
//! `self_model_prompt_section()` is what the loop should append to the
//! system prompt before sending it to the model.

use serde::{Deserialize, Serialize};

use super::cot_monitor::{CoTConfig, CoTMonitor};
use super::meta_controller::{
    IterationObservation, MetaAction, MetaConfig, MetaController, MetaDecision, StopReason,
};
use super::schema_reviser::{FailureEvent, ReviserConfig, SchemaReviser};
use super::self_model_store::SelfModelStore;

/// Diagnostic event emitted by the engine. The loop can subscribe via
/// `drain_events()` after each call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetacognitionEvent {
    pub kind: String,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    pub layer1: MetaConfig,
    pub layer2: CoTConfig,
    pub layer4: ReviserConfig,
    pub prompt_top_k: usize,
    pub enable_layer3: bool,
    pub enable_layer4: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            layer1: MetaConfig::default(),
            layer2: CoTConfig::default(),
            layer4: ReviserConfig::default(),
            prompt_top_k: 3,
            enable_layer3: true,
            enable_layer4: true,
        }
    }
}

pub struct MetacognitionEngine {
    cfg: EngineConfig,
    controller: MetaController,
    cot: CoTMonitor,
    store: SelfModelStore,
    reviser: SchemaReviser,
    iter: i32,
    tool_calls_in_iter: i32,
    tool_errors_in_iter: i32,
    turn_tool_errors: i32,
    turn_tool_calls: i32,
    current_task: String,
    events: Vec<MetacognitionEvent>,
}

impl Default for MetacognitionEngine {
    fn default() -> Self {
        Self::with_config(EngineConfig::default())
    }
}

impl MetacognitionEngine {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_config(cfg: EngineConfig) -> Self {
        let controller = MetaController::with_config(cfg.layer1.clone());
        let cot = CoTMonitor::with_config(cfg.layer2.clone());
        let reviser = SchemaReviser::with_config(cfg.layer4.clone());
        Self {
            cfg,
            controller,
            cot,
            store: SelfModelStore::new(),
            reviser,
            iter: 0,
            tool_calls_in_iter: 0,
            tool_errors_in_iter: 0,
            turn_tool_errors: 0,
            turn_tool_calls: 0,
            current_task: String::new(),
            events: Vec::new(),
        }
    }

    // ── Public accessors (for tests, diagnostics, persistence) ────────

    pub fn config(&self) -> &EngineConfig {
        &self.cfg
    }

    pub fn controller(&self) -> &MetaController {
        &self.controller
    }
    pub fn controller_mut(&mut self) -> &mut MetaController {
        &mut self.controller
    }

    pub fn cot(&self) -> &CoTMonitor {
        &self.cot
    }

    pub fn store(&self) -> &SelfModelStore {
        &self.store
    }
    pub fn store_mut(&mut self) -> &mut SelfModelStore {
        &mut self.store
    }

    pub fn reviser(&self) -> &SchemaReviser {
        &self.reviser
    }
    pub fn reviser_mut(&mut self) -> &mut SchemaReviser {
        &mut self.reviser
    }

    /// Drain accumulated diagnostic events. Caller usually logs them.
    pub fn drain_events(&mut self) -> Vec<MetacognitionEvent> {
        std::mem::take(&mut self.events)
    }

    fn emit(&mut self, kind: &str, detail: impl Into<String>) {
        self.events.push(MetacognitionEvent {
            kind: kind.to_string(),
            detail: detail.into(),
        });
    }

    // ── Lifecycle ─────────────────────────────────────────────────────

    pub fn on_turn_start(&mut self, user_input: &str) {
        self.controller.on_turn_start(user_input);
        self.cot.on_turn_start();
        self.iter = 0;
        self.tool_calls_in_iter = 0;
        self.tool_errors_in_iter = 0;
        self.turn_tool_errors = 0;
        self.turn_tool_calls = 0;
        self.current_task = user_input.to_string();
        self.emit("turn_start", format!("len={}", user_input.len()));
    }

    pub fn on_turn_end(&mut self) {
        if self.cfg.enable_layer4 {
            self.reviser.note_episode_complete();
            if self.reviser.should_review() {
                let props = self
                    .reviser
                    .propose_revisions(self.controller.belief(), &self.store);
                for p in &props {
                    let applied = self.reviser.apply(
                        p,
                        self.controller.belief_mut(),
                        &mut self.store,
                    );
                    if applied {
                        self.events.push(MetacognitionEvent {
                            kind: "schema_revise".to_string(),
                            detail: format!(
                                "applied {:?}: {} (novelty={:.2})",
                                p.kind, p.name_or_text, p.novelty
                            ),
                        });
                    }
                }
            }
        }
        if self.cfg.enable_layer3 {
            self.store.save_to_external();
        }
    }

    // ── Streaming hooks ───────────────────────────────────────────────

    pub fn on_cot_delta(&mut self, delta: &str) {
        self.cot.on_cot_delta(delta);
    }

    pub fn on_cot_end(&mut self) {
        self.cot.on_cot_block_end();
        self.controller.observe_cot_quality(self.cot.quality());
        if self.cot.low_quality() {
            self.emit(
                "cot_warn",
                format!("low_quality q={:.2}", self.cot.quality()),
            );
        }
        if self.cot.loop_detected() {
            self.emit("cot_warn", "tool_loop_detected");
        }
    }

    pub fn on_tool_use(&mut self, name: &str, input_canonical: &str) {
        self.cot.on_tool_call(name, input_canonical);
        self.tool_calls_in_iter += 1;
        self.turn_tool_calls += 1;
    }

    pub fn on_tool_result(&mut self, name: &str, is_error: bool) {
        self.controller.observe_tool_result(name, is_error);
        if is_error {
            self.tool_errors_in_iter += 1;
            self.turn_tool_errors += 1;
            if self.cfg.enable_layer4 {
                self.reviser.record_failure(FailureEvent {
                    ts: 0,
                    tool: name.to_string(),
                    description: format!("{} failed", name),
                    task_type: String::new(),
                });
            }
        }
    }

    pub fn observe_progress(&mut self, made: bool) {
        self.controller.observe_progress(made);
    }

    // ── Iteration policy ──────────────────────────────────────────────

    pub fn before_next_iteration(&mut self, iter_index: i32) -> MetaDecision {
        // Build observation from per-iteration counters.
        let obs = IterationObservation {
            turn: iter_index,
            tool_calls: self.tool_calls_in_iter,
            tool_errors: self.tool_errors_in_iter,
            any_progress: self.tool_calls_in_iter > 0 && self.tool_errors_in_iter == 0,
            loop_detected: self.cot.loop_detected(),
            low_quality_cot: self.cot.low_quality(),
            stop_reason: StopReason::Unknown,
        };
        self.controller.record_observation(&obs);
        let mut decision = self.controller.decide(iter_index);

        // CoTMonitor can override with a process-level intervention if the
        // controller picked Act but the process signals a pathology.
        if matches!(decision.action, MetaAction::Act) {
            if let Some(inj) = self.cot.propose_intervention() {
                decision.injection = inj;
                decision.reason = format!("cot_intervention; {}", decision.reason);
                self.emit("intervention", "cot_pathology");
            }
        } else {
            self.emit(
                "decision",
                format!("{} efe={:.2}", decision.action.as_str(), decision.efe_score),
            );
        }

        if matches!(decision.action, MetaAction::Abort) {
            self.emit("abort", decision.reason.clone());
        }

        // Reset per-iteration counters and CoT iteration flags.
        self.iter = iter_index + 1;
        self.tool_calls_in_iter = 0;
        self.tool_errors_in_iter = 0;
        self.cot.reset_iteration();
        decision
    }

    /// Markdown block to append to the system prompt. Empty if Layer 3 is
    /// disabled or the store has nothing relevant.
    pub fn self_model_prompt_section(&self) -> String {
        if !self.cfg.enable_layer3 {
            return String::new();
        }
        self.store
            .render_for_prompt(&self.current_task, self.cfg.prompt_top_k)
    }

    pub fn turn_tool_stats(&self) -> (i32, i32) {
        (self.turn_tool_calls, self.turn_tool_errors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_iter_returns_a_valid_non_abort_action() {
        let mut e = MetacognitionEngine::new();
        e.on_turn_start("write a hello world program");
        let d = e.before_next_iteration(0);
        // With a Jeffreys prior and progress_weight=5.0 the EFE policy
        // typically prefers Reflect first (utility of resolving epistemic
        // uncertainty dominates the per-action cost). What we DO guarantee
        // at iter 0 is that Abort is impossible (gated by `abort_min_iters`).
        assert_ne!(d.action, MetaAction::Abort);
    }

    #[test]
    fn reflect_kicks_in_after_two_tool_errors() {
        let mut e = MetacognitionEngine::new();
        e.on_turn_start("edit a file");
        e.on_tool_use("file_edit", "path=/a");
        e.on_tool_result("file_edit", true);
        e.on_tool_use("file_edit", "path=/b");
        e.on_tool_result("file_edit", true);
        let d = e.before_next_iteration(1);
        assert_eq!(d.action, MetaAction::Reflect);
        assert!(!d.injection.is_empty());
    }

    #[test]
    fn self_model_section_empty_when_no_props() {
        let mut e = MetacognitionEngine::new();
        e.on_turn_start("anything");
        assert!(e.self_model_prompt_section().is_empty());
    }
}
