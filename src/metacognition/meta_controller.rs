//! Layer 1 (meta-action policy) — `MetaController`.
//!
//! Picks one of `Act / Reflect / Decompose / Escalate / Abort` per iteration
//! by minimising Expected Free Energy (EFE) over a `SelfBelief`. Costs,
//! progress weights, and epistemic γ are configurable; the defaults are the
//! ones tuned in `agent/MetaController.cpp` in the AgentCpp tree.

use serde::{Deserialize, Serialize};

use super::self_belief::SelfBelief;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetaAction {
    Act,
    Reflect,
    Decompose,
    Escalate,
    Abort,
}

impl MetaAction {
    pub fn as_str(&self) -> &'static str {
        match self {
            MetaAction::Act => "act",
            MetaAction::Reflect => "reflect",
            MetaAction::Decompose => "decompose",
            MetaAction::Escalate => "escalate",
            MetaAction::Abort => "abort",
        }
    }
}

/// Why the model just stopped streaming (best-effort; populated by the agent
/// loop). Used to bias the EFE policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum StopReason {
    #[default]
    Unknown,
    EndTurn,
    ToolUse,
    MaxTokens,
    Refusal,
}

/// What the agent loop reports about each iteration before asking the
/// controller for its next action.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IterationObservation {
    pub turn: i32,
    pub tool_calls: i32,
    pub tool_errors: i32,
    pub any_progress: bool,
    pub loop_detected: bool,
    pub low_quality_cot: bool,
    pub stop_reason: StopReason,
}

/// What the controller returns to the agent loop.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MetaDecision {
    pub action: MetaAction,
    /// Ready-to-inject prompt fragment. Empty when `action == Act`.
    pub injection: String,
    pub reason: String,
    /// EFE score of the chosen action (for diagnostics).
    pub efe_score: f64,
}

impl Default for MetaAction {
    fn default() -> Self {
        MetaAction::Act
    }
}

/// Tunables for the EFE meta-policy. Defaults are calibrated in
/// `agent/MetaController.cpp`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaConfig {
    pub cost_act: f64,
    pub cost_reflect: f64,
    pub cost_decompose: f64,
    pub cost_escalate: f64,
    pub cost_abort: f64,
    pub progress_weight: f64,
    pub gamma_init: f64,
    pub gamma_min: f64,
    pub gamma_decay: f64,
    pub abort_min_iters: i32,
    pub abort_progress_thresh: f64,
    pub reflect_after_failures: i32,
    pub decompose_after_stalls: i32,
    pub allow_abort: bool,
}

impl Default for MetaConfig {
    fn default() -> Self {
        Self {
            cost_act: 1.0,
            cost_reflect: 1.4,
            cost_decompose: 2.0,
            cost_escalate: 0.3,
            cost_abort: 0.1,
            progress_weight: 5.0,
            gamma_init: 1.5,
            gamma_min: 0.1,
            gamma_decay: 0.85,
            abort_min_iters: 4,
            abort_progress_thresh: 0.25,
            reflect_after_failures: 2,
            decompose_after_stalls: 3,
            allow_abort: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaController {
    cfg: MetaConfig,
    belief: SelfBelief,
    iter: i32,
    consecutive_errors: i32,
    stall_count: i32,
    total_reflects: i32,
    total_aborts: i32,
    turn_started: bool,
    last_user_input: String,
}

impl Default for MetaController {
    fn default() -> Self {
        Self::with_config(MetaConfig::default())
    }
}

impl MetaController {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_config(cfg: MetaConfig) -> Self {
        Self {
            cfg,
            belief: SelfBelief::new(),
            iter: 0,
            consecutive_errors: 0,
            stall_count: 0,
            total_reflects: 0,
            total_aborts: 0,
            turn_started: false,
            last_user_input: String::new(),
        }
    }

    pub fn belief(&self) -> &SelfBelief {
        &self.belief
    }

    pub fn belief_mut(&mut self) -> &mut SelfBelief {
        &mut self.belief
    }

    pub fn config(&self) -> &MetaConfig {
        &self.cfg
    }

    /// Drop all per-run state. Belief is reset to the Jeffreys prior on all
    /// default dimensions.
    pub fn reset(&mut self) {
        self.belief = SelfBelief::new();
        self.iter = 0;
        self.consecutive_errors = 0;
        self.stall_count = 0;
        self.total_reflects = 0;
        self.total_aborts = 0;
        self.turn_started = false;
        self.last_user_input.clear();
    }

    pub fn on_turn_start(&mut self, user_input: &str) {
        self.turn_started = true;
        self.last_user_input = user_input.to_string();
        self.consecutive_errors = 0;
        self.stall_count = 0;
        self.iter = 0;
    }

    pub fn record_observation(&mut self, obs: &IterationObservation) {
        self.iter = obs.turn;
        let clean = obs.tool_errors == 0
            && obs.any_progress
            && obs.tool_calls > 0
            && !obs.loop_detected
            && !obs.low_quality_cot;
        if clean {
            self.stall_count = 0;
        } else if obs.tool_errors > 0 || !obs.any_progress || obs.loop_detected {
            self.stall_count += 1;
        }
    }

    /// Map a tool name to the dimension it informs, then update the belief.
    pub fn observe_tool_result(&mut self, tool_name: &str, is_error: bool) {
        let dim = match tool_name {
            t if Self::is_code_edit_tool(t) => "code_edit",
            t if Self::is_search_tool(t) => "search",
            _ => "tool_use",
        };
        if is_error {
            self.belief.observe_failure(dim, 1.0);
            self.belief.observe_failure("tool_use", 0.5);
            self.consecutive_errors += 1;
        } else {
            self.belief.observe_success(dim, 1.0);
            self.belief.observe_success("tool_use", 0.5);
            self.consecutive_errors = 0;
        }
    }

    /// Layer-2 quality signal in `[0,1]`.
    pub fn observe_cot_quality(&mut self, q: f64) {
        let q = q.clamp(0.0, 1.0);
        self.belief.observe_mixed("reasoning", q, 1.0);
    }

    /// Update `task_progress`. A `false` here is half-weight so a noisy
    /// signal doesn't dominate.
    pub fn observe_progress(&mut self, made: bool) {
        if made {
            self.belief.observe_success("task_progress", 1.0);
        } else {
            self.belief.observe_failure("task_progress", 0.5);
        }
    }

    /// Compute EFE for all five actions and pick the minimum.
    pub fn decide(&mut self, iter_index: i32) -> MetaDecision {
        self.iter = iter_index;
        let candidates = [
            MetaAction::Act,
            MetaAction::Reflect,
            MetaAction::Decompose,
            MetaAction::Escalate,
            MetaAction::Abort,
        ];
        let mut best = MetaAction::Act;
        let mut best_score = f64::INFINITY;
        for a in candidates {
            let s = self.expected_free_energy(a);
            if s < best_score {
                best_score = s;
                best = a;
            }
        }

        let (injection, reason) = match best {
            MetaAction::Reflect => {
                self.total_reflects += 1;
                let inj = format!(
                    "[Metacognition / reflect] You have hit {} consecutive tool error(s). \
                     Before the next action, briefly state in 2 lines: \
                     (a) what assumption is most likely wrong, and \
                     (b) one concretely different strategy you will try now.",
                    self.consecutive_errors.max(1)
                );
                let r = format!(
                    "consecutive_errors={}, reasoning_mean={:.2}",
                    self.consecutive_errors,
                    self.belief.mean("reasoning")
                );
                (inj, r)
            }
            MetaAction::Decompose => {
                let inj = format!(
                    "[Metacognition / decompose] Progress has been stuck for {} iterations. \
                     Decompose the remaining work into 2-4 smaller subtasks and tackle the \
                     most independent one first.",
                    self.stall_count.max(1)
                );
                let r = format!(
                    "stall_count={}, task_progress_mean={:.2}",
                    self.stall_count,
                    self.belief.mean("task_progress")
                );
                (inj, r)
            }
            MetaAction::Escalate => {
                let progress = self.belief.mean("task_progress");
                let inj = format!(
                    "[Metacognition / escalate] After {} iterations the estimated \
                     probability of success is only {:.2}. Consider asking the user a \
                     specific clarifying question before continuing.",
                    self.iter, progress
                );
                let r = format!("iter={}, progress={:.2}", self.iter, progress);
                (inj, r)
            }
            MetaAction::Abort => {
                self.total_aborts += 1;
                (
                    String::new(),
                    format!(
                        "iter={}, progress={:.2}, uncertainty={:.2}",
                        self.iter,
                        self.belief.mean("task_progress"),
                        self.belief.overall_uncertainty()
                    ),
                )
            }
            MetaAction::Act => (String::new(), String::from("act on best estimate")),
        };

        MetaDecision {
            action: best,
            injection,
            reason,
            efe_score: best_score,
        }
    }

    pub fn iteration(&self) -> i32 {
        self.iter
    }
    pub fn consecutive_errors(&self) -> i32 {
        self.consecutive_errors
    }
    pub fn stall_count(&self) -> i32 {
        self.stall_count
    }
    pub fn total_reflects(&self) -> i32 {
        self.total_reflects
    }
    pub fn total_aborts(&self) -> i32 {
        self.total_aborts
    }

    // ── EFE machinery ─────────────────────────────────────────────────

    fn current_gamma(&self) -> f64 {
        let g = self.cfg.gamma_init * self.cfg.gamma_decay.powi(self.iter.max(0));
        g.max(self.cfg.gamma_min)
    }

    fn expected_free_energy(&self, action: MetaAction) -> f64 {
        let progress = self.belief.mean("task_progress");
        let reasoning = self.belief.mean("reasoning");
        let tool = self.belief.mean("tool_use");
        let uncertainty = self.belief.overall_uncertainty();
        let gen_ig = uncertainty;
        let pw = self.cfg.progress_weight;

        let (c, u, info) = match action {
            MetaAction::Act => {
                let c = self.cfg.cost_act;
                let u = pw * progress * 0.5 * (tool + reasoning);
                (c, u, 0.5 * gen_ig)
            }
            MetaAction::Reflect => {
                let c = self.cfg.cost_reflect;
                let mut u = pw * (1.0 - reasoning) * 1.2;
                if self.consecutive_errors >= self.cfg.reflect_after_failures {
                    u += 2.0;
                }
                (c, u, 1.2 * gen_ig)
            }
            MetaAction::Decompose => {
                let c = self.cfg.cost_decompose;
                let mut u = pw * (1.0 - progress) * 0.8;
                if self.stall_count >= self.cfg.decompose_after_stalls {
                    u += 2.0;
                }
                (c, u, 1.5 * gen_ig)
            }
            MetaAction::Escalate => {
                let c = self.cfg.cost_escalate;
                let mut u = pw * (1.0 - progress) * 0.5;
                if self.iter >= 6 && progress < 0.3 {
                    u += 1.5;
                }
                (c, u, 0.0)
            }
            MetaAction::Abort => {
                if !self.cfg.allow_abort || self.iter < self.cfg.abort_min_iters {
                    return f64::INFINITY;
                }
                let c = self.cfg.cost_abort;
                let mut u = if progress < self.cfg.abort_progress_thresh {
                    3.0
                } else {
                    -2.0
                };
                u -= 3.0 * uncertainty;
                (c, u, 0.0)
            }
        };

        c - u - self.current_gamma() * info
    }

    fn is_code_edit_tool(name: &str) -> bool {
        matches!(
            name,
            "file_edit"
                | "file_write"
                | "FileEdit"
                | "FileWrite"
                | "Edit"
                | "Write"
                | "note_edit"
        )
    }
    fn is_search_tool(name: &str) -> bool {
        matches!(
            name,
            "search" | "glob_tool" | "Grep" | "Glob" | "GrepTool" | "GlobTool" | "list_files"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn act_chosen_when_things_are_fine() {
        let mut mc = MetaController::new();
        // Pretend lots of successful evidence on every dimension.
        for _ in 0..30 {
            mc.observe_tool_result("execute_command", false);
            mc.observe_cot_quality(0.9);
            mc.observe_progress(true);
        }
        let d = mc.decide(2);
        assert_eq!(d.action, MetaAction::Act);
    }

    #[test]
    fn reflect_chosen_after_consecutive_errors() {
        let mut mc = MetaController::new();
        // Three errors in a row should beat Act under default costs.
        mc.observe_tool_result("execute_command", true);
        mc.observe_tool_result("execute_command", true);
        mc.observe_tool_result("execute_command", true);
        let d = mc.decide(1);
        assert_eq!(d.action, MetaAction::Reflect);
        assert!(d.injection.contains("reflect"));
    }

    #[test]
    fn abort_blocked_before_min_iters() {
        let mut mc = MetaController::new();
        // Even with terrible signals, abort isn't allowed before iter 4.
        for _ in 0..10 {
            mc.observe_tool_result("execute_command", true);
            mc.observe_progress(false);
        }
        let d = mc.decide(1);
        assert_ne!(d.action, MetaAction::Abort);
    }
}
