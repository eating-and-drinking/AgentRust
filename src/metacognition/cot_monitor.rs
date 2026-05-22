//! Layer 2 (process monitoring) — `CoTMonitor`.
//!
//! Sliding-window analysis of the model's chain-of-thought trace and tool
//! call history. Detects:
//!
//! * **Step repetition** — the most recent normalised step matches an
//!   earlier one in the look-back window.
//! * **Low-content stalling** — most recent steps are very short.
//! * **Tool-parameter loops** — same `tool|canonical_input` reappears within
//!   the loop window.
//!
//! Outputs a `quality ∈ [0,1]` score and (optionally) an intervention prompt
//! the controller can splice into the next turn.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoTConfig {
    pub step_min_chars: usize,
    pub repeat_window: usize,
    pub loop_window: usize,
    pub low_quality_thresh: f64,
    pub enable_injection: bool,
}

impl Default for CoTConfig {
    fn default() -> Self {
        Self {
            step_min_chars: 60,
            repeat_window: 4,
            loop_window: 5,
            low_quality_thresh: 0.4,
            enable_injection: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CoTMonitor {
    cfg: CoTConfig,
    current_buf: String,
    recent_steps: VecDeque<String>,
    recent_tools: VecDeque<String>,
    loop_detected: bool,
    low_quality: bool,
    any_text: bool,
    last_quality: f64,
}

impl Default for CoTMonitor {
    fn default() -> Self {
        Self::with_config(CoTConfig::default())
    }
}

impl CoTMonitor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_config(cfg: CoTConfig) -> Self {
        Self {
            cfg,
            current_buf: String::new(),
            recent_steps: VecDeque::new(),
            recent_tools: VecDeque::new(),
            loop_detected: false,
            low_quality: false,
            any_text: false,
            last_quality: 0.5,
        }
    }

    pub fn config(&self) -> &CoTConfig {
        &self.cfg
    }

    /// Drop everything (per-turn).
    pub fn on_turn_start(&mut self) {
        self.current_buf.clear();
        self.recent_steps.clear();
        self.recent_tools.clear();
        self.loop_detected = false;
        self.low_quality = false;
        self.any_text = false;
        self.last_quality = 0.5;
    }

    /// Reset per-iteration flags. Keep the step/tool history.
    pub fn reset_iteration(&mut self) {
        self.current_buf.clear();
        self.loop_detected = false;
        self.low_quality = false;
        self.any_text = false;
        self.last_quality = 0.5;
    }

    /// Append a text delta to the current accumulating step. Splits on
    /// blank-line boundaries internally.
    pub fn on_cot_delta(&mut self, delta: &str) {
        if !delta.is_empty() {
            self.any_text = true;
        }
        self.current_buf.push_str(delta);
        // Eagerly close steps on every blank-line boundary so a long text
        // block doesn't all collapse into a single "step".
        while let Some(idx) = self.current_buf.find("\n\n") {
            let step: String = self.current_buf.drain(..idx).collect();
            // Drop the two newlines.
            self.current_buf.drain(..2.min(self.current_buf.len()));
            self.commit_step(&step);
        }
    }

    /// Mark end of a CoT block. Any unflushed buffer becomes the trailing
    /// step. Recomputes quality and pathology flags.
    pub fn on_cot_block_end(&mut self) {
        if !self.current_buf.is_empty() {
            let step = std::mem::take(&mut self.current_buf);
            self.commit_step(&step);
        }
        self.recompute_quality();
    }

    fn commit_step(&mut self, step: &str) {
        let n = normalize(step);
        if n.is_empty() {
            return;
        }
        self.recent_steps.push_back(n);
        // Keep at most repeat_window+1 so we can compare last to the earlier ones.
        while self.recent_steps.len() > self.cfg.repeat_window + 1 {
            self.recent_steps.pop_front();
        }
    }

    fn recompute_quality(&mut self) {
        // Repetition: last step matches any of the earlier ones in the window.
        let mut repeating = false;
        if let Some(last) = self.recent_steps.back() {
            for prior in self.recent_steps.iter().rev().skip(1).take(self.cfg.repeat_window) {
                if prior == last {
                    repeating = true;
                    break;
                }
            }
        }

        // Low-content: how many of the last `repeat_window` steps are short?
        let mut low_count = 0;
        for step in self.recent_steps.iter().rev().take(self.cfg.repeat_window) {
            if step.chars().count() < self.cfg.step_min_chars {
                low_count += 1;
            }
        }
        let too_short = self.cfg.repeat_window > 0 && low_count >= self.cfg.repeat_window;

        let mut q: f64 = 1.0;
        if repeating {
            q -= 0.5;
        }
        if too_short {
            q -= 0.3;
        }
        if !self.any_text {
            q = 0.5;
        }
        let q = q.clamp(0.0, 1.0);
        self.last_quality = q;
        self.low_quality = q < self.cfg.low_quality_thresh;
    }

    /// Record a tool call with its **canonical** input string. Sets the
    /// `loop_detected` flag if the exact pair reoccurs within the window.
    pub fn on_tool_call(&mut self, tool_name: &str, input_canonical: &str) {
        let key = format!("{}|{}", tool_name, input_canonical);
        let seen = self.recent_tools.iter().any(|k| k == &key);
        if seen {
            self.loop_detected = true;
        }
        self.recent_tools.push_back(key);
        while self.recent_tools.len() > self.cfg.loop_window {
            self.recent_tools.pop_front();
        }
    }

    pub fn quality(&self) -> f64 {
        self.last_quality
    }
    pub fn loop_detected(&self) -> bool {
        self.loop_detected
    }
    pub fn low_quality(&self) -> bool {
        self.low_quality
    }
    pub fn any_text_content(&self) -> bool {
        self.any_text
    }

    /// Return a single-sentence intervention to splice into the next turn,
    /// or `None` if nothing's wrong.
    pub fn propose_intervention(&self) -> Option<String> {
        if !self.cfg.enable_injection {
            return None;
        }
        if self.loop_detected {
            return Some(
                "[Metacognition / process] You are about to call a tool with effectively the \
                 same input as a very recent attempt that did not advance the task. Before the \
                 next action, briefly state in one line: (a) what changed since the prior \
                 attempt, and (b) what concretely different thing you will try now."
                    .to_string(),
            );
        }
        if self.low_quality {
            return Some(
                "[Metacognition / process] Your recent reasoning steps look thin or repetitive. \
                 Please write one substantive next step (≥ 2 sentences) describing concretely \
                 what you will try and why."
                    .to_string(),
            );
        }
        None
    }
}

/// Lowercase, collapse all whitespace to single spaces, trim ends.
fn normalize(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_space = true;
    for c in s.chars() {
        if c.is_whitespace() {
            if !prev_space {
                out.push(' ');
                prev_space = true;
            }
        } else {
            for low in c.to_lowercase() {
                out.push(low);
            }
            prev_space = false;
        }
    }
    out.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_loop_triggers_detection() {
        let mut m = CoTMonitor::new();
        m.on_tool_call("file_read", "path=/etc/passwd");
        m.on_tool_call("file_read", "path=/etc/passwd");
        assert!(m.loop_detected());
        assert!(m.propose_intervention().is_some());
    }

    #[test]
    fn distinct_inputs_dont_trip_loop() {
        let mut m = CoTMonitor::new();
        m.on_tool_call("file_read", "path=/a");
        m.on_tool_call("file_read", "path=/b");
        assert!(!m.loop_detected());
    }

    #[test]
    fn short_repeated_steps_lower_quality() {
        let mut m = CoTMonitor::new();
        for _ in 0..5 {
            m.on_cot_delta("hmm\n\n");
        }
        m.on_cot_block_end();
        assert!(m.quality() < 0.9);
    }

    #[test]
    fn rich_unique_steps_keep_quality_high() {
        let mut m = CoTMonitor::new();
        let steps = [
            "First, I will read the configuration file to understand how the agent is set up and which model it uses.\n\n",
            "Next, based on the model identifier, I'll branch and pick an appropriate tokenizer for the prompt budget calculation step.\n\n",
            "Finally, I'll run the budget check and emit a structured summary so downstream consumers can verify the prompt fits within limits.\n\n",
        ];
        for s in steps {
            m.on_cot_delta(s);
        }
        m.on_cot_block_end();
        assert!(m.quality() > 0.9, "got {}", m.quality());
    }
}
