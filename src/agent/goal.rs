//! Goal representation for an agent run.
//!
//! A [`Goal`] is the unit of work an [`AgentRunner`](super::AgentRunner)
//! consumes. It carries the natural-language objective plus optional
//! deadlines, success criteria, and free-form context.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Lifecycle state of a [`Goal`] as the runner progresses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GoalStatus {
    /// Goal has been accepted but not yet started.
    Pending,
    /// Runner is actively working on the goal.
    Running,
    /// Goal completed successfully.
    Succeeded,
    /// Runner aborted (max-steps, budget, explicit stop, fatal error).
    Failed,
}

/// A high-level objective handed to the agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Goal {
    /// Stable identifier — generated if not supplied.
    pub id: String,
    /// Natural-language objective. This is what gets injected into the
    /// system prompt as `# Goal`.
    pub objective: String,
    /// Optional success criteria. When the runner believes the criteria
    /// are met it can stop on its own.
    pub success_criteria: Vec<String>,
    /// Optional hard deadline — runner will refuse to start new
    /// iterations past this point.
    pub deadline: Option<DateTime<Utc>>,
    /// Free-form context strings appended after the goal (e.g.
    /// references, attached document summaries, user constraints).
    pub context: Vec<String>,
    /// Current status.
    pub status: GoalStatus,
}

impl Goal {
    /// Construct a goal from a free-form objective string.
    pub fn new(objective: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            objective: objective.into(),
            success_criteria: Vec::new(),
            deadline: None,
            context: Vec::new(),
            status: GoalStatus::Pending,
        }
    }

    /// Builder: add a success criterion.
    pub fn with_criterion(mut self, criterion: impl Into<String>) -> Self {
        self.success_criteria.push(criterion.into());
        self
    }

    /// Builder: add a context string.
    pub fn with_context(mut self, ctx: impl Into<String>) -> Self {
        self.context.push(ctx.into());
        self
    }

    /// Builder: set a deadline.
    pub fn with_deadline(mut self, deadline: DateTime<Utc>) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Render the goal as a markdown block to splice into the system
    /// prompt.
    pub fn to_prompt_block(&self) -> String {
        let mut out = String::new();
        out.push_str("# Goal\n");
        out.push_str(&self.objective);
        out.push('\n');

        if !self.success_criteria.is_empty() {
            out.push_str("\n## Success criteria\n");
            for c in &self.success_criteria {
                out.push_str(&format!("- {}\n", c));
            }
        }

        if !self.context.is_empty() {
            out.push_str("\n## Context\n");
            for c in &self.context {
                out.push_str(&format!("- {}\n", c));
            }
        }

        if let Some(deadline) = self.deadline {
            out.push_str(&format!("\n## Deadline\n{}\n", deadline.to_rfc3339()));
        }

        out
    }
}
