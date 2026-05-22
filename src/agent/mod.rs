//! Agent runtime — the domain-agnostic core of AgentRust.
//!
//! The [`AgentRunner`] drives a `plan → act → reflect` loop on top of the
//! existing [`ApiClient`](crate::api::ApiClient) and
//! [`ToolRegistry`](crate::tools::ToolRegistry). It is intentionally
//! decoupled from the REPL: any front-end (CLI, GUI, Web, WASM) can spin
//! up a runner with a [`Goal`], a [`Persona`], and a chosen set of
//! capability bundles, and drive it to completion.
//!
//! Coding is **one** persona, not the whole product.

pub mod goal;
pub mod persona;
pub mod runner;

pub use goal::{Goal, GoalStatus};
pub use persona::{Persona, PersonaProfile};
pub use runner::{AgentRunner, RunOutcome, RunStep, StepKind, StopReason};
