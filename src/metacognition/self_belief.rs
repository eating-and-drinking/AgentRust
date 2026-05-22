//! Layer 1 (outcome monitoring) — `SelfBelief`.
//!
//! Bayesian belief over a set of named competence dimensions. Each dimension
//! is a `Beta(α, β)` distribution. Successes nudge α, failures nudge β,
//! mixed observations split the weight. The controller (`MetaController`)
//! consumes the means and the entropies/uncertainties to drive its
//! Expected-Free-Energy meta-action policy.
//!
//! Faithful Rust port of `agent/SelfBelief.{hpp,cpp}` in the AgentCpp tree.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// One `Beta(α, β)` distribution over a competence dimension.
///
/// `α = 1.0, β = 1.0` is the Jeffreys / uniform prior used as the default
/// for a freshly-registered dimension.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BetaParam {
    pub alpha: f64,
    pub beta: f64,
}

impl Default for BetaParam {
    fn default() -> Self {
        Self { alpha: 1.0, beta: 1.0 }
    }
}

impl BetaParam {
    pub fn new(alpha: f64, beta: f64) -> Self {
        Self { alpha, beta }
    }

    /// `mean = α / (α + β)`; returns 0.5 if both params are zero.
    pub fn mean(&self) -> f64 {
        let s = self.alpha + self.beta;
        if s > 0.0 { self.alpha / s } else { 0.5 }
    }

    /// `variance = αβ / ((α+β)² · (α+β+1))`.
    pub fn variance(&self) -> f64 {
        let s = self.alpha + self.beta;
        if s <= 0.0 {
            return 0.0;
        }
        (self.alpha * self.beta) / (s * s * (s + 1.0))
    }

    /// `stddev = sqrt(variance)`.
    pub fn stddev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Differential entropy of the Beta distribution, in nats:
    ///
    /// ```text
    /// H(α, β) = ln B(α, β) − (α − 1) ψ(α) − (β − 1) ψ(β) + (α + β − 2) ψ(α + β)
    /// ```
    pub fn entropy(&self) -> f64 {
        if self.alpha <= 0.0 || self.beta <= 0.0 {
            return 0.0;
        }
        let ln_b = lgamma(self.alpha) + lgamma(self.beta) - lgamma(self.alpha + self.beta);
        ln_b
            - (self.alpha - 1.0) * digamma(self.alpha)
            - (self.beta - 1.0) * digamma(self.beta)
            + (self.alpha + self.beta - 2.0) * digamma(self.alpha + self.beta)
    }

    /// Expected information gain (in nats) from one Bernoulli observation
    /// drawn from the current belief — `H_prior − E[H_posterior]`. Clamped
    /// at zero (numerical noise can produce tiny negative values).
    pub fn expected_info_gain_bernoulli(&self) -> f64 {
        let h_prior = self.entropy();
        let p_success = self.mean();
        let p_fail = 1.0 - p_success;
        let post_success = BetaParam::new(self.alpha + 1.0, self.beta);
        let post_fail = BetaParam::new(self.alpha, self.beta + 1.0);
        let h_post = p_success * post_success.entropy() + p_fail * post_fail.entropy();
        (h_prior - h_post).max(0.0)
    }
}

/// Belief state over an ordered set of named competence dimensions.
///
/// Iteration order is preserved via a separate `order` vector so the prompt
/// and event renderings are stable across runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfBelief {
    /// Dimension name → Beta parameters.
    params: BTreeMap<String, BetaParam>,
    /// Stable iteration order (insertion order).
    order: Vec<String>,
}

impl Default for SelfBelief {
    /// Seed with the five default competence dimensions used by
    /// `MetaController`: `tool_use`, `code_edit`, `search`, `reasoning`,
    /// `task_progress`. All start at the uniform Jeffreys prior.
    fn default() -> Self {
        let mut s = Self {
            params: BTreeMap::new(),
            order: Vec::new(),
        };
        for d in [
            "tool_use",
            "code_edit",
            "search",
            "reasoning",
            "task_progress",
        ] {
            s.add_dimension(d, 1.0, 1.0);
        }
        s
    }
}

impl SelfBelief {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new competence dimension. No-op if one with the same name
    /// already exists.
    pub fn add_dimension(&mut self, name: &str, alpha: f64, beta: f64) {
        if self.params.contains_key(name) {
            return;
        }
        self.params.insert(name.to_string(), BetaParam::new(alpha, beta));
        self.order.push(name.to_string());
    }

    pub fn has_dimension(&self, name: &str) -> bool {
        self.params.contains_key(name)
    }

    /// Stable iteration order over dimension names.
    pub fn dimensions(&self) -> &[String] {
        &self.order
    }

    /// `α += weight` (clean success).
    pub fn observe_success(&mut self, dim: &str, weight: f64) {
        if let Some(p) = self.params.get_mut(dim) {
            p.alpha += weight;
        }
    }

    /// `β += weight` (clean failure).
    pub fn observe_failure(&mut self, dim: &str, weight: f64) {
        if let Some(p) = self.params.get_mut(dim) {
            p.beta += weight;
        }
    }

    /// Mixed observation — `α += w·p, β += w·(1−p)`. `p` clamped to `[0,1]`.
    pub fn observe_mixed(&mut self, dim: &str, p: f64, weight: f64) {
        let p = p.clamp(0.0, 1.0);
        if let Some(par) = self.params.get_mut(dim) {
            par.alpha += weight * p;
            par.beta += weight * (1.0 - p);
        }
    }

    pub fn get(&self, dim: &str) -> BetaParam {
        self.params.get(dim).copied().unwrap_or_default()
    }

    /// Mean of the dimension. Returns `0.5` for an unknown dimension.
    pub fn mean(&self, dim: &str) -> f64 {
        self.params.get(dim).map(|p| p.mean()).unwrap_or(0.5)
    }

    /// Standard deviation of the dimension. Returns `0.0` if unknown.
    pub fn stddev(&self, dim: &str) -> f64 {
        self.params.get(dim).map(|p| p.stddev()).unwrap_or(0.0)
    }

    /// Equal-weighted mean of all dimension means. `0.5` when empty.
    pub fn overall_competence(&self) -> f64 {
        if self.params.is_empty() {
            return 0.5;
        }
        let s: f64 = self.params.values().map(|p| p.mean()).sum();
        s / self.params.len() as f64
    }

    /// Equal-weighted mean of all dimension stddevs. `0.0` when empty.
    pub fn overall_uncertainty(&self) -> f64 {
        if self.params.is_empty() {
            return 0.0;
        }
        let s: f64 = self.params.values().map(|p| p.stddev()).sum();
        s / self.params.len() as f64
    }
}

// ── Numerical helpers ────────────────────────────────────────────────────
//
// We implement digamma + lgamma directly rather than pulling in a math crate.
// These are standard textbook implementations; accuracy is more than enough
// for an information-gain weight that gets multiplied by a hand-tuned γ.

/// Lanczos approximation to `lnΓ(x)`. Accurate to ~14 digits for `x > 0`.
pub(crate) fn lgamma(x: f64) -> f64 {
    // Use the well-known Lanczos coefficients (g = 7, n = 9).
    const G: f64 = 7.0;
    const COEF: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_13,
        -176.615_029_162_140_59,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_571_6e-6,
        1.505_632_735_149_311_6e-7,
    ];

    if x < 0.5 {
        // Reflection: lnΓ(x) + lnΓ(1−x) = lnπ − ln sin(πx)
        std::f64::consts::PI.ln() - (std::f64::consts::PI * x).sin().abs().ln() - lgamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = COEF[0];
        for (i, c) in COEF.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        let t = x + G + 0.5;
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}

/// Digamma `ψ(x) = d/dx lnΓ(x)`. Uses recurrence to push `x ≥ 6`, then the
/// asymptotic series.
pub(crate) fn digamma(mut x: f64) -> f64 {
    if x <= 0.0 {
        // Bail to zero — Beta params should never reach here.
        return 0.0;
    }
    let mut result = 0.0;
    while x < 6.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    // Asymptotic series for large x:
    //   ψ(x) ≈ ln(x) − 1/(2x) − 1/(12x²) + 1/(120x⁴) − 1/(252x⁶)
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result
        + x.ln()
        - 0.5 * inv
        - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 / 252.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jeffreys_prior_mean_is_half() {
        let bp = BetaParam::default();
        assert!((bp.mean() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn observe_success_raises_mean() {
        let mut sb = SelfBelief::new();
        let m0 = sb.mean("tool_use");
        for _ in 0..10 {
            sb.observe_success("tool_use", 1.0);
        }
        assert!(sb.mean("tool_use") > m0);
    }

    #[test]
    fn observe_failure_lowers_mean() {
        let mut sb = SelfBelief::new();
        let m0 = sb.mean("tool_use");
        for _ in 0..10 {
            sb.observe_failure("tool_use", 1.0);
        }
        assert!(sb.mean("tool_use") < m0);
    }

    #[test]
    fn mixed_observation_weights_correctly() {
        let mut sb = SelfBelief::new();
        sb.observe_mixed("reasoning", 0.7, 1.0);
        let bp = sb.get("reasoning");
        assert!((bp.alpha - 1.7).abs() < 1e-9);
        assert!((bp.beta - 1.3).abs() < 1e-9);
    }

    #[test]
    fn entropy_is_finite_for_jeffreys() {
        let bp = BetaParam::default();
        let h = bp.entropy();
        assert!(h.is_finite());
    }

    #[test]
    fn info_gain_is_nonneg_after_many_obs() {
        // After tens of observations the prior is concentrated and the
        // expected info-gain from one extra observation should be small but
        // non-negative.
        let mut bp = BetaParam::default();
        for _ in 0..50 {
            bp.alpha += 1.0;
        }
        assert!(bp.expected_info_gain_bernoulli() >= 0.0);
    }

    #[test]
    fn overall_uncertainty_drops_with_evidence() {
        let mut sb = SelfBelief::new();
        let u0 = sb.overall_uncertainty();
        for _ in 0..100 {
            sb.observe_success("tool_use", 1.0);
            sb.observe_success("code_edit", 1.0);
        }
        assert!(sb.overall_uncertainty() < u0);
    }
}
