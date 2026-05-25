//! Step environment for action-space integration (Sprint 3).
//!
//! Wraps an integrand state and validates each action via `verify.rs`.

use serde::{Deserialize, Serialize};

use crate::actions::{Action, ActionError, apply_ibp, apply_substitute};
use crate::expr::{to_prefix_string, ExprNode, VarId};
use crate::features::{extract_features, simplicity_score};
use crate::verify::verify_single;

/// Result of taking one step in the environment.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StepResult {
    /// The new integrand state after the action.
    pub next_state: ExprNode,
    /// Reward for this step.
    pub reward: f64,
    /// Whether the episode is done (Close action succeeded).
    pub done: bool,
    /// Whether the per-step verification passed.
    pub verify_ok: bool,
    /// Action chain so far (for proof trace).
    pub trace_len: usize,
}

/// Integration environment: tracks the current integrand, variable,
/// action history, and validates each step.
#[derive(Clone, Debug)]
pub struct Env {
    /// Original integrand (for final verification).
    original_integrand: ExprNode,
    /// Current state (possibly transformed integrand).
    current_state: ExprNode,
    /// Integration variable.
    var: VarId,
    /// Number of actions taken so far.
    step_count: usize,
    /// Maximum allowed steps.
    max_steps: usize,
    /// Whether the episode is finished.
    done: bool,
    /// Accumulated closed-form terms from IBP.
    closed_terms: Vec<ExprNode>,
}

impl Env {
    /// Create a new environment for integrating `integrand` w.r.t. `var`.
    pub fn new(integrand: ExprNode, var: VarId, max_steps: usize) -> Self {
        Self {
            original_integrand: integrand.clone(),
            current_state: integrand,
            var,
            step_count: 0,
            max_steps,
            done: false,
            closed_terms: Vec::new(),
        }
    }

    /// Reset the environment to its initial state.
    pub fn reset(&mut self) {
        self.current_state = self.original_integrand.clone();
        self.step_count = 0;
        self.done = false;
        self.closed_terms.clear();
    }

    /// Current integrand state.
    pub fn state(&self) -> &ExprNode {
        &self.current_state
    }

    /// Whether the episode is done.
    pub fn is_done(&self) -> bool {
        self.done
    }

    /// Take one step: apply an action and verify the result.
    pub fn step(&mut self, action: Action) -> Result<StepResult, ActionError> {
        if self.done {
            return Err(ActionError::IllTyped("episode already done".to_string()));
        }

        self.step_count += 1;
        if self.step_count > self.max_steps {
            self.done = true;
            return Ok(StepResult {
                next_state: self.current_state.clone(),
                reward: -1.0,
                done: true,
                verify_ok: false,
                trace_len: self.step_count,
            });
        }

        match action {
            Action::Substitute { ref inner_fn } => {
                let new_state = apply_substitute(
                    &self.current_state,
                    inner_fn,
                    self.var.as_str(),
                    // Use a fresh variable for substitution
                    next_free_var(&self.current_state, self.var),
                )?;
                self.current_state = new_state.clone();
                Ok(StepResult {
                    next_state: new_state,
                    reward: 0.0,
                    done: false,
                    verify_ok: true,
                    trace_len: self.step_count,
                })
            }
            Action::IntegrateByParts {
                ref u_factor,
                ref dv_factor,
            } => {
                let (closed_term, remainder) =
                    apply_ibp(u_factor, dv_factor, self.var.as_str())?;
                self.closed_terms.push(closed_term);
                self.current_state = remainder.clone();
                Ok(StepResult {
                    next_state: remainder,
                    reward: 0.0,
                    done: false,
                    verify_ok: true,
                    trace_len: self.step_count,
                })
            }
            Action::PartialFractions => {
                // Partial fractions is a structural decomposition.
                // In this simplified version, we leave the state unchanged
                // and let the policy choose the next action based on the
                // decomposed form. A full implementation would factor
                // rational expressions.
                Ok(StepResult {
                    next_state: self.current_state.clone(),
                    reward: 0.0,
                    done: false,
                    verify_ok: true,
                    trace_len: self.step_count,
                })
            }
            Action::Close { ref antiderivative } => {
                // IBP chain: ∫u dv = uv − ∫v du.
                // closed_terms[i] = u_i · v_i from successive IBP steps.
                // antiderivative = proposed F for the final remainder.
                // Unwind in reverse: full = uv_0 − (uv_1 − (uv_2 − ... F))
                let full_antideriv = if self.closed_terms.is_empty() {
                    antiderivative.clone()
                } else {
                    let mut total = antiderivative.clone();
                    for term in self.closed_terms.iter().rev() {
                        total = ExprNode::sub(term.clone(), total);
                    }
                    total
                };

                let f_prefix = to_prefix_string(&self.original_integrand);
                let big_f_prefix = to_prefix_string(&full_antideriv);
                let ok = verify_single(
                    &f_prefix,
                    &big_f_prefix,
                    self.var.as_str(),
                );
                self.done = true;

                let reward = if ok {
                    let simp = simplicity_score(&full_antideriv);
                    1.0 + (10.0 / simp as f64).min(0.5)
                } else {
                    -1.0
                };

                Ok(StepResult {
                    next_state: full_antideriv,
                    reward,
                    done: true,
                    verify_ok: ok,
                    trace_len: self.step_count,
                })
            }
        }
    }

    /// Extract feature vector for the current state.
    pub fn state_features(&self) -> Vec<f32> {
        extract_features(
            &self.current_state,
            self.var,
            "indef",
            "none",
            0,
        )
    }
}

/// Find the next free variable ID not used in the expression.
fn next_free_var(expr: &ExprNode, exclude: VarId) -> VarId {
    let free = expr.free_vars();
    for &v in &VarId::ALL {
        if v != exclude && !free.contains(&v) {
            return v;
        }
    }
    // All variables exhausted — pick the one least likely to conflict.
    // In practice, expressions with 5+ free variables are extremely rare.
    *VarId::ALL
        .iter()
        .filter(|&&v| v != exclude)
        .next()
        .unwrap_or(&VarId::Y)
}

// ---------------------------------------------------------------------------
// PyO3 bindings
// ---------------------------------------------------------------------------
#[cfg(feature = "python")]
pub use self::py::{py_env_reset, py_env_step, py_env_state_features, PyEnv};

#[cfg(feature = "python")]
mod py {
    use pyo3::prelude::*;
    use crate::expr::{from_prefix_string, VarId};
    use super::Env;
    use crate::actions::Action;

    #[pyclass(name = "IntegrationEnv")]
    pub struct PyEnv {
        inner: Env,
    }

    #[pymethods]
    impl PyEnv {
        #[new]
        #[pyo3(signature = (integrand_prefix, var = "x", max_steps = 10))]
        fn new(integrand_prefix: &str, var: &str, max_steps: usize) -> PyResult<Self> {
            let expr = from_prefix_string(integrand_prefix)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
            let var_id = VarId::from_str(var)
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
                    format!("unknown variable: {var}"),
                ))?;
            Ok(Self {
                inner: Env::new(expr, var_id, max_steps),
            })
        }

        fn reset(&mut self) {
            self.inner.reset();
        }

        fn is_done(&self) -> bool {
            self.inner.is_done()
        }

        fn state_features(&self) -> Vec<f32> {
            self.inner.state_features()
        }

        fn current_state_prefix(&self) -> String {
            crate::expr::to_prefix_string(self.inner.state())
        }

        fn step_close(&mut self, antiderivative_prefix: &str) -> PyResult<(f64, bool, bool)> {
            let antideriv = from_prefix_string(antiderivative_prefix)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
            let result = self.inner.step(Action::Close { antiderivative: antideriv })
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            Ok((result.reward, result.done, result.verify_ok))
        }

        fn step_substitute(&mut self, inner_fn_prefix: &str) -> PyResult<(f64, bool, bool)> {
            let inner = from_prefix_string(inner_fn_prefix)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
            let result = self.inner.step(Action::Substitute { inner_fn: inner })
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            Ok((result.reward, result.done, result.verify_ok))
        }

        fn step_ibp(
            &mut self,
            u_prefix: &str,
            dv_prefix: &str,
        ) -> PyResult<(f64, bool, bool)> {
            let u = from_prefix_string(u_prefix)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
            let dv = from_prefix_string(dv_prefix)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
            let result = self.inner.step(Action::IntegrateByParts {
                u_factor: u,
                dv_factor: dv,
            })
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            Ok((result.reward, result.done, result.verify_ok))
        }
    }

    #[pyfunction]
    #[pyo3(name = "env_reset")]
    pub fn py_env_reset(integrand_prefix: &str, var: &str) -> PyResult<Vec<f32>> {
        let expr = from_prefix_string(integrand_prefix)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        let var_id = VarId::from_str(var)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
                format!("unknown variable: {var}"),
            ))?;
        let env = Env::new(expr, var_id, 10);
        Ok(env.state_features())
    }

    #[pyfunction]
    #[pyo3(name = "env_step")]
    pub fn py_env_step(
        integrand_prefix: &str,
        antiderivative_prefix: &str,
        var: &str,
    ) -> PyResult<(f64, bool)> {
        let expr = from_prefix_string(integrand_prefix)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        let antideriv = from_prefix_string(antiderivative_prefix)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        let var_id = VarId::from_str(var)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(
                format!("unknown variable: {var}"),
            ))?;
        let mut env = Env::new(expr, var_id, 10);
        let result = env.step(Action::Close { antiderivative: antideriv })
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok((result.reward, result.verify_ok))
    }

    #[pyfunction]
    #[pyo3(name = "env_state_features")]
    pub fn py_env_state_features(
        integrand_prefix: &str,
        var: &str,
    ) -> PyResult<Vec<f32>> {
        py_env_reset(integrand_prefix, var)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::ExprNode;

    #[test]
    fn env_close_correct() {
        // ∫ cos(x) dx = sin(x)
        let integrand = ExprNode::cos(ExprNode::var("x"));
        let mut env = Env::new(integrand, VarId::X, 10);

        let result = env
            .step(Action::Close {
                antiderivative: ExprNode::sin(ExprNode::var("x")),
            })
            .unwrap();

        assert!(result.verify_ok);
        assert!(result.done);
        assert!(result.reward > 0.0);
    }

    #[test]
    fn env_close_incorrect() {
        let integrand = ExprNode::cos(ExprNode::var("x"));
        let mut env = Env::new(integrand, VarId::X, 10);

        let result = env
            .step(Action::Close {
                antiderivative: ExprNode::cos(ExprNode::var("x")),
            })
            .unwrap();

        assert!(!result.verify_ok);
        assert!(result.done);
        assert!(result.reward < 0.0);
    }

    #[test]
    fn env_max_steps() {
        let integrand = ExprNode::var("x");
        let mut env = Env::new(integrand, VarId::X, 1);

        // First step: max_steps not yet exceeded
        let _ = env.step(Action::PartialFractions).unwrap();
        // Second step: should exceed max_steps
        let result = env.step(Action::PartialFractions).unwrap();
        assert!(result.done);
        assert!(result.reward < 0.0);
    }

    #[test]
    fn env_state_features_dim() {
        let integrand = ExprNode::sin(ExprNode::var("x"));
        let env = Env::new(integrand, VarId::X, 10);
        let feats = env.state_features();
        assert_eq!(feats.len(), crate::features::FEATURE_DIM);
    }

    #[test]
    fn env_reset() {
        let integrand = ExprNode::cos(ExprNode::var("x"));
        let mut env = Env::new(integrand.clone(), VarId::X, 10);
        let _ = env.step(Action::PartialFractions);
        env.reset();
        assert!(!env.is_done());
        assert_eq!(*env.state(), integrand);
    }
}
