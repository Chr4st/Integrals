pub mod actions;
pub mod canonical;
pub mod compare;
pub mod diff;
pub mod env;
pub mod eval;
pub mod expr;
pub mod features;
pub mod gen;
pub mod skeleton;
pub mod verify;

#[cfg(feature = "python")]
mod python {
    use pyo3::prelude::*;

    use crate::expr::PyExprTree;

    #[pymodule]
    pub fn neurips_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PyExprTree>()?;
        m.add_class::<crate::gen::PyGenConfig>()?;
        m.add_class::<crate::env::PyEnv>()?;
        m.add_function(wrap_pyfunction!(crate::gen::py_generate_batch, m)?)?;
        m.add_function(wrap_pyfunction!(crate::diff::py_batch_differentiate, m)?)?;
        m.add_function(wrap_pyfunction!(crate::verify::py_verify_batch, m)?)?;
        m.add_function(wrap_pyfunction!(crate::verify::py_verify_batch_rich, m)?)?;
        m.add_function(wrap_pyfunction!(crate::features::py_extract_features, m)?)?;
        m.add_function(wrap_pyfunction!(crate::features::py_simplicity_score, m)?)?;
        m.add_function(wrap_pyfunction!(crate::compare::py_compare_equivalence_class, m)?)?;
        m.add_function(wrap_pyfunction!(crate::env::py_env_reset, m)?)?;
        m.add_function(wrap_pyfunction!(crate::env::py_env_step, m)?)?;
        m.add_function(wrap_pyfunction!(crate::env::py_env_state_features, m)?)?;
        Ok(())
    }
}
