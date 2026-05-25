pub mod actions;
pub mod canonical;
pub mod compare;
pub mod diff;
pub mod env;
pub mod eval;
pub mod expr;
pub mod features;
pub mod gen;
pub mod gen_coverage;
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
        m.add_function(wrap_pyfunction!(py_coverage_stats, m)?)?;
        m.add_function(wrap_pyfunction!(py_generate_covered_batch, m)?)?;
        Ok(())
    }

    #[pyfunction]
    fn py_generate_covered_batch(
        total: usize,
        config: &crate::gen::PyGenConfig,
    ) -> PyResult<Vec<(PyExprTree, PyExprTree)>> {
        let pairs = crate::gen_coverage::generate_covered_pairs(total, &config.inner);
        Ok(pairs
            .into_iter()
            .map(|p| {
                (
                    PyExprTree { inner: p.integrand },
                    PyExprTree { inner: p.integral },
                )
            })
            .collect())
    }

    #[pyfunction]
    fn py_coverage_stats(total: usize) -> PyResult<String> {
        let stats = crate::gen_coverage::coverage_stats(total);
        Ok(format!(
            "{{\"total_skeletons\": {}, \"total_generated\": {}, \"per_family_target\": {}, \"skeleton_fraction\": {:.2}, \"random_fraction\": {:.2}}}",
            stats.total_skeletons, stats.total_generated,
            stats.per_family_target, stats.skeleton_fraction, stats.random_fraction,
        ))
    }
}
