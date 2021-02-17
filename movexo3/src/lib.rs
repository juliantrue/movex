use std::vec::Vec;
use pyo3::prelude::*;
use pyo3::types::{PyList};
use pyo3::wrap_pyfunction;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn mvs_to_list(mvs: &PyList) -> PyResult<()> {

    for entry in mvs { 
        println!("{:?}", entry);
        //mv_list.push()
    }

    Ok(())
}

/// A Python module implemented in Rust.
#[pymodule]
fn movexo3(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mvs_to_list, m)?)?;
    Ok(())
}
