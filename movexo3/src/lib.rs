use std::vec::Vec;
use std::time::Instant;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use rayon::prelude::*;

#[derive(FromPyObject)]
struct MotionVectior {

    #[pyo3(item(3))]
    src_x: f32,

    #[pyo3(item(4))]
    src_y: f32,

    #[pyo3(item(5))]
    dst_x: f32,
 
    #[pyo3(item(6))]
    dst_y: f32,

    #[pyo3(item(8))]
    motion_x: f32,

    #[pyo3(item(9))]
    motion_y: f32,

    #[pyo3(item(10))]
    motion_scale: f32
}

impl MotionVectior {
    fn to_array(&self) -> [f32; 7] {
        [
            self.src_x, 
            self.src_y, 
            self.dst_x, 
            self.dst_y, 
            self.motion_x, 
            self.motion_y, 
            self.motion_scale
        ]
    }
}


#[pyfunction]
fn mvs_to_list(mvs: &PyAny) -> PyResult<Vec<[f32; 7]>> {
    let mut list_of_mvs: Vec<[f32; 7]> = Vec::new();
    let num_of_mvs = mvs.len()?;
    for i in 0..num_of_mvs { 
        let tup = mvs.get_item(i)?;
        let mv: MotionVectior = tup.extract()?;
        list_of_mvs.push(mv.to_array()); 

    }

    Ok(list_of_mvs)
}

/// A Python module implemented in Rust.
#[pymodule]
fn movexo3(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mvs_to_list, m)?)?;
    Ok(())
}
