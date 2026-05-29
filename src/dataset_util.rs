//! Shared helpers for built-in example datasets.

use ndarray::{Array2, Axis};

pub fn l2_normalize_rows(data: &mut Array2<f32>) {
    for mut row in data.axis_iter_mut(Axis(0)) {
        let norm = row.mapv(|x| x * x).sum().sqrt().max(1e-12);
        row.mapv_inplace(|x| x / norm);
    }
}
