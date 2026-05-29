//! Fashion-MNIST training set (IDX, same layout as MNIST).

pub use crate::idx_digits::{
    IdxDigitsError, DIGITS_DIM as FASHION_DIM, DIGITS_TRAIN_LEN as FASHION_TRAIN_LEN,
};

use crate::idx_digits::IdxDigitsDataset;
use ndarray::{Array1, Array2};
use std::path::{Path, PathBuf};

pub type FashionMnistError = IdxDigitsError;

pub fn cache_dir() -> PathBuf {
    IdxDigitsDataset::FashionMnist.cache_dir()
}

pub fn ensure_train_cached(cache: &Path) -> Result<(), FashionMnistError> {
    IdxDigitsDataset::FashionMnist.ensure_train_cached(cache)
}

pub fn load_train(cache: &Path) -> Result<(Vec<u8>, Vec<u8>), FashionMnistError> {
    IdxDigitsDataset::FashionMnist.load_train(cache)
}

pub fn sample_normalized(
    n: usize,
    seed: u64,
    cache: &Path,
) -> Result<(Array2<f32>, Array1<u8>, Vec<u8>), FashionMnistError> {
    IdxDigitsDataset::FashionMnist.sample_normalized(n, seed, cache)
}

pub fn class_name(label: u8) -> &'static str {
    IdxDigitsDataset::FashionMnist.class_name(label)
}
