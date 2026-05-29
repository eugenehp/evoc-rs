//! MNIST training set (IDX). See [`crate::fashion_mnist_data`] for Fashion-MNIST.

pub use crate::idx_digits::{
    IdxDigitsError, DIGITS_DIM as MNIST_DIM, DIGITS_TRAIN_LEN as MNIST_TRAIN_LEN,
};

use crate::idx_digits::IdxDigitsDataset;
use ndarray::{Array1, Array2};
use std::path::{Path, PathBuf};

pub type MnistError = IdxDigitsError;

pub fn cache_dir() -> PathBuf {
    IdxDigitsDataset::Mnist.cache_dir()
}

pub fn ensure_train_cached(cache: &Path) -> Result<(), MnistError> {
    IdxDigitsDataset::Mnist.ensure_train_cached(cache)
}

pub fn load_train(cache: &Path) -> Result<(Vec<u8>, Vec<u8>), MnistError> {
    IdxDigitsDataset::Mnist.load_train(cache)
}

pub fn sample_normalized(
    n: usize,
    seed: u64,
    cache: &Path,
) -> Result<(Array2<f32>, Array1<u8>, Vec<u8>), MnistError> {
    IdxDigitsDataset::Mnist.sample_normalized(n, seed, cache)
}
