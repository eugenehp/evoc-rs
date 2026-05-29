//! Download and load MNIST / Fashion-MNIST training sets (IDX format).

use crate::dataset_util::l2_normalize_rows;
use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;

pub const DIGITS_DIM: usize = 784;
pub const DIGITS_TRAIN_LEN: usize = 60_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IdxDigitsDataset {
    Mnist,
    FashionMnist,
}

impl IdxDigitsDataset {
    pub fn name(self) -> &'static str {
        match self {
            Self::Mnist => "MNIST",
            Self::FashionMnist => "Fashion-MNIST",
        }
    }

    pub fn cache_dir(self) -> PathBuf {
        let sub = match self {
            Self::Mnist => {
                if let Ok(dir) = std::env::var("EVOC_MNIST_DIR") {
                    return PathBuf::from(dir);
                }
                "mnist"
            }
            Self::FashionMnist => {
                if let Ok(dir) = std::env::var("EVOC_FASHION_MNIST_DIR") {
                    return PathBuf::from(dir);
                }
                "fashion-mnist"
            }
        };
        if let Some(home) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) {
            return PathBuf::from(home).join(".cache/evoc").join(sub);
        }
        std::env::temp_dir().join("evoc").join(sub)
    }

    fn train_images_url(self) -> &'static str {
        match self {
            Self::Mnist => {
                "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz"
            }
            Self::FashionMnist => {
                "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz"
            }
        }
    }

    fn train_labels_url(self) -> &'static str {
        match self {
            Self::Mnist => {
                "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz"
            }
            Self::FashionMnist => {
                "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz"
            }
        }
    }

    pub fn class_name(self, label: u8) -> &'static str {
        match self {
            Self::Mnist => MNIST_CLASS_NAMES
                .get(label as usize)
                .copied()
                .unwrap_or("?"),
            Self::FashionMnist => FASHION_CLASS_NAMES
                .get(label as usize)
                .copied()
                .unwrap_or("?"),
        }
    }

    pub fn ensure_train_cached(self, cache: &Path) -> Result<(), IdxDigitsError> {
        fs::create_dir_all(cache).map_err(|source| IdxDigitsError::Io {
            path: cache.to_path_buf(),
            source,
        })?;
        let images_path = cache.join("train-images-idx3-ubyte");
        let labels_path = cache.join("train-labels-idx1-ubyte");
        if !images_path.is_file() {
            download_gz(self.train_images_url(), &images_path)?;
        }
        if !labels_path.is_file() {
            download_gz(self.train_labels_url(), &labels_path)?;
        }
        Ok(())
    }

    pub fn load_train(self, cache: &Path) -> Result<(Vec<u8>, Vec<u8>), IdxDigitsError> {
        self.ensure_train_cached(cache)?;
        let pixels = read_train_images(&cache.join("train-images-idx3-ubyte"))?;
        let labels = read_train_labels(&cache.join("train-labels-idx1-ubyte"))?;
        if pixels.len() / DIGITS_DIM != labels.len() {
            return Err(IdxDigitsError::InvalidIdx {
                path: cache.join("train-images-idx3-ubyte"),
                message: "image and label counts differ".into(),
            });
        }
        if labels.len() != DIGITS_TRAIN_LEN {
            return Err(IdxDigitsError::UnexpectedLength {
                name: self.name(),
                got: labels.len(),
            });
        }
        Ok((pixels, labels))
    }

    pub fn sample_normalized(
        self,
        n: usize,
        seed: u64,
        cache: &Path,
    ) -> Result<(Array2<f32>, Array1<u8>, Vec<u8>), IdxDigitsError> {
        let (pixels, labels) = self.load_train(cache)?;
        let n_total = labels.len();
        let mut rng = StdRng::seed_from_u64(seed);
        let indices: Vec<usize> = if n <= n_total {
            let mut idx: Vec<usize> = (0..n_total).collect();
            idx.shuffle(&mut rng);
            idx.truncate(n);
            idx
        } else {
            (0..n).map(|_| rng.gen_range(0..n_total)).collect()
        };

        let mut data = Array2::<f32>::zeros((indices.len(), DIGITS_DIM));
        let mut class_labels = Array1::<u8>::zeros(indices.len());
        let mut sample_pixels = vec![0u8; indices.len() * DIGITS_DIM];
        for (row, &idx) in indices.iter().enumerate() {
            class_labels[row] = labels[idx];
            for col in 0..DIGITS_DIM {
                let p = pixels[idx * DIGITS_DIM + col];
                sample_pixels[row * DIGITS_DIM + col] = p;
                data[[row, col]] = p as f32;
            }
        }
        l2_normalize_rows(&mut data);
        Ok((data, class_labels, sample_pixels))
    }
}

const MNIST_CLASS_NAMES: [&str; 10] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];

const FASHION_CLASS_NAMES: [&str; 10] = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
];

#[derive(Debug, Error)]
pub enum IdxDigitsError {
    #[error("HTTP download failed for {url}: {source}")]
    Download { url: String, source: ureq::Error },
    #[error("I/O error at {path}: {source}")]
    Io { path: PathBuf, source: io::Error },
    #[error("invalid IDX file {path}: {message}")]
    InvalidIdx { path: PathBuf, message: String },
    #[error("{name} train set has {got} samples, expected {DIGITS_TRAIN_LEN}")]
    UnexpectedLength { name: &'static str, got: usize },
}

fn download_gz(url: &str, dest: &Path) -> Result<(), IdxDigitsError> {
    eprintln!("downloading {url} -> {}", dest.display());
    let response = ureq::get(url)
        .call()
        .map_err(|source| IdxDigitsError::Download {
            url: url.to_string(),
            source,
        })?;
    let mut reader = response.into_reader();
    let mut gz_bytes = Vec::new();
    reader
        .read_to_end(&mut gz_bytes)
        .map_err(|source| IdxDigitsError::Io {
            path: dest.to_path_buf(),
            source,
        })?;
    let mut decoder = flate2::read::GzDecoder::new(gz_bytes.as_slice());
    let mut raw = Vec::new();
    decoder
        .read_to_end(&mut raw)
        .map_err(|source| IdxDigitsError::Io {
            path: dest.to_path_buf(),
            source,
        })?;
    let mut file = File::create(dest).map_err(|source| IdxDigitsError::Io {
        path: dest.to_path_buf(),
        source,
    })?;
    file.write_all(&raw).map_err(|source| IdxDigitsError::Io {
        path: dest.to_path_buf(),
        source,
    })?;
    Ok(())
}

fn read_train_images(path: &Path) -> Result<Vec<u8>, IdxDigitsError> {
    let buf = read_idx_bytes(path)?;
    parse_idx_images(&buf, path)
}

fn read_train_labels(path: &Path) -> Result<Vec<u8>, IdxDigitsError> {
    let buf = read_idx_bytes(path)?;
    parse_idx_labels(&buf, path)
}

fn read_idx_bytes(path: &Path) -> Result<Vec<u8>, IdxDigitsError> {
    let mut file = File::open(path).map_err(|source| IdxDigitsError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)
        .map_err(|source| IdxDigitsError::Io {
            path: path.to_path_buf(),
            source,
        })?;
    if buf.len() >= 2 && buf[0] == 0x1f && buf[1] == 0x8b {
        let mut decoder = flate2::read::GzDecoder::new(buf.as_slice());
        let mut raw = Vec::new();
        decoder
            .read_to_end(&mut raw)
            .map_err(|source| IdxDigitsError::Io {
                path: path.to_path_buf(),
                source,
            })?;
        return Ok(raw);
    }
    Ok(buf)
}

fn parse_idx_images(bytes: &[u8], path: &Path) -> Result<Vec<u8>, IdxDigitsError> {
    if bytes.len() < 16 {
        return Err(IdxDigitsError::InvalidIdx {
            path: path.to_path_buf(),
            message: "file too short for IDX header".into(),
        });
    }
    let magic = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if magic != 2051 {
        return Err(IdxDigitsError::InvalidIdx {
            path: path.to_path_buf(),
            message: format!("expected magic 2051, got {magic}"),
        });
    }
    let count = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    let rows = u32::from_be_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
    let cols = u32::from_be_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize;
    if rows != 28 || cols != 28 {
        return Err(IdxDigitsError::InvalidIdx {
            path: path.to_path_buf(),
            message: format!("expected 28x28 images, got {rows}x{cols}"),
        });
    }
    let expected = 16 + count * DIGITS_DIM;
    if bytes.len() != expected {
        return Err(IdxDigitsError::InvalidIdx {
            path: path.to_path_buf(),
            message: format!("expected {expected} bytes, got {}", bytes.len()),
        });
    }
    Ok(bytes[16..].to_vec())
}

fn parse_idx_labels(bytes: &[u8], path: &Path) -> Result<Vec<u8>, IdxDigitsError> {
    if bytes.len() < 8 {
        return Err(IdxDigitsError::InvalidIdx {
            path: path.to_path_buf(),
            message: "file too short for IDX header".into(),
        });
    }
    let magic = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if magic != 2049 {
        return Err(IdxDigitsError::InvalidIdx {
            path: path.to_path_buf(),
            message: format!("expected magic 2049, got {magic}"),
        });
    }
    let count = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    let expected = 8 + count;
    if bytes.len() != expected {
        return Err(IdxDigitsError::InvalidIdx {
            path: path.to_path_buf(),
            message: format!("expected {expected} bytes, got {}", bytes.len()),
        });
    }
    Ok(bytes[8..].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_idx_headers() {
        let mut images = vec![0u8; 16 + 2 * DIGITS_DIM];
        images[0..4].copy_from_slice(&2051u32.to_be_bytes());
        images[4..8].copy_from_slice(&2u32.to_be_bytes());
        images[8..12].copy_from_slice(&28u32.to_be_bytes());
        images[12..16].copy_from_slice(&28u32.to_be_bytes());
        let parsed = parse_idx_images(&images, Path::new("test-images")).unwrap();
        assert_eq!(parsed.len(), 2 * DIGITS_DIM);

        let mut labels = vec![0u8; 10];
        labels[0..4].copy_from_slice(&2049u32.to_be_bytes());
        labels[4..8].copy_from_slice(&2u32.to_be_bytes());
        labels[8..10].copy_from_slice(&[3, 7]);
        let parsed = parse_idx_labels(&labels, Path::new("test-labels")).unwrap();
        assert_eq!(parsed, vec![3, 7]);
    }
}
