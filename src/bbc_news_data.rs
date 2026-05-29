//! BBC News corpus (5 topics) — download, vectorize, cluster (no Python).
//!
//! Source: [UCD MLG BBC full-text](http://mlg.ucd.ie/datasets/bbc.html).
//! Cache: `EVOC_BBC_NEWS_DIR` or `~/.cache/evoc/bbc-news`.

use crate::dataset_util::l2_normalize_rows;
use crate::text_bow::{hash_bow_into_row, normalize_body};
use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;
use zip::ZipArchive;

const ARCHIVE_URL: &str = "http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip";
const ARCHIVE_NAME: &str = "bbc-fulltext.zip";
const CORPUS_DIR: &str = "bbc";
const EXTRACTED_MARKER: &str = ".extracted";

pub const BBC_TRAIN_LEN: usize = 2225;

pub use crate::text_bow::DEFAULT_FEATURE_DIM;

#[derive(Debug, Error)]
pub enum BbcNewsError {
    #[error("HTTP download failed for {url}: {source}")]
    Download { url: String, source: ureq::Error },
    #[error("I/O error at {path}: {source}")]
    Io { path: PathBuf, source: io::Error },
    #[error("zip error: {0}")]
    Zip(String),
    #[error("no documents found under {path}")]
    EmptyCorpus { path: PathBuf },
}

pub fn cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("EVOC_BBC_NEWS_DIR") {
        return PathBuf::from(dir);
    }
    if let Some(home) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) {
        return PathBuf::from(home).join(".cache/evoc/bbc-news");
    }
    std::env::temp_dir().join("evoc/bbc-news")
}

pub fn ensure_cached(cache: &Path) -> Result<PathBuf, BbcNewsError> {
    fs::create_dir_all(cache).map_err(|source| BbcNewsError::Io {
        path: cache.to_path_buf(),
        source,
    })?;
    let corpus_root = cache.join(CORPUS_DIR);
    let marker = cache.join(EXTRACTED_MARKER);
    if marker.is_file() && corpus_root.is_dir() {
        return Ok(corpus_root);
    }

    let archive_path = cache.join(ARCHIVE_NAME);
    if !archive_path.is_file() {
        download_file(ARCHIVE_URL, &archive_path)?;
    }

    if corpus_root.is_dir() {
        let _ = fs::remove_dir_all(&corpus_root);
    }

    let file = File::open(&archive_path).map_err(|source| BbcNewsError::Io {
        path: archive_path.clone(),
        source,
    })?;
    let mut archive = ZipArchive::new(file).map_err(|e| BbcNewsError::Zip(e.to_string()))?;
    archive
        .extract(cache)
        .map_err(|e| BbcNewsError::Zip(e.to_string()))?;

    if !corpus_root.is_dir() {
        return Err(BbcNewsError::Zip(format!(
            "expected {CORPUS_DIR}/ after extract"
        )));
    }

    fs::write(&marker, b"ok").map_err(|source| BbcNewsError::Io {
        path: marker,
        source,
    })?;
    Ok(corpus_root)
}

fn download_file(url: &str, dest: &Path) -> Result<(), BbcNewsError> {
    eprintln!("downloading {url} -> {}", dest.display());
    let response = ureq::get(url)
        .call()
        .map_err(|source| BbcNewsError::Download {
            url: url.to_string(),
            source,
        })?;
    let mut reader = response.into_reader();
    let mut bytes = Vec::new();
    reader
        .read_to_end(&mut bytes)
        .map_err(|source| BbcNewsError::Io {
            path: dest.to_path_buf(),
            source,
        })?;
    let mut file = File::create(dest).map_err(|source| BbcNewsError::Io {
        path: dest.to_path_buf(),
        source,
    })?;
    file.write_all(&bytes).map_err(|source| BbcNewsError::Io {
        path: dest.to_path_buf(),
        source,
    })?;
    Ok(())
}

/// Load all articles (`text`, topic label).
pub fn load_documents(cache: &Path) -> Result<Vec<(String, u8)>, BbcNewsError> {
    let corpus_root = ensure_cached(cache)?;
    let mut categories: Vec<String> = fs::read_dir(&corpus_root)
        .map_err(|source| BbcNewsError::Io {
            path: corpus_root.clone(),
            source,
        })?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    categories.sort();

    let mut label_of: HashMap<String, u8> = HashMap::new();
    for (i, name) in categories.iter().enumerate() {
        label_of.insert(name.clone(), i as u8);
    }

    let mut docs = Vec::new();
    for category in &categories {
        let dir = corpus_root.join(category);
        for entry in fs::read_dir(&dir).map_err(|source| BbcNewsError::Io {
            path: dir.clone(),
            source,
        })? {
            let entry = entry.map_err(|source| BbcNewsError::Io {
                path: dir.clone(),
                source,
            })?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let bytes = fs::read(&path).map_err(|source| BbcNewsError::Io {
                path: path.clone(),
                source,
            })?;
            let raw = String::from_utf8_lossy(&bytes).into_owned();
            let body = normalize_body(&raw);
            if body.is_empty() {
                continue;
            }
            docs.push((body, label_of[category]));
        }
    }

    if docs.is_empty() {
        return Err(BbcNewsError::EmptyCorpus { path: corpus_root });
    }
    Ok(docs)
}

pub fn category_name(label: u8) -> &'static str {
    BBC_CATEGORIES.get(label as usize).copied().unwrap_or("?")
}

pub fn sample_normalized(
    n: usize,
    seed: u64,
    cache: &Path,
    feature_dim: usize,
) -> Result<(Array2<f32>, Array1<u8>), BbcNewsError> {
    let docs = load_documents(cache)?;
    let n_total = docs.len();
    let mut rng = StdRng::seed_from_u64(seed);
    let indices: Vec<usize> = if n <= n_total {
        let mut idx: Vec<usize> = (0..n_total).collect();
        idx.shuffle(&mut rng);
        idx.truncate(n);
        idx
    } else {
        (0..n).map(|_| rng.gen_range(0..n_total)).collect()
    };

    let mut data = Array2::<f32>::zeros((indices.len(), feature_dim));
    let mut labels = Array1::<u8>::zeros(indices.len());
    for (row, &doc_idx) in indices.iter().enumerate() {
        let (ref text, label) = docs[doc_idx];
        labels[row] = label;
        hash_bow_into_row(text, data.row_mut(row));
    }
    l2_normalize_rows(&mut data);
    Ok((data, labels))
}

/// Sorted folder names → label index (alphabetical).
pub const BBC_CATEGORIES: [&str; 5] = ["business", "entertainment", "politics", "sport", "tech"];
