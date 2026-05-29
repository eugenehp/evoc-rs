//! Download and vectorize the 20 Newsgroups training split (classic text clustering benchmark).
//!
//! Uses the sklearn/figshare archive, strips headers/footers, and builds hashed bag-of-words
//! features (standalone — no Python). Cache: `EVOC_NEWS20_DIR` or `~/.cache/evoc/news20`.

use crate::dataset_util::l2_normalize_rows;
use crate::text_bow::{hash_bow_into_row, strip_newsgroup_text};
use flate2::read::GzDecoder;
use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use tar::Archive;
use thiserror::Error;

/// Original 20 Newsgroups by-date archive (train + test splits).
const TRAIN_ARCHIVE_URL: &str = "http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz";
const TRAIN_ARCHIVE_NAME: &str = "20news-bydate.tar.gz";
const TRAIN_DIR_NAME: &str = "20news-bydate-train";
const EXTRACTED_MARKER: &str = ".extracted";

pub use crate::text_bow::DEFAULT_FEATURE_DIM;

#[derive(Debug, Error)]
pub enum News20Error {
    #[error("HTTP download failed for {url}: {source}")]
    Download { url: String, source: ureq::Error },
    #[error("I/O error at {path}: {source}")]
    Io { path: PathBuf, source: io::Error },
    #[error("archive error: {0}")]
    Archive(String),
    #[error("no training documents found under {path}")]
    EmptyCorpus { path: PathBuf },
}

/// Cache root (`EVOC_NEWS20_DIR` or `~/.cache/evoc/news20`).
pub fn cache_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("EVOC_NEWS20_DIR") {
        return PathBuf::from(dir);
    }
    if let Some(home) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) {
        return PathBuf::from(home).join(".cache/evoc/news20");
    }
    std::env::temp_dir().join("evoc/news20")
}

pub fn ensure_train_cached(cache: &Path) -> Result<PathBuf, News20Error> {
    fs::create_dir_all(cache).map_err(|source| News20Error::Io {
        path: cache.to_path_buf(),
        source,
    })?;
    let train_root = cache.join(TRAIN_DIR_NAME);
    let marker = cache.join(EXTRACTED_MARKER);
    if marker.is_file() && train_root.is_dir() {
        return Ok(train_root);
    }

    let archive_path = cache.join(TRAIN_ARCHIVE_NAME);
    if !archive_path.is_file() {
        download_file(TRAIN_ARCHIVE_URL, &archive_path)?;
    }

    if train_root.is_dir() {
        let _ = fs::remove_dir_all(&train_root);
    }

    let file = File::open(&archive_path).map_err(|source| News20Error::Io {
        path: archive_path.clone(),
        source,
    })?;
    let decoder = GzDecoder::new(file);
    let mut archive = Archive::new(decoder);
    archive
        .unpack(cache)
        .map_err(|e| News20Error::Archive(e.to_string()))?;

    if !train_root.is_dir() {
        return Err(News20Error::Archive(format!(
            "expected {TRAIN_DIR_NAME} after extract"
        )));
    }

    fs::write(&marker, b"ok").map_err(|source| News20Error::Io {
        path: marker,
        source,
    })?;
    Ok(train_root)
}

fn download_file(url: &str, dest: &Path) -> Result<(), News20Error> {
    eprintln!("downloading {url} -> {}", dest.display());
    let response = ureq::get(url)
        .call()
        .map_err(|source| News20Error::Download {
            url: url.to_string(),
            source,
        })?;
    let mut reader = response.into_reader();
    let mut bytes = Vec::new();
    reader
        .read_to_end(&mut bytes)
        .map_err(|source| News20Error::Io {
            path: dest.to_path_buf(),
            source,
        })?;
    let mut file = File::create(dest).map_err(|source| News20Error::Io {
        path: dest.to_path_buf(),
        source,
    })?;
    file.write_all(&bytes).map_err(|source| News20Error::Io {
        path: dest.to_path_buf(),
        source,
    })?;
    Ok(())
}

/// Load all training documents (category folder name + stripped body).
pub fn load_train_documents(cache: &Path) -> Result<Vec<(String, u8)>, News20Error> {
    let train_root = ensure_train_cached(cache)?;
    let mut categories: Vec<String> = fs::read_dir(&train_root)
        .map_err(|source| News20Error::Io {
            path: train_root.clone(),
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
        let dir = train_root.join(category);
        for entry in fs::read_dir(&dir).map_err(|source| News20Error::Io {
            path: dir.clone(),
            source,
        })? {
            let entry = entry.map_err(|source| News20Error::Io {
                path: dir.clone(),
                source,
            })?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let bytes = fs::read(&path).map_err(|source| News20Error::Io {
                path: path.clone(),
                source,
            })?;
            let raw = String::from_utf8_lossy(&bytes).into_owned();
            let body = strip_newsgroup_text(&raw);
            if body.is_empty() {
                continue;
            }
            docs.push((body, label_of[category]));
        }
    }

    if docs.is_empty() {
        return Err(News20Error::EmptyCorpus { path: train_root });
    }
    Ok(docs)
}

pub fn category_name(label: u8) -> &'static str {
    NEWS20_CATEGORIES
        .get(label as usize)
        .copied()
        .unwrap_or("?")
}

/// Subsample documents, hash bag-of-words → dense features, L2-normalize rows.
pub fn sample_normalized(
    n: usize,
    seed: u64,
    cache: &Path,
    feature_dim: usize,
) -> Result<(Array2<f32>, Array1<u8>), News20Error> {
    let docs = load_train_documents(cache)?;
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

/// Sorted newsgroup names (label index = position in this list).
pub const NEWS20_CATEGORIES: [&str; 20] = [
    "alt.atheism",
    "comp.graphics",
    "comp.os.ms-windows.misc",
    "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware",
    "comp.windows.x",
    "misc.forsale",
    "rec.autos",
    "rec.motorcycles",
    "rec.sport.baseball",
    "rec.sport.hockey",
    "sci.crypt",
    "sci.electronics",
    "sci.med",
    "sci.space",
    "soc.religion.christian",
    "talk.politics.guns",
    "talk.politics.mideast",
    "talk.politics.misc",
    "talk.religion.misc",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_headers_and_footer() {
        let raw = "From: a@b.c\nSubject: Hi\n\nHello world\n-- \nSignature";
        let t = strip_newsgroup_text(raw);
        assert!(t.contains("Hello"));
        assert!(!t.contains("From:"));
        assert!(!t.contains("Signature"));
    }

    #[test]
    fn hash_bow_nonzero() {
        let mut v = vec![0.0f32; 128];
        let row = ndarray::ArrayViewMut1::from(&mut v[..]);
        hash_bow_into_row("clustering embedding vectors news", row);
        assert!(v.iter().any(|&x| x > 0.0));
    }
}
