//! Hashed bag-of-words features for text clustering examples.

use std::hash::{Hash, Hasher};

pub const DEFAULT_FEATURE_DIM: usize = 16_384;

pub fn normalize_body(raw: &str) -> String {
    raw.replace('\r', "")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Strip common Usenet headers/footers (20 Newsgroups).
pub fn strip_newsgroup_text(raw: &str) -> String {
    let mut text = raw.replace('\r', "");
    if let Some(pos) = text.find("\n\n") {
        text = text[pos + 2..].to_string();
    }
    if let Some(pos) = text.rfind("\n-- \n") {
        text.truncate(pos);
    }
    if let Some(pos) = text.rfind("\n--\n") {
        text.truncate(pos);
    }
    normalize_body(&text)
}

fn is_stopword(word: &str) -> bool {
    matches!(
        word,
        "a" | "an"
            | "and"
            | "are"
            | "as"
            | "at"
            | "be"
            | "been"
            | "but"
            | "by"
            | "can"
            | "did"
            | "do"
            | "for"
            | "from"
            | "had"
            | "has"
            | "have"
            | "he"
            | "her"
            | "him"
            | "his"
            | "if"
            | "in"
            | "into"
            | "is"
            | "it"
            | "its"
            | "may"
            | "more"
            | "most"
            | "not"
            | "of"
            | "on"
            | "or"
            | "our"
            | "out"
            | "she"
            | "so"
            | "such"
            | "than"
            | "that"
            | "the"
            | "their"
            | "them"
            | "then"
            | "there"
            | "these"
            | "they"
            | "this"
            | "to"
            | "up"
            | "was"
            | "we"
            | "were"
            | "what"
            | "when"
            | "which"
            | "who"
            | "will"
            | "with"
            | "would"
            | "you"
            | "your"
    )
}

fn normalize_token(raw: &str) -> Option<String> {
    let t: String = raw
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect();
    if t.len() < 3 || is_stopword(&t) {
        None
    } else {
        Some(t)
    }
}

pub fn hash_bow_into_row(text: &str, mut row: ndarray::ArrayViewMut1<f32>) {
    let dim = row.len();
    for token in text.split_whitespace() {
        let Some(t) = normalize_token(token) else {
            continue;
        };
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        t.hash(&mut hasher);
        let idx = (hasher.finish() as usize) % dim;
        row[idx] += 1.0;
    }
    row.mapv_inplace(|v| if v > 0.0 { (1.0 + v).ln() } else { 0.0 });
}
