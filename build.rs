fn main() {
    // C reference cosine for deterministic kNN (parity with Python goldens).
    if std::env::var("CARGO_FEATURE_KNN").is_ok() {
        cc::Build::new()
            .file("native/fast_cosine.c")
            .flag("-ffast-math")
            .flag("-O3")
            .compile("evoc_fast_cosine");
    }
}
