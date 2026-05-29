//! Download Fashion-MNIST, subsample, L2-normalize, write `.npy` files.
//!
//! ```bash
//! cargo run --release --bin fashion_mnist_fetch -- [n] [seed] [data.npy] [labels.npy]
//! ```

use evoc::fashion_mnist_data::{self, FashionMnistError};
use ndarray::Array1;
use ndarray_npy::WriteNpyExt;
use std::fs::File;
use std::path::PathBuf;

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), FashionMnistError> {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3000);
    let seed: u64 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    let data_path = std::env::args()
        .nth(3)
        .unwrap_or_else(|| format!("fashion_mnist_{n}_{seed}.npy"));
    let labels_path = std::env::args().nth(4);

    let cache = fashion_mnist_data::cache_dir();
    eprintln!(
        "Fashion-MNIST n={n} seed={seed} cache={} ...",
        cache.display()
    );
    let (data, labels, _) = fashion_mnist_data::sample_normalized(n, seed, &cache)?;

    write_npy(&data_path, &data)?;
    eprintln!(
        "wrote {} shape=({},{})",
        data_path,
        data.nrows(),
        data.ncols()
    );

    if let Some(path) = labels_path {
        let labels_i64: Array1<i64> = labels.mapv(i64::from);
        write_npy(&path, &labels_i64)?;
        eprintln!("wrote {path}");
    }

    Ok(())
}

fn write_npy<T, S, D>(path: &str, array: &ndarray::ArrayBase<S, D>) -> Result<(), FashionMnistError>
where
    T: ndarray_npy::WritableElement,
    S: ndarray::Data<Elem = T>,
    D: ndarray::Dimension,
{
    let path = PathBuf::from(path);
    array
        .write_npy(File::create(&path).map_err(|source| FashionMnistError::Io {
            path: path.clone(),
            source,
        })?)
        .map_err(|err| FashionMnistError::Io {
            path,
            source: std::io::Error::new(std::io::ErrorKind::Other, err),
        })
}
