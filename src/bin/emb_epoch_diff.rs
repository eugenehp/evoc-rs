use evoc::embed_kernels::{make_epochs_per_sample, node_embedding_epoch_repr};
use evoc::{load_graph_coo_npz, load_graph_csr_npz, NumpyRandomState};
use ndarray::Array2;
use ndarray_npy::ReadNpyExt;
use sprs::CsMat;
use std::fs::File;
use std::path::PathBuf;

fn max_abs(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn load_numpy_rng_from(inter: &PathBuf, tag: &str) -> NumpyRandomState {
    let mut kf = File::open(inter.join(format!("rng_{tag}_key.npy"))).unwrap();
    let key: ndarray::Array1<u32> = ndarray::Array1::read_npy(&mut kf).unwrap();
    let mut arr = [0u32; 624];
    arr.copy_from_slice(key.as_slice().unwrap());
    let mut npz =
        ndarray_npy::NpzReader::new(File::open(inter.join(format!("rng_{tag}_meta.npz"))).unwrap())
            .unwrap();
    let pos: ndarray::Array0<i32> = npz.by_name("pos").unwrap();
    let has_gauss: ndarray::Array0<i32> = npz.by_name("has_gauss").unwrap();
    let gauss: ndarray::Array0<f64> = npz.by_name("gauss").unwrap();
    NumpyRandomState::from_numpy_state(
        &arr,
        pos.as_slice().unwrap()[0],
        has_gauss.as_slice().unwrap()[0],
        gauss.as_slice().unwrap()[0],
    )
}

fn main() {
    let fixture = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "large_2000".to_string());
    let mode = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "full".to_string());

    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(&fixture);
    let inter = dir.join("intermediates");

    if mode == "single" {
        let epoch: usize = std::env::args()
            .nth(3)
            .and_then(|s| s.parse().ok())
            .unwrap_or(20);
        run_single_epoch(&inter, epoch);
        return;
    }

    let n_epochs: usize = mode.parse().unwrap_or(50);
    run_full(&inter, n_epochs);
}

fn run_single_epoch(inter: &PathBuf, epoch: usize) {
    let mut npz = ndarray_npy::NpzReader::new(
        File::open(inter.join("emb_state_epoch20.npz")).expect("state npz"),
    )
    .expect("reader");
    let mut embedding: Array2<f32> = npz.by_name("embedding").expect("embedding");
    let mut updates: Array2<f32> = npz.by_name("updates").expect("updates");
    let node_order: ndarray::Array1<u32> = npz.by_name("node_order").expect("node_order");
    let epoch_of_next_sample: ndarray::Array1<f32> =
        npz.by_name("epoch_of_next_sample").expect("eos");
    let epoch_of_next_negative_sample: ndarray::Array1<f32> =
        npz.by_name("epoch_of_next_negative_sample").expect("eon");
    let alpha0: ndarray::Array0<f32> = npz.by_name("alpha").expect("alpha");
    let rng_vals: ndarray::Array1<i64> = npz.by_name("rng_vals").expect("rng_vals");

    let graph: CsMat<f32> = load_graph_csr_npz(inter.join("graph_csr.npz").as_path())
        .or_else(|| load_graph_coo_npz(inter.join("graph_coo.npz").as_path()))
        .expect("graph");
    let graph = graph.to_csr();
    let n_vertices = graph.rows();
    let n_epochs = 50usize;
    let weights: Vec<f32> = graph.data().to_vec();
    let epochs_per_sample = make_epochs_per_sample(&weights, n_epochs);
    let mut epochs_per_negative_sample: Vec<f32> =
        epochs_per_sample.iter().map(|&e| e / 1.0).collect();
    for e in &mut epochs_per_negative_sample {
        *e *= 1.5;
    }

    let csr_indptr: Vec<u32> = graph
        .indptr()
        .raw_storage()
        .iter()
        .map(|&p| p as u32)
        .collect();
    let csr_indices: Vec<u32> = graph.indices().iter().map(|&i| i as u32).collect();
    let dim = embedding.ncols().min(255);
    let gamma = 0.5 + (epoch as f32 / (n_epochs - 1) as f32);
    let block_size = (1024usize).max(n_vertices / 8) as u32;
    let alpha = alpha0.as_slice().unwrap()[0];

    let mut node_order_vec = node_order.to_vec();
    let mut eos = epoch_of_next_sample.to_vec();
    let mut eon = epoch_of_next_negative_sample.to_vec();

    node_embedding_epoch_repr(
        &mut embedding,
        &csr_indptr,
        &csr_indices,
        n_vertices as u32,
        &epochs_per_sample,
        rng_vals[epoch] as u32,
        dim,
        alpha,
        &epochs_per_negative_sample,
        &mut eon,
        &mut eos,
        epoch as u8,
        0.5,
        gamma,
        &mut updates,
        &mut node_order_vec,
        block_size,
    );

    let py: Array2<f32> = {
        let mut npz = ndarray_npy::NpzReader::new(
            File::open(inter.join("emb_after_epoch20_single.npz")).expect("py single"),
        )
        .expect("reader");
        npz.by_name("embedding").expect("embedding")
    };
    println!(
        "single epoch {epoch} max_abs {:.9e}",
        max_abs(&embedding, &py)
    );
}

fn run_full(inter: &PathBuf, n_epochs: usize) {
    let graph: CsMat<f32> = load_graph_csr_npz(inter.join("graph_csr.npz").as_path())
        .or_else(|| load_graph_coo_npz(inter.join("graph_coo.npz").as_path()))
        .expect("graph");
    let graph = graph.to_csr();
    let n_vertices = graph.rows();

    let mut init_f = File::open(inter.join("init_embedding.npy")).unwrap();
    let mut embedding: Array2<f32> = Array2::read_npy(&mut init_f).unwrap();
    let dim = embedding.ncols().min(255);

    let weights: Vec<f32> = graph.data().to_vec();
    let epochs_per_sample = make_epochs_per_sample(&weights, n_epochs);
    let mut epochs_per_negative_sample: Vec<f32> =
        epochs_per_sample.iter().map(|&e| e / 1.0).collect();
    for e in &mut epochs_per_negative_sample {
        *e *= 1.5;
    }
    let mut epoch_of_next_negative_sample = epochs_per_negative_sample.clone();
    let mut epoch_of_next_sample = epochs_per_sample.clone();

    let csr_indptr: Vec<u32> = graph
        .indptr()
        .raw_storage()
        .iter()
        .map(|&p| p as u32)
        .collect();
    let csr_indices: Vec<u32> = graph.indices().iter().map(|&i| i as u32).collect();

    let mut updates = Array2::zeros((n_vertices, embedding.ncols()));
    let mut node_order: Vec<u32> = (0..n_vertices as u32).collect();

    let gamma_schedule: Vec<f32> = (0..n_epochs)
        .map(|n| {
            if n_epochs <= 1 {
                1.0
            } else {
                0.5 + (n as f32 / (n_epochs - 1) as f32)
            }
        })
        .collect();

    let block_size = (1024usize).max(n_vertices / 8) as u32;
    let alpha0 = 0.1f32;
    let noise_level = 0.5f32;

    // Match Python: pre-draw rng vals and shuffle after each epoch.
    let mut rng = load_numpy_rng_from(&inter, "after_init");
    let rng_vals: Vec<u32> = (0..n_epochs)
        .map(|_| rng.randint_high((i32::MAX - 1) as i64) as u32)
        .collect();

    let py_dir = inter.join("emb_epochs");
    let mut alpha = alpha0;
    for n in 0..n_epochs {
        node_embedding_epoch_repr(
            &mut embedding,
            &csr_indptr,
            &csr_indices,
            n_vertices as u32,
            &epochs_per_sample,
            rng_vals[n],
            dim,
            alpha,
            &epochs_per_negative_sample,
            &mut epoch_of_next_negative_sample,
            &mut epoch_of_next_sample,
            n as u8,
            noise_level,
            gamma_schedule[n],
            &mut updates,
            &mut node_order,
            block_size,
        );
        let decay = ((1.0 - f64::from(alpha)).powi(2) * 0.5) as f32;
        updates.mapv_inplace(|v| v * decay);
        rng.shuffle(&mut node_order);
        alpha = alpha0 * (1.0 - (n as f32 / n_epochs as f32));

        let mut f = File::open(py_dir.join(format!("emb_epoch_{n:03}.npy"))).unwrap();
        let py: Array2<f32> = Array2::read_npy(&mut f).unwrap();
        let diff = max_abs(&embedding, &py);
        println!("epoch {n:03} max_abs {diff:.9e}");
    }
}
