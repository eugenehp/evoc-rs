//! SciPy-compatible CSR × CSR multiply (`scipy.sparse._sparsetools.csr_matmat`).

use sprs::{CompressedStorage, CsMat};

/// Sparse matrix multiply with SciPy SMMP nnz ordering (required for embedding parity).
pub fn scipy_csr_matmul(a: &CsMat<f32>, b: &CsMat<f32>) -> CsMat<f32> {
    assert_eq!(a.cols(), b.rows());
    let n_row = a.rows();
    let n_col = b.cols();
    let ap: Vec<usize> = a.indptr().raw_storage().to_vec();
    let ai = a.indices();
    let ad = a.data();
    let bp: Vec<usize> = b.indptr().raw_storage().to_vec();
    let bi = b.indices();
    let bd = b.data();

    let mut next = vec![-1i32; n_col];
    let mut sums = vec![0.0f32; n_col];
    let mut cp = vec![0usize; n_row + 1];
    let mut cj = Vec::new();
    let mut cd = Vec::new();
    let mut nnz = 0usize;

    for i in 0..n_row {
        let mut head: i32 = -2;
        let mut length = 0i32;

        for jj in ap[i]..ap[i + 1] {
            let j = ai[jj];
            let v = ad[jj];
            for kk in bp[j]..bp[j + 1] {
                let k = bi[kk];
                sums[k] += v * bd[kk];
                if next[k] == -1 {
                    next[k] = head;
                    head = k as i32;
                    length += 1;
                }
            }
        }

        for _ in 0..length {
            let k = head as usize;
            if sums[k] != 0.0 {
                cj.push(k);
                cd.push(sums[k]);
                nnz += 1;
            }
            let temp = head;
            head = next[k];
            next[k] = -1;
            sums[k] = 0.0;
            let _ = temp;
        }
        cp[i + 1] = nnz;
    }

    unsafe { CsMat::new_unchecked(CompressedStorage::CSR, (n_row, n_col), cp, cj, cd) }
}

/// Element-wise product on aligned CSR entries (SciPy `sparse.multiply`).
pub fn scipy_csr_elementwise_mul(a: &CsMat<f32>, b: &CsMat<f32>) -> CsMat<f32> {
    scipy_csr_rowwise_binop(a, b, |x, y| x * y, BinopMode::Intersect)
}

/// Sparse sum (SciPy `sparse.+`).
pub fn scipy_csr_add(a: &CsMat<f32>, b: &CsMat<f32>) -> CsMat<f32> {
    scipy_csr_rowwise_binop(a, b, |x, y| x + y, BinopMode::Union)
}

/// Sparse difference (SciPy `sparse.-`).
pub fn scipy_csr_sub(a: &CsMat<f32>, b: &CsMat<f32>) -> CsMat<f32> {
    scipy_csr_rowwise_binop(a, b, |x, y| x - y, BinopMode::Union)
}

enum BinopMode {
    Union,
    Intersect,
}

fn scipy_csr_rowwise_binop(
    a: &CsMat<f32>,
    b: &CsMat<f32>,
    op: impl Fn(f32, f32) -> f32,
    mode: BinopMode,
) -> CsMat<f32> {
    assert_eq!(a.shape(), b.shape());
    let n_row = a.rows();
    let n_col = a.cols();
    let ap: Vec<usize> = a.indptr().raw_storage().to_vec();
    let bp: Vec<usize> = b.indptr().raw_storage().to_vec();
    let ai = a.indices();
    let bi = b.indices();
    let ad = a.data();
    let bd = b.data();

    let mut cp = vec![0usize; n_row + 1];
    let mut cj = Vec::new();
    let mut cd = Vec::new();

    for i in 0..n_row {
        let mut ja = ap[i];
        let mut jb = bp[i];
        let ea = ap[i + 1];
        let eb = bp[i + 1];
        while ja < ea || jb < eb {
            let col_a = if ja < ea { ai[ja] } else { usize::MAX };
            let col_b = if jb < eb { bi[jb] } else { usize::MAX };
            let col = col_a.min(col_b);
            if col == usize::MAX {
                break;
            }
            let va = if col_a == col { ad[ja] } else { 0.0 };
            let vb = if col_b == col { bd[jb] } else { 0.0 };
            let has_a = col_a == col;
            let has_b = col_b == col;
            let out = match mode {
                BinopMode::Union => {
                    if has_a || has_b {
                        Some(op(va, vb))
                    } else {
                        None
                    }
                }
                BinopMode::Intersect => {
                    if has_a && has_b {
                        Some(op(va, vb))
                    } else {
                        None
                    }
                }
            };
            if let Some(v) = out {
                if v != 0.0 {
                    cj.push(col);
                    cd.push(v);
                }
            }
            if col_a == col {
                ja += 1;
            }
            if col_b == col {
                jb += 1;
            }
        }
        cp[i + 1] = cj.len();
    }

    unsafe { CsMat::new_unchecked(CompressedStorage::CSR, (n_row, n_col), cp, cj, cd) }
}
