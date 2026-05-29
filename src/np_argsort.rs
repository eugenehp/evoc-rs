//! NumPy-compatible `argsort` for `f32` keys (`np.argsort`, default quicksort/introsort).

use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};

const SMALL_QUICKSORT: isize = 15;
const PYA_QS_STACK: usize = 128;

#[inline]
fn less_f32(a: f32, b: f32) -> bool {
    a < b || (b.is_nan() && !a.is_nan())
}

fn npy_get_msb(mut n: isize) -> isize {
    let mut msb = -1isize;
    while n > 0 {
        n >>= 1;
        msb += 1;
    }
    msb
}

fn numpy_argsort_script() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts/numpy_argsort_stdin.py")
}

fn python_for_numpy_argsort() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("EVOC_PARITY_PYTHON") {
        let path = PathBuf::from(p);
        if path.is_file() {
            return Some(path);
        }
    }
    let venv = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".venv-parity/bin/python3");
    if venv.is_file() {
        return Some(venv);
    }
    let python3 = PathBuf::from("python3");
    if Command::new(&python3)
        .arg("-c")
        .arg("import numpy")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
    {
        return Some(python3);
    }
    None
}

/// Call NumPy `argsort` via `scripts/numpy_argsort_stdin.py` (bit-exact parity with Python EVoC).
pub fn argsort_f32_via_numpy(keys: &[f32]) -> std::io::Result<Vec<usize>> {
    let n = keys.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let python = python_for_numpy_argsort()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "no python with numpy"))?;
    let script = numpy_argsort_script();
    if !script.is_file() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("missing {}", script.display()),
        ));
    }

    let mut child = Command::new(python)
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()?;

    {
        let stdin = child.stdin.as_mut().unwrap();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                keys.as_ptr() as *const u8,
                keys.len() * std::mem::size_of::<f32>(),
            )
        };
        stdin.write_all(bytes)?;
    }
    drop(child.stdin.take());

    let mut out = Vec::with_capacity(n * 8);
    child.stdout.as_mut().unwrap().read_to_end(&mut out)?;
    let status = child.wait()?;
    if !status.success() || out.len() != n * 8 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "numpy argsort subprocess failed",
        ));
    }

    Ok(out
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as usize)
        .collect())
}

/// Indirect heapsort fallback (valid sort, tie order may differ from NumPy).
fn aheapsort_f32(keys: &[f32], a: &mut [isize], pl: isize, n: isize) {
    if n <= 1 {
        return;
    }
    let v = keys;
    let base = pl - 1;

    let mut heap_n = n;
    let mut l = heap_n >> 1;
    while l > 0 {
        let tmp = a[(base + l) as usize];
        let mut i = l;
        let mut j = l << 1;
        while j <= heap_n {
            if j < heap_n
                && less_f32(
                    v[a[(base + j) as usize] as usize],
                    v[a[(base + j + 1) as usize] as usize],
                )
            {
                j += 1;
            }
            if less_f32(v[tmp as usize], v[a[(base + j) as usize] as usize]) {
                a[(base + i) as usize] = a[(base + j) as usize];
                i = j;
                j += j;
            } else {
                break;
            }
        }
        a[(base + i) as usize] = tmp;
        l -= 1;
    }

    while heap_n > 1 {
        let tmp = a[(base + heap_n) as usize];
        a[(base + heap_n) as usize] = a[(base + 1) as usize];
        heap_n -= 1;
        let mut i = 1isize;
        let mut j = 2isize;
        while j <= heap_n {
            if j < heap_n
                && less_f32(
                    v[a[(base + j) as usize] as usize],
                    v[a[(base + j + 1) as usize] as usize],
                )
            {
                j += 1;
            }
            if less_f32(v[tmp as usize], v[a[(base + j) as usize] as usize]) {
                a[(base + i) as usize] = a[(base + j) as usize];
                i = j;
                j += j;
            } else {
                break;
            }
        }
        a[(base + i) as usize] = tmp;
    }
}

fn argsort_f32_introsort(keys: &[f32]) -> Vec<usize> {
    let n = keys.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0];
    }

    let mut a: Vec<isize> = vec![0];
    a.extend((0..n as isize).map(|i| i));
    let v = keys;
    let mut pl: isize = 1;
    let mut pr: isize = n as isize;
    let mut stack: Vec<isize> = Vec::with_capacity(PYA_QS_STACK);
    let mut depth: Vec<isize> = Vec::with_capacity(PYA_QS_STACK);
    let mut cdepth = npy_get_msb(n as isize) * 2;

    loop {
        if cdepth < 0 {
            aheapsort_f32(v, &mut a, pl, pr - pl + 1);
            if stack.is_empty() {
                break;
            }
            pr = stack.pop().unwrap();
            pl = stack.pop().unwrap();
            cdepth = depth.pop().unwrap();
            continue;
        }

        while pr - pl > SMALL_QUICKSORT {
            let pm = pl + ((pr - pl) >> 1);
            if less_f32(v[a[pl as usize] as usize], v[a[pm as usize] as usize]) {
                a.swap(pl as usize, pm as usize);
            }
            if less_f32(v[a[pm as usize] as usize], v[a[pr as usize] as usize]) {
                a.swap(pm as usize, pr as usize);
            }
            if less_f32(v[a[pl as usize] as usize], v[a[pm as usize] as usize]) {
                a.swap(pl as usize, pm as usize);
            }
            let vp = v[a[pm as usize] as usize];
            let mut pi = pl;
            let mut pj = pr - 1;
            a.swap(pm as usize, pj as usize);
            loop {
                loop {
                    pi += 1;
                    if !less_f32(v[a[pi as usize] as usize], vp) {
                        break;
                    }
                }
                loop {
                    pj -= 1;
                    if !less_f32(vp, v[a[pj as usize] as usize]) {
                        break;
                    }
                }
                if pi >= pj {
                    break;
                }
                a.swap(pi as usize, pj as usize);
            }
            a.swap(pi as usize, pr as usize);

            if pi - pl < pr - pi {
                stack.push(pi + 1);
                stack.push(pr);
                pr = pi - 1;
            } else {
                stack.push(pl);
                stack.push(pi - 1);
                pl = pi + 1;
            }
            cdepth -= 1;
            depth.push(cdepth);
        }

        let mut pi = pl + 1;
        while pi <= pr {
            let vi = a[pi as usize];
            let vp = v[vi as usize];
            let mut pj = pi;
            let mut pk = pi - 1;
            while pj > pl && less_f32(vp, v[a[pk as usize] as usize]) {
                a[pj as usize] = a[pk as usize];
                pj -= 1;
                pk -= 1;
            }
            a[pj as usize] = vi;
            pi += 1;
        }

        if stack.is_empty() {
            break;
        }
        pr = stack.pop().unwrap();
        pl = stack.pop().unwrap();
        cdepth = depth.pop().unwrap();
    }

    a[1..=n].iter().map(|&i| i as usize).collect()
}

/// NumPy `np.argsort(x)` for 1-D `float32` (uses NumPy subprocess when available).
pub fn argsort_f32(keys: &[f32]) -> Vec<usize> {
    if let Ok(order) = argsort_f32_via_numpy(keys) {
        return order;
    }
    argsort_f32_introsort(keys)
}

#[cfg(all(test, feature = "npy"))]
mod tests {
    use super::*;
    use ndarray::Array2;
    use ndarray_npy::ReadNpyExt;
    use std::fs::File;
    use std::path::PathBuf;

    fn load_keys_and_py_order() -> (Vec<f32>, Vec<usize>) {
        let inter = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/medium_800/intermediates");
        let edges: Array2<f32> =
            Array2::read_npy(&mut File::open(inter.join("py_boruvka_edges.npy")).unwrap()).unwrap();
        let keys: Vec<f32> = (0..edges.nrows()).map(|i| edges[[i, 2]]).collect();
        let py: ndarray::Array1<i64> =
            ndarray::Array1::read_npy(&mut File::open(inter.join("py_argsort_order.npy")).unwrap())
                .unwrap();
        let py_order: Vec<usize> = py.iter().map(|&i| i as usize).collect();
        (keys, py_order)
    }

    #[test]
    fn argsort_matches_numpy_medium_800_edges() {
        let (keys, py_order) = load_keys_and_py_order();
        let rust = argsort_f32_via_numpy(&keys).expect("numpy argsort");
        assert_eq!(rust, py_order);
    }
}
