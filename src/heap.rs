//! NN-descent heap operations (port of `evoc.common_nndescent`).

use crate::rng::{offset_state, tau_rand};
use ndarray::parallel::prelude::*;
use ndarray::{Array2, Axis};

pub const INF: f32 = f32::MAX;

/// Raw pointer wrapper for disjoint parallel row updates.
pub(crate) struct SendMutPtr<T>(*mut T);
unsafe impl<T> Send for SendMutPtr<T> {}
unsafe impl<T> Sync for SendMutPtr<T> {}

impl<T> SendMutPtr<T> {
    #[inline]
    pub(crate) fn new(ptr: *mut T) -> Self {
        Self(ptr)
    }

    #[inline]
    pub(crate) unsafe fn as_ptr(&self) -> *mut T {
        self.0
    }
}

/// `(indices, distances, flags)` neighbor heap.
pub type GraphHeap = (Array2<i32>, Array2<f32>, Array2<u8>);

pub fn make_heap(n_points: usize, size: usize) -> GraphHeap {
    let indices = Array2::from_elem((n_points, size), -1);
    let distances = Array2::from_elem((n_points, size), INF);
    let flags = Array2::zeros((n_points, size));
    (indices, distances, flags)
}

#[inline]
fn siftdown(heap1: &mut [f32], heap2: &mut [i32], mut elt: usize) {
    let n = heap1.len();
    while elt * 2 + 1 < n {
        let left_child = elt * 2 + 1;
        let right_child = left_child + 1;
        let mut swap = elt;

        if heap1[swap] < heap1[left_child] {
            swap = left_child;
        }

        if right_child < n && heap1[swap] < heap1[right_child] {
            swap = right_child;
        }

        if swap == elt {
            break;
        }

        heap1.swap(elt, swap);
        heap2.swap(elt, swap);
        elt = swap;
    }
}

fn deheap_sort_row(indices: &mut [i32], distances: &mut [f32]) {
    let n = indices.len();
    for j in (1..n).rev() {
        indices.swap(0, j);
        distances.swap(0, j);
        siftdown(&mut distances[..j], &mut indices[..j], 0);
    }
}

/// In-place heap sort of each row (parallel over vertices).
pub fn deheap_sort(indices: &mut Array2<i32>, distances: &mut Array2<f32>) {
    indices
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(distances.axis_iter_mut(Axis(0)).into_par_iter())
        .for_each(|(mut idx_row, mut dist_row)| {
            deheap_sort_row(
                idx_row.as_slice_mut().unwrap(),
                dist_row.as_slice_mut().unwrap(),
            );
        });
}

/// Max-heap push for candidate generation (priority = random key).
pub fn build_candidates_heap_push(
    priorities: &mut [f32],
    indices: &mut [i32],
    p: f32,
    n: i32,
) -> u8 {
    if p >= priorities[0] {
        return 0;
    }

    let size = priorities.len();

    for i in 0..size {
        if n == indices[i] {
            return 0;
        }
    }

    priorities[0] = p;
    indices[0] = n;

    let mut i = 0usize;
    loop {
        let ic1 = 2 * i + 1;
        let ic2 = ic1 + 1;

        let i_swap = if ic1 >= size {
            break;
        } else if ic2 >= size {
            if priorities[ic1] > p {
                ic1
            } else {
                break;
            }
        } else if priorities[ic1] >= priorities[ic2] {
            if p < priorities[ic1] {
                ic1
            } else {
                break;
            }
        } else if p < priorities[ic2] {
            ic2
        } else {
            break;
        };

        priorities[i] = priorities[i_swap];
        indices[i] = indices[i_swap];
        i = i_swap;
    }

    priorities[i] = p;
    indices[i] = n;
    1
}

/// Push into the neighbor graph heap; sets `flags[i] = 1` for the new slot.
pub fn flagged_heap_push(
    priorities: &mut [f32],
    indices: &mut [i32],
    flags: &mut [u8],
    p: f32,
    n: i32,
) -> u8 {
    if p >= priorities[0] {
        return 0;
    }

    let size = priorities.len();

    for i in 0..size {
        if n == indices[i] {
            return 0;
        }
    }

    priorities[0] = p;
    indices[0] = n;

    let mut i = 0usize;
    loop {
        let ic1 = 2 * i + 1;
        let ic2 = ic1 + 1;

        let i_swap = if ic1 >= size {
            break;
        } else if ic2 >= size {
            if priorities[ic1] > p {
                ic1
            } else {
                break;
            }
        } else if priorities[ic1] >= priorities[ic2] {
            if p < priorities[ic1] {
                ic1
            } else {
                break;
            }
        } else if p < priorities[ic2] {
            ic2
        } else {
            break;
        };

        priorities[i] = priorities[i_swap];
        indices[i] = indices[i_swap];
        flags[i] = flags[i_swap];
        i = i_swap;
    }

    priorities[i] = p;
    indices[i] = n;
    flags[i] = 1;
    1
}

/// Build candidate neighbor heaps for one NN-descent iteration.
pub fn build_candidates(
    current_graph: &mut GraphHeap,
    max_candidates: usize,
    rng_state: &[i64; 3],
    n_threads: usize,
) -> (Array2<i32>, Array2<i32>) {
    let (current_indices, _, current_flags) = current_graph;
    let n_vertices = current_indices.nrows();
    let n_neighbors = current_indices.ncols();

    let mut new_candidate_indices = Array2::from_elem((n_vertices, max_candidates), -1);
    let mut new_candidate_priority = Array2::from_elem((n_vertices, max_candidates), INF);
    let mut old_candidate_indices = Array2::from_elem((n_vertices, max_candidates), -1);
    let mut old_candidate_priority = Array2::from_elem((n_vertices, max_candidates), INF);

    let block_size = n_vertices / n_threads + 1;

    let new_pri_ptr = SendMutPtr::new(new_candidate_priority.as_mut_ptr());
    let new_idx_ptr = SendMutPtr::new(new_candidate_indices.as_mut_ptr());
    let old_pri_ptr = SendMutPtr::new(old_candidate_priority.as_mut_ptr());
    let old_idx_ptr = SendMutPtr::new(old_candidate_indices.as_mut_ptr());

    (0..n_threads).into_par_iter().for_each(|n| {
        let mut local_rng = offset_state(rng_state, n as i64);
        let block_start = n * block_size;
        let block_end = (block_start + block_size).min(n_vertices);

        for i in 0..n_vertices {
            for j in 0..n_neighbors {
                let idx = current_indices[[i, j]];
                if idx < 0 {
                    continue;
                }

                if !((i >= block_start && i < block_end)
                    || ((idx as usize) >= block_start && (idx as usize) < block_end))
                {
                    continue;
                }

                let isn = current_flags[[i, j]] != 0;
                let d = tau_rand(&mut local_rng);

                if isn {
                    if i >= block_start && i < block_end {
                        unsafe {
                            build_candidates_heap_push(
                                std::slice::from_raw_parts_mut(
                                    new_pri_ptr.as_ptr().add(i * max_candidates),
                                    max_candidates,
                                ),
                                std::slice::from_raw_parts_mut(
                                    new_idx_ptr.as_ptr().add(i * max_candidates),
                                    max_candidates,
                                ),
                                d,
                                idx,
                            );
                        }
                    }
                    if (idx as usize) >= block_start && (idx as usize) < block_end {
                        let idx_u = idx as usize;
                        unsafe {
                            build_candidates_heap_push(
                                std::slice::from_raw_parts_mut(
                                    new_pri_ptr.as_ptr().add(idx_u * max_candidates),
                                    max_candidates,
                                ),
                                std::slice::from_raw_parts_mut(
                                    new_idx_ptr.as_ptr().add(idx_u * max_candidates),
                                    max_candidates,
                                ),
                                d,
                                i as i32,
                            );
                        }
                    }
                } else {
                    if i >= block_start && i < block_end {
                        unsafe {
                            build_candidates_heap_push(
                                std::slice::from_raw_parts_mut(
                                    old_pri_ptr.as_ptr().add(i * max_candidates),
                                    max_candidates,
                                ),
                                std::slice::from_raw_parts_mut(
                                    old_idx_ptr.as_ptr().add(i * max_candidates),
                                    max_candidates,
                                ),
                                d,
                                idx,
                            );
                        }
                    }
                    if (idx as usize) >= block_start && (idx as usize) < block_end {
                        let idx_u = idx as usize;
                        unsafe {
                            build_candidates_heap_push(
                                std::slice::from_raw_parts_mut(
                                    old_pri_ptr.as_ptr().add(idx_u * max_candidates),
                                    max_candidates,
                                ),
                                std::slice::from_raw_parts_mut(
                                    old_idx_ptr.as_ptr().add(idx_u * max_candidates),
                                    max_candidates,
                                ),
                                d,
                                i as i32,
                            );
                        }
                    }
                }
            }
        }
    });

    let indices = &current_graph.0;
    let flags = &mut current_graph.2;

    flags
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(indices.axis_iter(Axis(0)))
        .zip(new_candidate_indices.axis_iter(Axis(0)))
        .for_each(|((mut flag_row, idx_row), new_row)| {
            for j in 0..n_neighbors {
                let idx = idx_row[j];
                if idx < 0 {
                    continue;
                }
                for k in 0..max_candidates {
                    if new_row[k] == idx {
                        flag_row[j] = 0;
                        break;
                    }
                }
            }
        });

    (new_candidate_indices, old_candidate_indices)
}

/// Apply unsorted per-thread update batches to the graph.
pub fn apply_graph_update_array(
    current_graph: &mut GraphHeap,
    update_array: &ndarray::Array3<f32>,
    n_updates_per_thread: &[i32],
    n_threads: usize,
) -> u32 {
    let n_vertices = current_graph.1.nrows();
    let ncols = current_graph.1.ncols();
    let block_size = n_vertices / n_threads + 1;

    let indices_ptr = SendMutPtr::new(current_graph.0.as_mut_ptr());
    let distances_ptr = SendMutPtr::new(current_graph.1.as_mut_ptr());
    let flags_ptr = SendMutPtr::new(current_graph.2.as_mut_ptr());

    (0..n_threads)
        .into_par_iter()
        .map(|n| {
            let block_start = n * block_size;
            let block_end = (block_start + block_size).min(n_vertices);
            let mut local_changes = 0u32;

            for t in 0..n_threads {
                let count = n_updates_per_thread[t] as usize;
                for j in 0..count {
                    let p = update_array[[t, j, 0]] as i32;
                    if p == -1 {
                        break;
                    }
                    let q = update_array[[t, j, 1]] as i32;
                    let d = update_array[[t, j, 2]];

                    if (p as usize) >= block_start && (p as usize) < block_end {
                        unsafe {
                            local_changes += flagged_heap_push(
                                std::slice::from_raw_parts_mut(
                                    distances_ptr.as_ptr().add(p as usize * ncols),
                                    ncols,
                                ),
                                std::slice::from_raw_parts_mut(
                                    indices_ptr.as_ptr().add(p as usize * ncols),
                                    ncols,
                                ),
                                std::slice::from_raw_parts_mut(
                                    flags_ptr.as_ptr().add(p as usize * ncols),
                                    ncols,
                                ),
                                d,
                                q,
                            ) as u32;
                        }
                    }
                    if (q as usize) >= block_start && (q as usize) < block_end {
                        unsafe {
                            local_changes += flagged_heap_push(
                                std::slice::from_raw_parts_mut(
                                    distances_ptr.as_ptr().add(q as usize * ncols),
                                    ncols,
                                ),
                                std::slice::from_raw_parts_mut(
                                    indices_ptr.as_ptr().add(q as usize * ncols),
                                    ncols,
                                ),
                                std::slice::from_raw_parts_mut(
                                    flags_ptr.as_ptr().add(q as usize * ncols),
                                    ncols,
                                ),
                                d,
                                p,
                            ) as u32;
                        }
                    }
                }
            }
            local_changes
        })
        .sum()
}

/// Apply updates bucketed by target vertex block.
pub fn apply_sorted_graph_updates(
    current_graph: &mut GraphHeap,
    update_array: &ndarray::Array3<f32>,
    n_updates_per_block: &Array2<i32>,
    n_threads: usize,
) -> u32 {
    let n_vertices = current_graph.1.nrows();
    let ncols = current_graph.1.ncols();
    let vertex_block_size = n_vertices / n_threads + 1;
    let max_updates_per_thread = update_array.len_of(Axis(1)) / n_threads;

    let indices_ptr = SendMutPtr::new(current_graph.0.as_mut_ptr());
    let distances_ptr = SendMutPtr::new(current_graph.1.as_mut_ptr());
    let flags_ptr = SendMutPtr::new(current_graph.2.as_mut_ptr());

    (0..n_threads)
        .into_par_iter()
        .map(|n| {
            let block_start = n * vertex_block_size;
            let block_end = (block_start + vertex_block_size).min(n_vertices);
            let mut local_changes = 0u32;

            for t in 0..n_threads {
                let thread_start = t * max_updates_per_thread;
                let thread_count = n_updates_per_block[[n, t + 1]] as usize;

                for j in 0..thread_count {
                    let idx = thread_start + j;
                    let p = update_array[[n, idx, 0]] as i32;
                    let q = update_array[[n, idx, 1]] as i32;
                    let d = update_array[[n, idx, 2]];

                    if (p as usize) >= block_start && (p as usize) < block_end {
                        unsafe {
                            local_changes += flagged_heap_push(
                                std::slice::from_raw_parts_mut(
                                    distances_ptr.as_ptr().add(p as usize * ncols),
                                    ncols,
                                ),
                                std::slice::from_raw_parts_mut(
                                    indices_ptr.as_ptr().add(p as usize * ncols),
                                    ncols,
                                ),
                                std::slice::from_raw_parts_mut(
                                    flags_ptr.as_ptr().add(p as usize * ncols),
                                    ncols,
                                ),
                                d,
                                q,
                            ) as u32;
                        }
                    }
                    if (q as usize) >= block_start && (q as usize) < block_end {
                        unsafe {
                            local_changes += flagged_heap_push(
                                std::slice::from_raw_parts_mut(
                                    distances_ptr.as_ptr().add(q as usize * ncols),
                                    ncols,
                                ),
                                std::slice::from_raw_parts_mut(
                                    indices_ptr.as_ptr().add(q as usize * ncols),
                                    ncols,
                                ),
                                std::slice::from_raw_parts_mut(
                                    flags_ptr.as_ptr().add(q as usize * ncols),
                                    ncols,
                                ),
                                d,
                                p,
                            ) as u32;
                        }
                    }
                }
            }
            local_changes
        })
        .sum()
}
