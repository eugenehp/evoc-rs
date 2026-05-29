//! Shared NN-descent helpers (re-exports from `heap` for module layout).

#[allow(unused_imports)]
pub use crate::heap::{
    apply_graph_update_array, apply_sorted_graph_updates, build_candidates,
    build_candidates_heap_push, deheap_sort, flagged_heap_push, make_heap, GraphHeap, INF,
};
