//! Union-find structures from `evoc.disjoint_set`.

pub struct RankDisjointSet {
    pub parent: Vec<i32>,
    pub rank: Vec<i32>,
}

pub fn ds_rank_create(n_elements: usize) -> RankDisjointSet {
    RankDisjointSet {
        parent: (0..n_elements as i32).collect(),
        rank: vec![0; n_elements],
    }
}

#[inline]
pub fn ds_find(ds: &mut RankDisjointSet, mut x: i32) -> i32 {
    while ds.parent[x as usize] != x {
        let p = ds.parent[x as usize];
        let gp = ds.parent[p as usize];
        ds.parent[x as usize] = gp;
        x = gp;
    }
    x
}

pub fn ds_union_by_rank(ds: &mut RankDisjointSet, mut x: i32, mut y: i32) {
    x = ds_find(ds, x);
    y = ds_find(ds, y);
    if x == y {
        return;
    }
    if ds.rank[x as usize] < ds.rank[y as usize] {
        std::mem::swap(&mut x, &mut y);
    }
    ds.parent[y as usize] = x;
    if ds.rank[x as usize] == ds.rank[y as usize] {
        ds.rank[x as usize] += 1;
    }
}
