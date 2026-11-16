use super::graph::toy;
use crate::spatial_baselines::cluster::{
    cluster_embedding, kmeans_restarts, KMEANS_MAX_ITER, KMEANS_RESTARTS,
};
use crate::spatial_baselines::graph::knn_within_sections;
use crate::spatial_baselines::stagate::*;
use crate::util::common::Mat;
use legume_numeric::candle::candle_core::{DType, Device, Tensor, Var};
use nalgebra_sparse::{CooMatrix, CscMatrix};

#[test]
fn loess_reproduces_a_quadratic() {
    let x: Vec<f64> = (0..200).map(|i| (i as f64 * 0.37).sin() * 3.0).collect();
    let y: Vec<f64> = x.iter().map(|v| 1.0 + 2.0 * v - 0.5 * v * v).collect();
    let fit = loess_quadratic(&x, &y, 0.3);
    for (f, t) in fit.iter().zip(&y) {
        assert!((f - t).abs() < 1e-8, "{f} vs {t}");
    }
}

#[test]
fn seurat_v3_ranks_an_overdispersed_gene_first() {
    // 50 genes x 400 cells: gene g has mean (g + 1) / 5, Poisson-like
    // except gene 7, which is zero in most cells and large in a few
    let (n_genes, n) = (50usize, 400usize);
    let mut coo = CooMatrix::<f32>::new(n_genes, n);
    let mut state = 12345u64;
    let mut unif = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    for c in 0..n {
        for g in 0..n_genes {
            let v = if g == 7 {
                if c % 20 == 0 {
                    40.0
                } else {
                    0.0
                }
            } else {
                // Poisson by inversion
                let lam = (g as f64 + 1.0) / 5.0;
                let (mut k, mut p, mut s) = (0.0, (-lam).exp(), (-lam).exp());
                let u = unif();
                while u > s {
                    k += 1.0;
                    p *= lam / k;
                    s += p;
                }
                k
            };
            if v > 0.0 {
                coo.push(g, c, v as f32);
            }
        }
    }
    let counts = CscMatrix::from(&coo);
    let (top, nv) = seurat_v3_hvg(&counts, 1);
    assert_eq!(top, vec![7]);
    assert!(nv[7] > 5.0, "{}", nv[7]);
    let (all, _) = seurat_v3_hvg(&counts, 3000);
    assert_eq!(all, (0..n_genes).collect::<Vec<_>>());
}

#[test]
fn spatial_edges_add_self_loops_and_drop_zero_distance() {
    // cells 0 and 1 share a position; 2 and 3 are apart
    let d = toy(&[(0., 0.), (0., 0.), (1., 0.), (3., 0.)], &[0, 0, 0, 0]);
    let nb = knn_within_sections(&d, 2);
    let (src, dst) = spatial_edges(&nb);
    let edges: Vec<(u32, u32)> = src.into_iter().zip(dst).collect();
    for i in 0..4u32 {
        assert!(edges.contains(&(i, i)));
    }
    assert!(!edges.contains(&(0, 1)) && !edges.contains(&(1, 0)));
    // cell 3's two nearest are 2 and one of 0/1
    assert!(edges.contains(&(3, 2)));
    let mut sorted = edges.clone();
    sorted.dedup();
    assert_eq!(sorted.len(), edges.len());
}

fn ring(n: usize, dev: &Device) -> EdgeIndex {
    // each node i sends to i-1, i, i+1
    let mut src = Vec::new();
    let mut dst = Vec::new();
    for i in 0..n {
        for j in [(i + n - 1) % n, i, (i + 1) % n] {
            src.push(j as u32);
            dst.push(i as u32);
        }
    }
    EdgeIndex::new(&src, &dst, n, dev).unwrap()
}

#[test]
fn attention_is_a_softmax_of_sigmoid_logits_over_incoming_edges() {
    let dev = Device::Cpu;
    let g = ring(5, &dev);
    let h = Tensor::from_vec(
        (0..10).map(|v| (v as f32 * 0.7).sin()).collect::<Vec<_>>(),
        (5, 2),
        &dev,
    )
    .unwrap();
    let a_src = Tensor::from_vec(vec![0.5f32, -1.0], (2, 1), &dev).unwrap();
    let a_dst = Tensor::from_vec(vec![2.0f32, 0.3], (2, 1), &dev).unwrap();
    let alpha: Vec<f32> = attention(&h, &a_src, &a_dst, &g)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let hv: Vec<Vec<f32>> = h.to_vec2().unwrap();
    let src: Vec<u32> = g.src.to_vec1().unwrap();
    let dst: Vec<u32> = g.dst.to_vec1().unwrap();
    let s = |i: usize| hv[i][0] * 0.5 - hv[i][1];
    let d = |i: usize| hv[i][0] * 2.0 + hv[i][1] * 0.3;
    let sig = |v: f32| 1.0 / (1.0 + (-v).exp());
    for i in 0..5usize {
        let es: Vec<usize> = (0..src.len()).filter(|&e| dst[e] as usize == i).collect();
        let total: f32 = es.iter().map(|&e| alpha[e]).sum();
        assert!((total - 1.0).abs() < 1e-6);
        let z: f32 = es
            .iter()
            .map(|&e| sig(s(src[e] as usize) + d(i)).exp())
            .sum();
        for &e in &es {
            let want = sig(s(src[e] as usize) + d(i)).exp() / z;
            assert!((alpha[e] - want).abs() < 1e-6);
        }
    }
}

#[test]
fn aggregation_keeps_a_constant_signal_on_a_ring() {
    // weights sum to one per node, so a constant feature passes unchanged
    // and a smooth one is averaged over i-1, i, i+1
    let dev = Device::Cpu;
    let g = ring(6, &dev);
    let h = Tensor::from_vec(vec![3f32; 12], (6, 2), &dev).unwrap();
    let a = Tensor::from_vec(vec![0.1f32, 0.2], (2, 1), &dev).unwrap();
    let alpha = attention(&h, &a, &a, &g).unwrap();
    let out: Vec<Vec<f32>> = aggregate(&alpha, &h, &g).unwrap().to_vec2().unwrap();
    for row in out {
        for v in row {
            assert!((v - 3.0).abs() < 1e-5);
        }
    }
}

#[test]
fn attention_gradient_stays_finite_at_very_negative_logits() {
    // edge logits -100 and -50: exp(-logit) overflows f32, where torch's
    // sigmoid has a finite (vanishing) gradient
    let dev = Device::Cpu;
    let g = EdgeIndex::new(&[0, 1, 0, 1], &[0, 0, 1, 1], 2, &dev).unwrap();
    let h = Tensor::from_vec(vec![1f32, 0.5], (2, 1), &dev).unwrap();
    let a_src = Var::from_tensor(&Tensor::from_vec(vec![-100f32], (1, 1), &dev).unwrap()).unwrap();
    let a_dst = Var::from_tensor(&Tensor::zeros((1, 1), DType::F32, &dev).unwrap()).unwrap();
    let alpha = attention(&h, a_src.as_tensor(), a_dst.as_tensor(), &g).unwrap();
    let a: Vec<f32> = alpha.flatten_all().unwrap().to_vec1().unwrap();
    for v in &a {
        assert!((v - 0.5).abs() < 1e-6, "{a:?}");
    }
    let w = Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (4, 1), &dev).unwrap();
    let grads = (alpha * w).unwrap().sum_all().unwrap().backward().unwrap();
    for p in [&a_src, &a_dst] {
        let gr: Vec<f32> = grads
            .get(p.as_tensor())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert!(gr.iter().all(|v| v.is_finite()), "{gr:?}");
    }
}

#[test]
fn aggregation_sums_messages_at_the_target_of_directed_edges() {
    // a chain 0 -> 1 -> 2 -> 3 plus self loops: node 0 hears only itself,
    // node i > 0 itself and i - 1; a direction swap would mix in i + 1
    let dev = Device::Cpu;
    let src = [0u32, 0, 1, 1, 2, 2, 3];
    let dst = [0u32, 1, 1, 2, 2, 3, 3];
    let g = EdgeIndex::new(&src, &dst, 4, &dev).unwrap();
    let hv = [1f32, 10.0, 100.0, 1000.0];
    let h = Tensor::from_vec(hv.to_vec(), (4, 1), &dev).unwrap();
    let a_src = Tensor::from_vec(vec![1e-3f32], (1, 1), &dev).unwrap();
    let a_dst = Tensor::from_vec(vec![-2e-3f32], (1, 1), &dev).unwrap();
    let alpha = attention(&h, &a_src, &a_dst, &g).unwrap();
    let av: Vec<f32> = alpha.flatten_all().unwrap().to_vec1().unwrap();
    let out: Vec<f32> = aggregate(&alpha, &h, &g)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    for (i, &o) in out.iter().enumerate() {
        let want: f32 = (0..src.len())
            .filter(|&e| dst[e] as usize == i)
            .map(|e| av[e] * hv[src[e] as usize])
            .sum();
        assert!(
            (o - want).abs() < 1e-4 * want.abs(),
            "node {i}: {o} vs {want}"
        );
    }
    assert!((out[0] - 1.0).abs() < 1e-6);
}

fn toy_model_input(dev: &Device) -> (Tensor, EdgeIndex) {
    let n = 40;
    let p = 12;
    let x: Vec<f32> = (0..n * p)
        .map(|k| {
            let (i, j) = (k / p, k % p);
            let domain = if i < n / 2 { 1.0 } else { 0.0 };
            (domain * (j % 3) as f32 + ((k * 7919) % 13) as f32 / 13.0).ln_1p()
        })
        .collect();
    (Tensor::from_vec(x, (n, p), dev).unwrap(), ring(n, dev))
}

#[test]
fn training_lowers_the_reconstruction_loss() {
    let dev = Device::Cpu;
    let (x, g) = toy_model_input(&dev);
    let mut m = Stagate::new(12, 16, 3, 7, &dev).unwrap();
    let opts = TrainOpts {
        n_epochs: 200,
        lr: 1e-3,
        weight_decay: 1e-4,
        gradient_clipping: 5.0,
    };
    let losses = m.fit(&Input::Dense(x.clone()), &g, &opts).unwrap();
    assert_eq!(losses.len(), 200);
    assert!(
        losses[199] < 0.5 * losses[0],
        "{} -> {}",
        losses[0],
        losses[199]
    );
    let z = m.forward(&x, &g).unwrap().z;
    assert_eq!(z.dims(), &[40, 3]);
}

#[test]
fn decoder_stays_tied_to_the_encoder() {
    // after a step, forward() (decoder = encoder transposed) equals an
    // explicit forward_with the transposes
    let dev = Device::Cpu;
    let (x, g) = toy_model_input(&dev);
    let mut m = Stagate::new(12, 8, 2, 3, &dev).unwrap();
    let opts = TrainOpts {
        n_epochs: 1,
        lr: 1e-2,
        weight_decay: 0.0,
        gradient_clipping: 5.0,
    };
    let w1_before: Vec<Vec<f32>> = m.w1.as_tensor().to_vec2().unwrap();
    m.step(&Input::Dense(x.clone()), &g, &opts).unwrap();
    let w1_after: Vec<Vec<f32>> = m.w1.as_tensor().to_vec2().unwrap();
    // two Adam updates of at most lr each land on W1 (its own and conv4's)
    let mut max_move = 0f32;
    for (a, b) in w1_before.iter().flatten().zip(w1_after.iter().flatten()) {
        max_move = max_move.max((a - b).abs());
    }
    assert!(max_move > 1.5e-2 && max_move <= 2.0e-2 + 1e-6, "{max_move}");
    let w3 = m.w2.as_tensor().t().unwrap().contiguous().unwrap();
    let w4 = m.w1.as_tensor().t().unwrap().contiguous().unwrap();
    let a: Vec<Vec<f32>> = m.forward(&x, &g).unwrap().recon.to_vec2().unwrap();
    let b: Vec<Vec<f32>> = m
        .forward_with(&x, &g, &w3, &w4)
        .unwrap()
        .recon
        .to_vec2()
        .unwrap();
    assert_eq!(a, b);
}

#[test]
fn cuda_forward_matches_cpu() {
    let Ok(cuda) = Device::new_cuda(0) else {
        return;
    };
    let mut out = Vec::new();
    for dev in [Device::Cpu, cuda] {
        let (x, g) = toy_model_input(&dev);
        let m = Stagate::new(12, 16, 3, 7, &dev).unwrap();
        let r: Vec<f32> = m
            .forward(&x, &g)
            .unwrap()
            .recon
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        out.push(r);
    }
    for (a, b) in out[0].iter().zip(&out[1]) {
        assert!((a - b).abs() < 1e-4 * (1.0 + a.abs()), "{a} vs {b}");
    }
}

#[test]
fn loess_keeps_degree_two_in_a_narrow_window() {
    // means within 0.01 decades: the local quadratic must still be exact
    let x: Vec<f64> = (0..300).map(|i| 1e-2 * (i as f64 * 0.37).sin()).collect();
    let y: Vec<f64> = x.iter().map(|v| 1.0 + 2.0 * v - 50.0 * v * v).collect();
    let fit = loess_quadratic(&x, &y, 0.3);
    for (f, t) in fit.iter().zip(&y) {
        assert!((f - t).abs() < 1e-9, "{f} vs {t}");
    }
}

#[test]
fn filter_genes_keeps_genes_in_enough_cells() {
    // gene 0 in 3 cells, gene 1 in 1, gene 2 in 2
    let mut coo = CooMatrix::<f32>::new(3, 4);
    for c in 0..3 {
        coo.push(0, c, 1.0);
    }
    coo.push(1, 3, 5.0);
    coo.push(2, 0, 2.0);
    coo.push(2, 1, 2.0);
    let counts = CscMatrix::from(&coo);
    let kept = filter_genes(counts.clone(), 2);
    assert_eq!((kept.nrows(), kept.ncols()), (2, 4));
    let t: Vec<(usize, usize, f32)> = kept.triplet_iter().map(|(r, c, &v)| (r, c, v)).collect();
    // genes 0 and 2 renumbered 0 and 1, by cell then gene
    assert_eq!(
        t,
        [
            (0, 0, 1.0),
            (1, 0, 2.0),
            (0, 1, 1.0),
            (1, 1, 2.0),
            (0, 2, 1.0)
        ]
    );
    assert_eq!(filter_genes(counts.clone(), 0).nnz(), counts.nnz());
}

/// `n` points in `d` dimensions around `k` well-separated centres, in
/// order of their centre.
fn blobs(n: usize, d: usize, k: usize) -> nalgebra::DMatrix<f64> {
    let mut state = 99u64;
    let mut unif = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
    };
    nalgebra::DMatrix::from_fn(n, d, |i, j| {
        let c = i * k / n;
        10.0 * ((c + j) % k) as f64 + unif()
    })
}

/// The embedding of `blobs`, in f32.
fn embedding(n: usize, d: usize, k: usize) -> Mat {
    let x = blobs(n, d, k);
    Mat::from_fn(n, d, |i, j| x[(i, j)] as f32)
}

/// The labels used, in order.
fn distinct(labels: &[usize]) -> Vec<usize> {
    let mut seen = labels.to_vec();
    seen.sort_unstable();
    seen.dedup();
    seen
}

/// Every group of `blobs(n, _, k)` under one label, and how many labels.
fn labels_follow_groups(labels: &[usize], n: usize, k: usize) -> usize {
    assert_eq!(labels.len(), n);
    let mut of_blob = vec![None; k];
    for (i, &l) in labels.iter().enumerate() {
        assert_eq!(*of_blob[i * k / n].get_or_insert(l), l, "{labels:?}");
    }
    distinct(labels).len()
}

#[test]
fn n_clusters_gives_exactly_that_many_kmeans_clusters() {
    let (n, k) = (120, 3);
    let z = embedding(n, 5, k);
    let (labels, method) = cluster_embedding(&z, Some(k), 0.5, 7).unwrap();
    assert_eq!(method, "kmeans");
    assert_eq!(labels_follow_groups(&labels, n, k), k);
    assert_eq!(distinct(&labels), (0..k).collect::<Vec<_>>());
    // one seed, one answer
    assert_eq!(cluster_embedding(&z, Some(k), 0.5, 7).unwrap().0, labels);
    // the count is taken as given, even past the groups the data has
    let (more, _) = cluster_embedding(&z, Some(k + 1), 0.5, 7).unwrap();
    assert_eq!(distinct(&more), (0..=k).collect::<Vec<_>>());
}

#[test]
fn without_n_clusters_leiden_finds_the_groups() {
    let (n, k) = (120, 3);
    let z = embedding(n, 5, k);
    let (labels, method) = cluster_embedding(&z, None, 0.5, 7).unwrap();
    assert_eq!(method, "leiden");
    assert_eq!(labels_follow_groups(&labels, n, k), k);
    assert_eq!(cluster_embedding(&z, None, 0.5, 7).unwrap().0, labels);
}

#[test]
fn stagate_help_names_its_clusterers() {
    let help = crate::built_cli()
        .find_subcommand_mut("stagate")
        .expect("stagate")
        .render_long_help()
        .to_string();
    for want in ["k-means", "Leiden"] {
        assert!(help.contains(want), "stagate --help lacks {want}");
    }
}

/// One forward pass and 20 training steps of a torch replica of
/// STAGATE_pyG (`STAGATE.py`, `gat_conv.py`, `Train_STAGATE.py` at the
/// commit named in the module doc: the `.data =` weight tie, torch's
/// multi-tensor Adam with weight decay, `clip_grad_norm_`), float32, on
/// synthetic counts with a 3-NN graph that is not symmetric, plus self
/// loops. Weight decay and clipping are set high enough to act on every
/// step.
#[derive(serde::Deserialize)]
struct Parity {
    n: usize,
    p: usize,
    hidden: usize,
    latent: usize,
    lr: f64,
    weight_decay: f64,
    gradient_clipping: f64,
    src: Vec<u32>,
    dst: Vec<u32>,
    x: Vec<f32>,
    w1: Vec<f32>,
    a_src: Vec<f32>,
    a_dst: Vec<f32>,
    w2: Vec<f32>,
    z_init: Vec<f32>,
    recon_init: Vec<f32>,
    losses: Vec<f32>,
    z_final: Vec<f32>,
    recon_final: Vec<f32>,
}

fn flat(t: &Tensor) -> Vec<f32> {
    t.flatten_all().unwrap().to_vec1().unwrap()
}

fn assert_close(what: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert!(
            (a - b).abs() <= tol * (1.0 + b.abs()),
            "{what}[{i}]: {a} vs {b}"
        );
    }
}

#[test]
fn model_and_training_match_a_torch_replica() {
    let f: Parity = serde_json::from_str(include_str!("stagate_parity.json")).unwrap();
    let dev = Device::Cpu;
    let g = EdgeIndex::new(&f.src, &f.dst, f.n, &dev).unwrap();
    let x = Tensor::from_vec(f.x.clone(), (f.n, f.p), &dev).unwrap();
    let mut m = Stagate::new(f.p, f.hidden, f.latent, 0, &dev).unwrap();
    let set = |v: &Var, w: &[f32], r: usize, c: usize| {
        v.set(&Tensor::from_vec(w.to_vec(), (r, c), &dev).unwrap())
            .unwrap()
    };
    set(&m.w1, &f.w1, f.p, f.hidden);
    set(&m.a_src, &f.a_src, f.hidden, 1);
    set(&m.a_dst, &f.a_dst, f.hidden, 1);
    set(&m.w2, &f.w2, f.hidden, f.latent);
    let out = m.forward(&x, &g).unwrap();
    assert_close("z at init", &flat(&out.z), &f.z_init, 1e-5);
    assert_close("recon at init", &flat(&out.recon), &f.recon_init, 1e-5);
    let opts = TrainOpts {
        n_epochs: f.losses.len(),
        lr: f.lr,
        weight_decay: f.weight_decay,
        gradient_clipping: f.gradient_clipping,
    };
    let losses = m.fit(&Input::Dense(x.clone()), &g, &opts).unwrap();
    // looser after training: Adam's early steps are about lr · sign(g), so
    // rounding that differs between matmul backends moves the weights
    // more than it moves one forward pass
    assert_close("losses", &losses, &f.losses, 1e-4);
    let out = m.forward(&x, &g).unwrap();
    assert_close("z after training", &flat(&out.z), &f.z_final, 1e-4);
    assert_close(
        "recon after training",
        &flat(&out.recon),
        &f.recon_final,
        1e-4,
    );
}

/// The toy input with some entries zero, as a sparse input (genes × cells
/// log data, every gene kept) whose loss runs in blocks of `block` cells.
fn toy_sparse(x: &Tensor, block: usize) -> (Tensor, SparseInput) {
    let rows: Vec<Vec<f32>> = x.to_vec2().unwrap();
    let (n, p) = (rows.len(), rows[0].len());
    let mut coo = CooMatrix::<f32>::new(p, n);
    let mut dense = vec![0f32; n * p];
    for (i, r) in rows.iter().enumerate() {
        for (j, &v) in r.iter().enumerate() {
            if (i * 5 + j * 3) % 4 != 0 {
                coo.push(j, i, v);
                dense[i * p + j] = v;
            }
        }
    }
    let mut sparse = SparseInput::from_rows(&CscMatrix::from(&coo), &(0..p).collect::<Vec<_>>());
    sparse.block_cells = block;
    let dense = Tensor::from_vec(dense, (n, p), x.device()).unwrap();
    (dense, sparse)
}

/// The sparse products and blocks are the dense input's.
#[test]
fn sparse_input_products_are_the_dense_ones() {
    let dev = Device::Cpu;
    let (x, _) = toy_model_input(&dev);
    let (dense, sparse) = toy_sparse(&x, 7);
    let (n, p, h) = (40, 12, 5);
    let w: Vec<f32> = (0..p * h).map(|k| (k as f32 * 0.37).sin()).collect();
    let d: Vec<f32> = (0..n * h).map(|k| (k as f32 * 0.11).cos()).collect();
    let wt = Tensor::from_vec(w.clone(), (p, h), &dev).unwrap();
    let dt = Tensor::from_vec(d.clone(), (n, h), &dev).unwrap();
    assert_close(
        "x W",
        &sparse.matmul(&w, h),
        &flat(&dense.matmul(&wt).unwrap()),
        1e-6,
    );
    let xt_d = dense.t().unwrap().matmul(&dt).unwrap();
    assert_close("xᵀ D", &sparse.t_matmul(&d, h), &flat(&xt_d), 1e-6);
    let rows = dense.narrow(0, 7, 14).unwrap();
    assert_eq!(sparse.dense_block(7..21), flat(&rows));
}

/// The sparse formulation (the default) trains to the same numbers as the
/// dense one (the GPU layout): every epoch's loss and the embedding, within
/// float rounding, with the loss summed over uneven blocks of cells.
#[test]
fn sparse_training_matches_the_dense_layout() {
    let dev = Device::Cpu;
    let (x, g) = toy_model_input(&dev);
    let (dense, sparse) = toy_sparse(&x, 7);
    let opts = TrainOpts {
        n_epochs: 40,
        lr: 1e-2,
        weight_decay: 1e-4,
        gradient_clipping: 5.0,
    };
    let (dense, sparse) = (Input::Dense(dense), Input::Sparse(sparse));
    let mut a = Stagate::new(12, 16, 3, 7, &dev).unwrap();
    let mut b = Stagate::new(12, 16, 3, 7, &dev).unwrap();
    let la = a.fit(&dense, &g, &opts).unwrap();
    let lb = b.fit(&sparse, &g, &opts).unwrap();
    assert_close("losses", &lb, &la, 1e-5);
    let za = flat(&a.embed(&dense, &g).unwrap());
    let zb = flat(&b.embed(&sparse, &g).unwrap());
    assert_close("embedding", &zb, &za, 1e-4);
    assert_close("W1", &flat(b.w1.as_tensor()), &flat(a.w1.as_tensor()), 1e-4);
}

/// The sparse formulation matches the torch replica too.
#[test]
fn sparse_training_matches_the_torch_replica() {
    let f: Parity = serde_json::from_str(include_str!("stagate_parity.json")).unwrap();
    let dev = Device::Cpu;
    let g = EdgeIndex::new(&f.src, &f.dst, f.n, &dev).unwrap();
    let mut coo = CooMatrix::<f32>::new(f.p, f.n);
    for (k, &v) in f.x.iter().enumerate() {
        if v != 0.0 {
            coo.push(k % f.p, k / f.p, v);
        }
    }
    let mut x = SparseInput::from_rows(&CscMatrix::from(&coo), &(0..f.p).collect::<Vec<_>>());
    x.block_cells = 3;
    let x = Input::Sparse(x);
    let mut m = Stagate::new(f.p, f.hidden, f.latent, 0, &dev).unwrap();
    let set = |v: &Var, w: &[f32], r: usize, c: usize| {
        v.set(&Tensor::from_vec(w.to_vec(), (r, c), &dev).unwrap())
            .unwrap()
    };
    set(&m.w1, &f.w1, f.p, f.hidden);
    set(&m.a_src, &f.a_src, f.hidden, 1);
    set(&m.a_dst, &f.a_dst, f.hidden, 1);
    set(&m.w2, &f.w2, f.hidden, f.latent);
    assert_close(
        "z at init",
        &flat(&m.embed(&x, &g).unwrap()),
        &f.z_init,
        1e-5,
    );
    let opts = TrainOpts {
        n_epochs: f.losses.len(),
        lr: f.lr,
        weight_decay: f.weight_decay,
        gradient_clipping: f.gradient_clipping,
    };
    let losses = m.fit(&x, &g, &opts).unwrap();
    assert_close("losses", &losses, &f.losses, 1e-4);
    assert_close(
        "z after training",
        &flat(&m.embed(&x, &g).unwrap()),
        &f.z_final,
        1e-4,
    );
}

/// A step's loss holds no graph: training keeps a hundred of them before
/// reading them, and a traced one would keep its step's tensors alive.
#[test]
fn a_step_s_loss_is_detached() {
    let dev = Device::Cpu;
    let (x, g) = toy_model_input(&dev);
    let (dense, sparse) = toy_sparse(&x, 7);
    let opts = TrainOpts {
        n_epochs: 1,
        lr: 1e-3,
        weight_decay: 0.0,
        gradient_clipping: 5.0,
    };
    for input in [Input::Dense(dense), Input::Sparse(sparse)] {
        let mut m = Stagate::new(12, 8, 2, 3, &dev).unwrap();
        assert!(!m.step(&input, &g, &opts).unwrap().track_op());
    }
}

/// Inertia of `labels` on `z`: squared distances to each cluster's mean.
fn inertia(z: &Mat, labels: &[usize]) -> f64 {
    let k = labels.iter().max().map_or(0, |&m| m + 1);
    let mut sum = vec![vec![0f64; z.ncols()]; k];
    let mut count = vec![0f64; k];
    for (i, &l) in labels.iter().enumerate() {
        count[l] += 1.0;
        for j in 0..z.ncols() {
            sum[l][j] += f64::from(z[(i, j)]);
        }
    }
    (0..z.nrows())
        .map(|i| {
            let l = labels[i];
            (0..z.ncols())
                .map(|j| (f64::from(z[(i, j)]) - sum[l][j] / count[l]).powi(2))
                .sum::<f64>()
        })
        .sum()
}

/// The restarts keep the best start: their inertia is at most that of
/// each of their single starts, and below the worst on a cloud where
/// starts disagree.
#[test]
fn kmeans_restarts_keep_the_lowest_inertia_start() {
    use legume_numeric::matrix::kmeans::kmeans_centroids_seeded;
    // an unstructured cloud, so k-means++ starts land in different optima
    let (n, k) = (300, 8);
    let g = crate::spatial_baselines::spatial_factor::kit::gaussian(n, 2, 5);
    let z = Mat::from_fn(n, 2, |i, j| g[(i, j)] as f32);
    let mut worst = 0f64;
    for seed in [3u64, 11, 40] {
        let best = inertia(&z, &kmeans_restarts(&z, k, seed));
        for r in 0..KMEANS_RESTARTS as u64 {
            let (_, l) = kmeans_centroids_seeded(&z, k, KMEANS_MAX_ITER, seed + r);
            let single = inertia(&z, &l);
            assert!(best <= single * (1.0 + 1e-6), "seed {seed} start {r}");
            worst = worst.max(single);
        }
        assert_eq!(kmeans_restarts(&z, k, seed), kmeans_restarts(&z, k, seed));
        assert!(best < worst, "seed {seed}: starts never differ");
    }
}
