use super::graph::toy;
use crate::spatial_baselines::graph::knn_within_sections;
use crate::spatial_baselines::stagate::*;
use crate::util::common::Mat;
use legume_numeric::candle::candle_core::{Device, Tensor};
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
    let losses = m.fit(&x, &g, &opts).unwrap();
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
    m.step(&x, &g, &opts).unwrap();
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
fn eee_posteriors_sum_to_one_and_find_two_blobs() {
    let n = 200;
    let z = Mat::from_fn(n, 2, |i, j| {
        let centre = if i < n / 2 { 0.0 } else { 6.0 };
        centre + (((i * 31 + j * 17) % 23) as f32 / 23.0 - 0.5)
    });
    let (labels, post) = eee_gmm(&z, 2, 1).unwrap();
    for i in 0..n {
        let s: f32 = post.row(i).iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
    }
    let first = labels[0];
    assert!(labels[..n / 2].iter().all(|&l| l == first));
    assert!(labels[n / 2..].iter().all(|&l| l != first));
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
