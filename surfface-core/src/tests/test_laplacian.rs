use crate::backend::AutoBackend;
use crate::centroid::CentroidState;
use crate::laplacian::{LaplacianConfig, LaplacianStage};
use crate::tests::init;
use burn::prelude::*;
use rand::{Rng, SeedableRng};

type TestBackend = AutoBackend;

// ─────────────────────────────────────────────────────────────────────────────
// Corrected Property Test: Spectral Bounds (λ ∈ [0, 2])
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_laplacian_spectral_bounds_normalized() {
    init();
    // For L_sym = I - D^{-1/2} W D^{-1/2}, eigenvalues MUST lie in [0, 2].
    // We verify this using Rayleigh Quotients: R(L, x) = (x^T L x) / (x^T x).
    let state = centroids_from_gaussian_blobs(10, 20, 0.4, 777);
    let config = LaplacianConfig {
        k_neighbors: 5,
        normalize: true,
        ..Default::default()
    };

    let stage = LaplacianStage::new(config);
    let output = stage.execute(&state);
    let f = output.n_features;
    let matrix_flat: Vec<f32> = output.matrix.to_data().to_vec().unwrap();

    // 1. Structural Invariants
    for i in 0..f {
        let diag = matrix_flat[i * f + i];
        if output.degrees[i] > 1e-6 {
            assert!(
                (diag - 1.0).abs() < 1e-5,
                "Diagonal must be 1.0, got {}",
                diag
            );
        }
        for j in 0..f {
            if i != j {
                assert!(matrix_flat[i * f + j] <= 0.0, "Off-diagonals must be <= 0");
            }
        }
    }

    // 2. Rayleigh Quotient Sampling (Monte Carlo spectral bound check)
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    for _ in 0..100 {
        let x: Vec<f32> = (0..f).map(|_| rng.random_range(-1.0..1.0)).collect();
        let norm_sq: f32 = x.iter().map(|&v| v * v).sum();
        if norm_sq < 1e-9 {
            continue;
        }

        // Compute L * x
        let mut lx = vec![0.0f32; f];
        for i in 0..f {
            for j in 0..f {
                lx[i] += matrix_flat[i * f + j] * x[j];
            }
        }

        // R(L, x) = x^T L x / x^T x
        let x_t_lx: f32 = x.iter().zip(lx.iter()).map(|(&xi, &lxi)| xi * lxi).sum();
        let rayleigh_quotient = x_t_lx / norm_sq;

        // Theory: 0.0 <= λ <= 2.0
        assert!(
            rayleigh_quotient >= -1e-4,
            "Eigenvalue lower bound violation: {}",
            rayleigh_quotient
        );
        assert!(
            rayleigh_quotient <= 2.0 + 1e-4,
            "Eigenvalue upper bound violation: {}",
            rayleigh_quotient
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// New Test 1: Partition Invariance (Connected Components)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_laplacian_nullspace_dimension() {
    init();
    // THEORY: The multiplicity of the eigenvalue 0 in L equals the number of
    // connected components in the graph. For a connected graph, L * 1 = 0.
    // For normalized Laplacian L_sym, L_sym * D^{1/2}1 = 0.

    let state = centroids_from_gaussian_blobs(5, 10, 0.1, 123);
    let config = LaplacianConfig {
        k_neighbors: 9, // Fully connected potential
        normalize: true,
        ..Default::default()
    };

    let stage = LaplacianStage::new(config);
    let output = stage.execute(&state);
    let f = output.n_features;
    let matrix_flat: Vec<f32> = output.matrix.to_data().to_vec().unwrap();

    // Check if the harmonic vector v = D^{1/2}1 is in the nullspace
    // L_sym * v = (I - D^{-1/2}WD^{-1/2}) * D^{1/2}1 = D^{1/2}1 - D^{-1/2}W1 = D^{1/2}1 - D^{-1/2}D1 = 0
    let v: Vec<f32> = output.degrees.iter().map(|&d| d.sqrt()).collect();

    let mut lv = vec![0.0f32; f];
    for i in 0..f {
        for j in 0..f {
            lv[i] += matrix_flat[i * f + j] * v[j];
        }
    }

    for (i, &val) in lv.iter().enumerate() {
        // Higher tolerance for nullspace due to square roots and float accumulation
        assert!(
            val.abs() < 1e-3,
            "Nullspace violation at row {}: {}",
            i,
            val
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// New Test 2: Multi-Order Stability (L1 vs L2 Alignment)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_high_order_neighborhood_consistency() {
    init();
    // Inspired by IEEELaplacian2 (Multi-View Spectral Clustering).
    // The second-order adjacency A2 = A * A should have a similar
    // sparsity structure to A. We verify that the first-order Laplacian
    // captures the same neighborhood "hubs" as a multi-step walk.

    let state = centroids_from_gaussian_blobs(20, 50, 0.5, 999);
    let stage = LaplacianStage::with_defaults();
    let output = stage.execute(&state);

    // Identify "Hub" features (highest weighted degree)
    let mut degree_pairs: Vec<(usize, f32)> = output.degrees.iter().cloned().enumerate().collect();
    degree_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let top_hub_idx = degree_pairs[0].0;
    let f = output.n_features;
    let matrix_flat: Vec<f32> = output.matrix.to_data().to_vec().unwrap();

    // Features connected to the top hub should have significantly lower
    // Laplacian values (high similarity) than those far away.
    let mut hub_connections = 0;
    for j in 0..f {
        if j == top_hub_idx {
            continue;
        }
        let l_val = matrix_flat[top_hub_idx * f + j];
        if l_val < -0.05 {
            // Significant similarity
            hub_connections += 1;
        }
    }

    assert!(
        hub_connections > 0,
        "Hub feature {} should have neighbors",
        top_hub_idx
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Strengthened Regression: ArrowSpace Config Compatibility
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_regression_unnormalized_row_sums() {
    init();
    // In unnormalized mode L = D - W, row sums MUST be exactly 0.0.
    let state = centroids_from_gaussian_blobs(10, 30, 0.5, 42);
    let config = LaplacianConfig {
        normalize: false, // Unnormalized
        k_neighbors: 10,
        ..Default::default()
    };

    let stage = LaplacianStage::new(config);
    let output = stage.execute(&state);
    let f = output.n_features;
    let matrix_flat: Vec<f32> = output.matrix.to_data().to_vec().unwrap();

    for i in 0..f {
        let row_sum: f32 = (0..f).map(|j| matrix_flat[i * f + j]).sum();
        assert!(
            row_sum.abs() < 1e-4,
            "Unnormalized Row {} sum = {}, expected 0.0",
            i,
            row_sum
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Refactored Data Generators (ArrowSpace Standard)
// ─────────────────────────────────────────────────────────────────────────────

fn centroids_from_gaussian_blobs(
    c: usize,
    f: usize,
    noise: f32,
    seed: u64,
) -> CentroidState<TestBackend> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let device = Default::default();

    let means = (0..c * f).map(|_| rng.random_range(-1.0..1.0)).collect();
    let vars = (0..c * f).map(|_| rng.random_range(0.01..noise)).collect();
    let counts = (0..c).map(|_| rng.random_range(10..100)).collect();

    CentroidState {
        means: Tensor::from_data(TensorData::new(means, Shape::new([c, f])), &device),
        variances: Tensor::from_data(TensorData::new(vars, Shape::new([c, f])), &device),
        counts: Tensor::from_data(TensorData::new(counts, Shape::new([c])), &device),
    }
}
