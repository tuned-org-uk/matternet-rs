use crate::backend::AutoBackend;
use crate::centroid::CentroidState;
use crate::mst::{Edge, MSTOutput};
use crate::smoothing_chain::{SmoothingConfig, SmoothingStage, TransitionModel};
use burn::tensor::{Int, Tensor, backend::Backend};

type TestBackend = AutoBackend;
/// Initialize logging for tests
fn init() {
    let _ = env_logger::builder()
        .filter_level(log::LevelFilter::Debug)
        .is_test(true)
        .try_init();
}

/// Create synthetic noisy centroid state
fn create_noisy_centroids(c: usize, f: usize, noise_level: f32) -> CentroidState<TestBackend> {
    let device = Default::default();

    // Create smooth underlying signal (sine wave)
    let mut means = Vec::new();
    for i in 0..c {
        for j in 0..f {
            let base = (i as f32 * 0.5 + j as f32 * 0.1).sin();
            let noise = (i * 7 + j * 13) as f32 % 100.0 / 100.0 - 0.5; // Deterministic noise
            means.push(base + noise * noise_level);
        }
    }

    // Variances proportional to noise
    let variances = vec![noise_level * noise_level; c * f];
    let counts = vec![10; c]; // All centroids have same count

    CentroidState {
        means: Tensor::<TestBackend, 2>::from_data(
            burn::tensor::TensorData::new(means, burn::tensor::Shape::new([c, f])),
            &device,
        ),
        variances: Tensor::<TestBackend, 2>::from_data(
            burn::tensor::TensorData::new(variances, burn::tensor::Shape::new([c, f])),
            &device,
        ),
        counts: Tensor::<TestBackend, 1, Int>::from_data(
            burn::tensor::TensorData::new(counts, burn::tensor::Shape::new([c])),
            &device,
        ),
    }
}

/// Create simple linear MST order (1D chain)
fn create_linear_mst(c: usize) -> MSTOutput {
    let order: Vec<usize> = (0..c).collect();

    // Create dummy edges for chain topology
    let mut edges = Vec::new();
    for i in 0..c - 1 {
        edges.push(Edge {
            u: i,
            v: i + 1,
            distance: 1.0,
            thickness_u: 1.0,
            thickness_v: 1.0,
            cost: 1.0,
        });
    }

    MSTOutput {
        candidate_edges: edges.clone(),
        mst_edges: edges,
        centroid_order: order,
        trunk_nodes: vec![],
        thickness: vec![1.0; c],
        total_weight: (c - 1) as f32,
        nodes_in_mst: c,
    }
}

#[test]
fn test_kalman_reduces_variance() {
    init();
    let c = 10;
    let f = 3;
    let noise_level = 0.5;

    let noisy_state = create_noisy_centroids(c, f, noise_level);
    let mst_output = create_linear_mst(c);

    let kalman = SmoothingStage::new(SmoothingConfig::default());
    let output = kalman.execute(&noisy_state, &mst_output);

    println!(
        "Variance reduction: {:.2}%",
        output.variance_reduction * 100.0
    );

    // Test: Smoothing should reduce variance
    assert!(
        output.variance_reduction > 0.0,
        "Smoothing should reduce variance"
    );
    assert!(
        output.variance_reduction < 1.0,
        "Variance reduction should be < 100%"
    );

    // Test: Smoothed variance should be lower than raw
    let raw_vars: Vec<f32> = noisy_state.variances.to_data().to_vec().unwrap();
    let smooth_vars: Vec<f32> = output.smoothed_variances.to_data().to_vec().unwrap();

    let raw_mean = raw_vars.iter().sum::<f32>() / raw_vars.len() as f32;
    let smooth_mean = smooth_vars.iter().sum::<f32>() / smooth_vars.len() as f32;

    assert!(
        smooth_mean < raw_mean,
        "Smoothed variance {:.4} should be < raw {:.4}",
        smooth_mean,
        raw_mean
    );
}

#[test]
fn test_kalman_preserves_counts() {
    init();
    let c = 5;
    let f = 2;

    let noisy_state = create_noisy_centroids(c, f, 0.3);
    let mst_output = create_linear_mst(c);

    let kalman = SmoothingStage::new(SmoothingConfig::default());
    let output = kalman.execute(&noisy_state, &mst_output);

    // Test: Counts should be preserved exactly
    let input_counts: Vec<i64> = noisy_state.counts.to_data().to_vec().unwrap();
    let output_counts: Vec<i64> = output.counts.to_data().to_vec().unwrap();

    assert_eq!(
        input_counts, output_counts,
        "Counts should be preserved during smoothing"
    );
}

#[test]
fn test_kalman_smoothness_property() {
    init();
    let c = 20;
    let f = 1;

    // Create very noisy data
    let noisy_state = create_noisy_centroids(c, f, 1.0);
    let mst_output = create_linear_mst(c);

    let kalman = SmoothingStage::new(SmoothingConfig::aggressive()); // Heavy smoothing
    let output = kalman.execute(&noisy_state, &mst_output);

    // Compute total variation (roughness metric)
    let raw_means: Vec<f32> = noisy_state.means.to_data().to_vec().unwrap();
    let smooth_means: Vec<f32> = output.smoothed_means.to_data().to_vec().unwrap();

    let raw_tv: f32 = (0..c - 1)
        .map(|i| (raw_means[i + 1] - raw_means[i]).abs())
        .sum();
    let smooth_tv: f32 = (0..c - 1)
        .map(|i| (smooth_means[i + 1] - smooth_means[i]).abs())
        .sum();

    println!("Raw total variation: {:.4}", raw_tv);
    println!("Smoothed total variation: {:.4}", smooth_tv);

    // Test: Smoothed signal should have lower total variation (smoother)
    assert!(
        smooth_tv < raw_tv,
        "Smoothed trajectory should be smoother (lower TV)"
    );
}

#[test]
fn test_kalman_conservative_vs_aggressive() {
    init();
    let c = 10;
    let f = 2;

    let noisy_state = create_noisy_centroids(c, f, 0.5);
    let mst_output = create_linear_mst(c);

    // Conservative: trust observations
    let conservative = SmoothingStage::new(SmoothingConfig::conservative());
    let output_cons = conservative.execute(&noisy_state, &mst_output);

    // Aggressive: smooth heavily
    let aggressive = SmoothingStage::new(SmoothingConfig::aggressive());
    let output_agg = aggressive.execute(&noisy_state, &mst_output);

    println!(
        "Conservative variance reduction: {:.2}%",
        output_cons.variance_reduction * 100.0
    );
    println!(
        "Aggressive variance reduction: {:.2}%",
        output_agg.variance_reduction * 100.0
    );

    // Compare smoothness via total variation (better metric)
    let cons_means: Vec<f32> = output_cons.smoothed_means.to_data().to_vec().unwrap();
    let agg_means: Vec<f32> = output_agg.smoothed_means.to_data().to_vec().unwrap();

    // Compute total variation (sum of differences along MST order)
    let calc_tv = |data: &[f32]| -> f32 {
        data.chunks_exact(f)
            .collect::<Vec<_>>()
            .windows(2)
            .flat_map(|w| w[0].iter().zip(w[1]).map(|(a, b)| (a - b).abs()))
            .sum()
    };

    let cons_tv = calc_tv(&cons_means);
    let agg_tv = calc_tv(&agg_means);

    println!("Conservative total variation: {:.4}", cons_tv);
    println!("Aggressive total variation: {:.4}", agg_tv);

    // Test: Aggressive should produce smoother trajectory (lower total variation)
    assert!(
        agg_tv < cons_tv,
        "Aggressive smoothing (TV={:.4}) should be smoother than conservative (TV={:.4})",
        agg_tv,
        cons_tv
    );

    // Test: Aggressive should have higher smoothing gains
    let cons_gain_mean =
        output_cons.smoothing_gains.iter().sum::<f32>() / output_cons.smoothing_gains.len() as f32;
    let agg_gain_mean =
        output_agg.smoothing_gains.iter().sum::<f32>() / output_agg.smoothing_gains.len() as f32;

    println!("Conservative mean gain: {:.4}", cons_gain_mean);
    println!("Aggressive mean gain: {:.4}", agg_gain_mean);

    assert!(
        agg_gain_mean > cons_gain_mean,
        "Aggressive should have higher smoothing gains"
    );
}

#[test]
fn test_kalman_single_centroid() {
    init();
    let c = 1;
    let f = 3;

    let state = create_noisy_centroids(c, f, 0.2);
    let mst_output = MSTOutput {
        candidate_edges: vec![],
        mst_edges: vec![],
        centroid_order: vec![0],
        trunk_nodes: vec![],
        thickness: vec![1.0],
        total_weight: 0.0,
        nodes_in_mst: 1,
    };

    let kalman = SmoothingStage::new(SmoothingConfig::default());
    let output = kalman.execute(&state, &mst_output);

    // Test: Single centroid should remain unchanged (no neighbors to smooth with)
    let input_means: Vec<f32> = state.means.to_data().to_vec().unwrap();
    let output_means: Vec<f32> = output.smoothed_means.to_data().to_vec().unwrap();

    for (i, (inp, out)) in input_means.iter().zip(output_means.iter()).enumerate() {
        assert!(
            (inp - out).abs() < 1e-5,
            "Single centroid means should be unchanged at index {}",
            i
        );
    }
}

#[test]
fn test_kalman_numerical_stability() {
    init();
    let c = 5;
    let f = 2;
    let device = Default::default();

    // Create centroids with extreme variances
    let means = vec![1.0; c * f];
    let variances = vec![0.0, 1e10, 1e-10, 1.0, 1e5]; // Extreme values per centroid
    let variances_expanded = variances
        .iter()
        .flat_map(|&v| vec![v; f])
        .collect::<Vec<_>>();
    let counts = vec![5; c];

    let state = CentroidState {
        means: Tensor::<TestBackend, 2>::from_data(
            burn::tensor::TensorData::new(means, burn::tensor::Shape::new([c, f])),
            &device,
        ),
        variances: Tensor::<TestBackend, 2>::from_data(
            burn::tensor::TensorData::new(variances_expanded, burn::tensor::Shape::new([c, f])),
            &device,
        ),
        counts: Tensor::<TestBackend, 1, Int>::from_data(
            burn::tensor::TensorData::new(counts, burn::tensor::Shape::new([c])),
            &device,
        ),
    };

    let mst_output = create_linear_mst(c);

    let kalman = SmoothingStage::new(SmoothingConfig::default());
    let output = kalman.execute(&state, &mst_output);

    // Test: No NaN or Inf values in output
    let smooth_means: Vec<f32> = output.smoothed_means.to_data().to_vec().unwrap();
    let smooth_vars: Vec<f32> = output.smoothed_variances.to_data().to_vec().unwrap();

    for (i, val) in smooth_means.iter().enumerate() {
        assert!(val.is_finite(), "Smoothed mean at {} is not finite", i);
    }

    for (i, val) in smooth_vars.iter().enumerate() {
        assert!(val.is_finite(), "Smoothed variance at {} is not finite", i);
        assert!(*val >= 0.0, "Smoothed variance at {} is negative", i);
    }

    // Test: Variances should be clamped to reasonable range
    let config = SmoothingConfig::default();
    for (i, val) in smooth_vars.iter().enumerate() {
        assert!(
            *val >= config.variance_floor,
            "Variance {} below floor: {}",
            i,
            val
        );
        assert!(
            *val <= config.variance_ceiling,
            "Variance {} above ceiling: {}",
            i,
            val
        );
    }
}

#[test]
fn test_kalman_to_centroid_state() {
    init();
    let c = 5;
    let f = 3;

    let noisy_state = create_noisy_centroids(c, f, 0.3);
    let mst_output = create_linear_mst(c);

    let kalman = SmoothingStage::new(SmoothingConfig::default());
    let output = kalman.execute(&noisy_state, &mst_output);

    // Test: Conversion to CentroidState preserves structure
    let smoothed_state = output.to_centroid_state();

    assert_eq!(
        smoothed_state.num_centroids(),
        c,
        "Number of centroids should be preserved"
    );
    assert_eq!(
        smoothed_state.feature_dim(),
        f,
        "Feature dimension should be preserved"
    );

    // Test: Counts are preserved
    let original_counts: Vec<i64> = noisy_state.counts.to_data().to_vec().unwrap();
    let smoothed_counts: Vec<i64> = smoothed_state.counts.to_data().to_vec().unwrap();
    assert_eq!(
        original_counts, smoothed_counts,
        "Counts should be preserved"
    );
}

#[test]
fn test_kalman_forward_backward_consistency() {
    init();
    let c = 8;
    let f = 2;

    let noisy_state = create_noisy_centroids(c, f, 0.4);
    let mst_output = create_linear_mst(c);

    let kalman = SmoothingStage::new(SmoothingConfig::default());
    let output = kalman.execute(&noisy_state, &mst_output);

    // Test 1: Smoothed variances should be <= filtered variances
    // (backward pass can only reduce or maintain uncertainty)
    let filtered_vars: Vec<f32> = output.filtered_variances.to_data().to_vec().unwrap();
    let smoothed_vars: Vec<f32> = output.smoothed_variances.to_data().to_vec().unwrap();

    for (i, (filt, smooth)) in filtered_vars.iter().zip(smoothed_vars.iter()).enumerate() {
        assert!(
            smooth <= filt,
            "Smoothed variance {} should be <= filtered variance {} at index {}",
            smooth,
            filt,
            i
        );
    }

    // Test 2: Smoothed estimates should be within reasonable bounds
    let raw_means: Vec<f32> = noisy_state.means.to_data().to_vec().unwrap();
    let filtered_means: Vec<f32> = output.filtered_means.to_data().to_vec().unwrap();
    let smoothed_means: Vec<f32> = output.smoothed_means.to_data().to_vec().unwrap();

    // Compute bounds: min/max of raw and filtered
    for i in 0..c * f {
        let min_bound = raw_means[i].min(filtered_means[i]) - 1.0; // Add tolerance
        let max_bound = raw_means[i].max(filtered_means[i]) + 1.0;

        assert!(
            smoothed_means[i] >= min_bound && smoothed_means[i] <= max_bound,
            "Smoothed mean {:.4} should be within reasonable bounds [{:.4}, {:.4}] at index {}",
            smoothed_means[i],
            min_bound,
            max_bound,
            i
        );
    }

    // Test 3: RTS smoothing should produce more consistent estimates
    // Measure: standard deviation of differences should be lower for smoothed
    let mut filt_diffs = Vec::new();
    let mut smooth_diffs = Vec::new();

    for i in 0..c - 1 {
        for j in 0..f {
            let idx = i * f + j;
            let next_idx = (i + 1) * f + j;

            filt_diffs.push(filtered_means[next_idx] - filtered_means[idx]);
            smooth_diffs.push(smoothed_means[next_idx] - smoothed_means[idx]);
        }
    }

    let filt_std = {
        let mean = filt_diffs.iter().sum::<f32>() / filt_diffs.len() as f32;
        let var =
            filt_diffs.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / filt_diffs.len() as f32;
        var.sqrt()
    };

    let smooth_std = {
        let mean = smooth_diffs.iter().sum::<f32>() / smooth_diffs.len() as f32;
        let var = smooth_diffs.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
            / smooth_diffs.len() as f32;
        var.sqrt()
    };

    println!("Filtered std of diffs: {:.4}", filt_std);
    println!("Smoothed std of diffs: {:.4}", smooth_std);

    // Smoothed trajectory should be more consistent
    assert!(
        smooth_std <= filt_std,
        "Smoothed trajectory should be more consistent (lower std)"
    );
}

#[test]
fn test_kalman_smoothing_gains() {
    init();
    let c = 10;
    let f = 2;

    let noisy_state = create_noisy_centroids(c, f, 0.5);
    let mst_output = create_linear_mst(c);

    let kalman = SmoothingStage::new(SmoothingConfig::default());
    let output = kalman.execute(&noisy_state, &mst_output);

    // Test: Smoothing gains should be reasonable
    assert_eq!(
        output.smoothing_gains.len(),
        c - 1,
        "Should have C-1 smoothing gains"
    );

    for (i, gain) in output.smoothing_gains.iter().enumerate() {
        assert!(gain.is_finite(), "Smoothing gain {} should be finite", i);
        assert!(*gain >= 0.0, "Smoothing gain {} should be non-negative", i);
        assert!(
            *gain <= 1.0,
            "Smoothing gain {} should be <= 1.0 (typically)",
            i
        );
    }

    println!("Smoothing gains: {:?}", output.smoothing_gains);
}

#[test]
fn test_kalman_disconnected_mst() {
    init();
    let c = 5;
    let f = 2;

    let noisy_state = create_noisy_centroids(c, f, 0.3);

    // Create MST with gap (disconnected component)
    let mst_output = MSTOutput {
        candidate_edges: vec![],
        mst_edges: vec![
            Edge {
                u: 0,
                v: 1,
                distance: 1.0,
                thickness_u: 1.0,
                thickness_v: 1.0,
                cost: 1.0,
            },
            // Gap: no edge connecting to 2, 3, 4
        ],
        centroid_order: vec![0, 1, 2, 3, 4], // Order still includes all nodes
        trunk_nodes: vec![],
        thickness: vec![1.0; c],
        total_weight: 1.0,
        nodes_in_mst: 2, // Only 2 nodes covered
    };

    let kalman = SmoothingStage::new(SmoothingConfig::default());
    let output = kalman.execute(&noisy_state, &mst_output);

    // Test: Should complete without panic
    assert!(output.variance_reduction >= 0.0);

    // Test: Smoothed state should have all centroids
    let smoothed_state = output.to_centroid_state();
    assert_eq!(smoothed_state.num_centroids(), c);
}

#[test]
fn test_kalman_config_variants() {
    init();

    // Test: All config constructors should work
    let default = SmoothingConfig::default();
    let conservative = SmoothingConfig::conservative();
    let aggressive = SmoothingConfig::aggressive();
    let trunk_aware = SmoothingConfig::trunk_aware(0.5);

    // Conservative should trust observations more (lower R)
    assert!(conservative.observation_noise < default.observation_noise);

    // Aggressive should smooth more (higher R, lower Q)
    assert!(aggressive.observation_noise > default.observation_noise);
    assert!(aggressive.process_noise < default.process_noise);

    // Trunk-aware should have proper model
    assert!(matches!(
        trunk_aware.transition_model,
        TransitionModel::TrunkAware { .. }
    ));
}

#[test]
fn test_kalman_deterministic() {
    init();
    let c = 10;
    let f = 3;

    let noisy_state = create_noisy_centroids(c, f, 0.4);
    let mst_output = create_linear_mst(c);

    let kalman = SmoothingStage::new(SmoothingConfig::default());

    // Run twice with same input
    let output1 = kalman.execute(&noisy_state, &mst_output);
    let output2 = kalman.execute(&noisy_state, &mst_output);

    // Test: Output should be deterministic
    let means1: Vec<f32> = output1.smoothed_means.to_data().to_vec().unwrap();
    let means2: Vec<f32> = output2.smoothed_means.to_data().to_vec().unwrap();

    for (i, (m1, m2)) in means1.iter().zip(means2.iter()).enumerate() {
        assert!(
            (m1 - m2).abs() < 1e-6,
            "Smoothed means should be deterministic at index {}",
            i
        );
    }

    let vars1: Vec<f32> = output1.smoothed_variances.to_data().to_vec().unwrap();
    let vars2: Vec<f32> = output2.smoothed_variances.to_data().to_vec().unwrap();

    for (i, (v1, v2)) in vars1.iter().zip(vars2.iter()).enumerate() {
        assert!(
            (v1 - v2).abs() < 1e-6,
            "Smoothed variances should be deterministic at index {}",
            i
        );
    }
}
