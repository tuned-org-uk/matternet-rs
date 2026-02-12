// surfface-core/src/tests/test_projection.rs
use crate::clustering::{ImplicitProjection, compute_jl_dimension};

// ============================================================================
// ImplicitProjection Tests
// ============================================================================

#[test]
fn test_implicit_projection_creates() {
    let proj = ImplicitProjection::new(100, 10, Some(42));
    assert_eq!(proj.original_dim, 100);
    assert_eq!(proj.target_dim, 10);
    assert_eq!(proj.seed, 42);
}

#[test]
fn test_implicit_projection_dimensions() {
    let proj = ImplicitProjection::new(50, 8, Some(42));
    let query = vec![0.5; 50];

    let projected = proj.project(&query);

    assert_eq!(projected.len(), 8);
    assert!(projected.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_implicit_projection_deterministic() {
    // Same seed should produce same projection
    let proj = ImplicitProjection::new(30, 5, Some(42));
    let query = vec![1.0; 30];

    let result1 = proj.project(&query);
    let result2 = proj.project(&query);

    assert_eq!(result1, result2);
}

#[test]
fn test_implicit_projection_different_seeds() {
    // Different seeds should produce different results
    let proj1 = ImplicitProjection::new(20, 5, Some(42));
    let proj2 = ImplicitProjection::new(20, 5, Some(99));

    let query = vec![1.0; 20];
    let result1 = proj1.project(&query);
    let result2 = proj2.project(&query);

    // Results should differ due to different seeds
    assert_ne!(result1, result2);
}

#[test]
fn test_implicit_projection_zero_vector() {
    let proj = ImplicitProjection::new(40, 10, Some(42));
    let query = vec![0.0; 40];

    let projected = proj.project(&query);

    assert_eq!(projected.len(), 10);
    // All should be near-zero
    assert!(projected.iter().all(|&x| x.abs() < 1e-10));
}

#[test]
fn test_implicit_projection_linearity() {
    let proj = ImplicitProjection::new(25, 6, Some(42));

    let query = vec![1.0; 25];
    let scaled_query: Vec<f32> = query.iter().map(|x| x * 2.0).collect();

    let proj1 = proj.project(&query);
    let proj2 = proj.project(&scaled_query);

    // Projection is linear: project(2x) = 2*project(x)
    for i in 0..proj1.len() {
        let expected = proj1[i] * 2.0;
        let actual = proj2[i];
        assert!(
            (expected - actual).abs() < 1e-6,
            "Linearity violation at {}: expected {}, got {}",
            i,
            expected,
            actual
        );
    }
}

#[test]
fn test_implicit_projection_preserves_scale() {
    let proj = ImplicitProjection::new(50, 15, Some(42));
    let query = vec![1.0; 50];

    let projected = proj.project(&query);

    let orig_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
    let proj_norm: f32 = projected.iter().map(|x| x * x).sum::<f32>().sqrt();

    let ratio = proj_norm / orig_norm;

    // JL guarantees approximate norm preservation
    assert!(
        ratio > 0.5 && ratio < 2.0,
        "Norm ratio {} outside [0.5, 2.0]",
        ratio
    );
}

#[test]
fn test_implicit_projection_non_trivial() {
    let proj = ImplicitProjection::new(30, 8, Some(42));
    let query = vec![1.0; 30];

    let projected = proj.project(&query);

    // Should have at least one non-zero value
    let has_nonzero = projected.iter().any(|&x| x.abs() > 1e-10);
    assert!(has_nonzero, "Projection should produce non-zero values");
}

#[test]
fn test_implicit_projection_different_inputs() {
    let proj = ImplicitProjection::new(20, 5, Some(42));

    let query1 = vec![1.0; 20];
    let query2 = vec![2.0; 20];

    let result1 = proj.project(&query1);
    let result2 = proj.project(&query2);

    // Different inputs should produce different outputs
    assert_ne!(result1, result2);
}

// ============================================================================
// compute_jl_dimension Tests (v0.25.12 - adaptive buffer)
// ============================================================================

#[test]
fn test_jl_dimension_preserves_low_dims() {
    // Rule 1: Dimensions < 32 are preserved exactly
    assert_eq!(compute_jl_dimension(100, 16, 0.3), 16);
    assert_eq!(compute_jl_dimension(1000, 8, 0.1), 8);
    assert_eq!(compute_jl_dimension(50, 31, 0.2), 31);
    assert_eq!(compute_jl_dimension(10, 1, 0.5), 1);
}

#[test]
fn test_jl_dimension_never_expands() {
    // Rule 2: Never exceed original_dim (upper bound clamp)
    let n = 10;
    let epsilon = 0.3;

    // JL bound for n=10, ε=0.3: ~205 dims
    // But if original is only 100, must cap at 100
    assert_eq!(compute_jl_dimension(n, 100, epsilon), 100);
    assert_eq!(compute_jl_dimension(n, 50, epsilon), 50);

    // For n=100, ε=0.5: JL bound = 8*ln(100)/0.25 ≈ 148
    let dim = compute_jl_dimension(100, 200, 0.5);
    assert!(dim >= 148 && dim <= 149); // Within [32, 200], uses JL bound
}

#[test]
fn test_jl_dimension_minimum_bound() {
    // For very small n and large ε, JL bound might be < 32
    // Should clamp to 32 minimum (if original_dim >= 32)
    let n = 2;
    let epsilon = 0.9;

    // JL: 8*ln(2)/0.81 ≈ 6.9 → clamped to 32
    assert_eq!(compute_jl_dimension(n, 1000, epsilon), 32);

    // But if original_dim < 32, preserve it
    assert_eq!(compute_jl_dimension(n, 20, epsilon), 20);
}

#[test]
fn test_jl_dimension_grows_with_n() {
    // More points → more dimensions needed (logarithmic growth)
    let epsilon = 0.2;
    let original_dim = 10_000;

    let dim_100 = compute_jl_dimension(100, original_dim, epsilon);
    let dim_1000 = compute_jl_dimension(1000, original_dim, epsilon);
    let dim_10000 = compute_jl_dimension(10000, original_dim, epsilon);

    assert!(
        dim_1000 > dim_100,
        "dim_1000 ({}) should be > dim_100 ({})",
        dim_1000,
        dim_100
    );
    assert!(
        dim_10000 > dim_1000,
        "dim_10000 ({}) should be > dim_1000 ({})",
        dim_10000,
        dim_1000
    );
}

#[test]
fn test_jl_dimension_inversely_proportional_epsilon() {
    // Smaller ε → more dimensions needed (quadratic relationship)
    let n = 5000;
    let original_dim = 10_000;

    let dim_05 = compute_jl_dimension(n, original_dim, 0.5);
    let dim_02 = compute_jl_dimension(n, original_dim, 0.2);
    let dim_01 = compute_jl_dimension(n, original_dim, 0.1);

    assert!(
        dim_02 > dim_05,
        "dim_02 ({}) should be > dim_05 ({})",
        dim_02,
        dim_05
    );
    assert!(
        dim_01 > dim_02,
        "dim_01 ({}) should be > dim_02 ({})",
        dim_01,
        dim_02
    );
}

#[test]
fn test_jl_dimension_dorothea_scenario() {
    // Real-world case from Dorothea experiments
    // 17 clusters, 100K features [file:1]
    let n_clusters = 17;
    let original_dim = 100_000;
    let epsilon = 0.3;

    let dim = compute_jl_dimension(n_clusters, original_dim, epsilon);

    // JL: 8*ln(17)/0.09 ≈ 251
    // Should be significantly compressed but > 32
    assert!(
        dim >= 32 && dim <= 1000,
        "Dorothea projection should be 32-1000, got {}",
        dim
    );
}

#[test]
fn test_jl_dimension_reasonable_range() {
    // Sanity checks: results should be practical
    let test_cases = vec![
        (100, 384, 0.2),    // BERT embeddings
        (200, 1536, 0.3),   // OpenAI embeddings
        (50, 100_000, 0.3), // Dorothea-like
        (1000, 768, 0.15),  // RoBERTa embeddings
    ];

    for (n, original_dim, eps) in test_cases {
        let dim = compute_jl_dimension(n, original_dim, eps);

        // Basic sanity
        assert!(
            dim >= 32 || dim == original_dim,
            "Result {} should be >= 32 or == original_dim for n={}, F={}",
            dim,
            n,
            original_dim
        );
        assert!(
            dim <= original_dim,
            "Result {} should be <= original_dim {} for n={}",
            dim,
            original_dim,
            n
        );

        // Practical bounds
        assert!(
            dim < 10_000,
            "Result too large: {} for n={}, F={}",
            dim,
            n,
            original_dim
        );
    }
}

#[test]
fn test_jl_dimension_edge_case_single_point() {
    // Pathological case: n=1 (single point)
    let n = 1;
    let epsilon = 0.1;

    // JL: 8*ln(1)/0.01 = 0 → clamped to 32 or original_dim
    assert_eq!(compute_jl_dimension(n, 100, epsilon), 32);
    assert_eq!(compute_jl_dimension(n, 10, epsilon), 10); // Preserve if < 32
}

#[test]
fn test_jl_dimension_consistency() {
    // Same n and ε should give same result for same original_dim
    let n = 500;
    let original_dim = 5000;
    let epsilon = 0.2;

    let dim1 = compute_jl_dimension(n, original_dim, epsilon);
    let dim2 = compute_jl_dimension(n, original_dim, epsilon);

    assert_eq!(dim1, dim2, "Function should be deterministic");
}

#[test]
fn test_jl_dimension_monotonicity() {
    // Verify monotonicity: more points → more dims needed
    let original_dim = 10_000;
    let epsilon = 0.3;

    let d1 = compute_jl_dimension(10, original_dim, epsilon);
    let d2 = compute_jl_dimension(100, original_dim, epsilon);
    let d3 = compute_jl_dimension(1000, original_dim, epsilon);

    assert!(d2 > d1, "d2 ({}) should be > d1 ({})", d2, d1);
    assert!(d3 > d2, "d3 ({}) should be > d2 ({})", d3, d2);
}

// ============================================================================
// Integration Tests
// ============================================================================

#[test]
fn test_full_pipeline_projection() {
    use crate::backend::AutoBackend;
    use crate::clustering::ClusteringStage;

    crate::init();
    let device = Default::default();

    // Create high-dimensional data that triggers projection
    let n_samples = 100;
    let orig_dim = 2000;

    let data: Vec<Vec<f32>> = (0..n_samples)
        .map(|_| (0..orig_dim).map(|_| rand::random::<f32>()).collect())
        .collect();

    let config = crate::clustering::ClusteringConfig {
        max_clusters: 20,
        radius_threshold: 1.5,
        seed: Some(42),
        use_projection: true,
        projection_threshold: 1000,
        jl_epsilon: 0.3,
        min_projected_dim: 64,
    };

    let stage = ClusteringStage::new(config);
    let output = stage.execute_from_vec::<AutoBackend>(data, &device);

    // Verify projection happened
    assert!(
        output.projection.is_some(),
        "Projection should be used for F=2000"
    );
    assert!(
        output.working_dim < orig_dim,
        "Working dim should be reduced"
    );
    assert!(output.working_dim >= 64, "Working dim should be >= min");

    // Verify clustering worked
    assert!(output.state.num_centroids() > 0);
    assert!(output.state.num_centroids() <= 20);
}

#[test]
fn test_projection_preserves_distances_approximately() {
    let proj = ImplicitProjection::new(100, 20, Some(42));

    let vec1 = vec![1.0; 100];
    let vec2: Vec<f32> = (0..100)
        .map(|i| if i % 2 == 0 { 1.0 } else { 0.0 })
        .collect();

    // Original distance
    let orig_dist_sq: f32 = vec1
        .iter()
        .zip(vec2.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum();

    // Projected distance
    let proj1 = proj.project(&vec1);
    let proj2 = proj.project(&vec2);
    let proj_dist_sq: f32 = proj1
        .iter()
        .zip(proj2.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum();

    let ratio = proj_dist_sq / orig_dist_sq;

    // JL lemma: distances preserved within (1±ε)² with high probability
    // For ε=0.3, expect ratio in [0.49, 1.69]
    assert!(
        ratio > 0.3 && ratio < 2.5,
        "Distance ratio {} outside reasonable JL bounds",
        ratio
    );
}

#[test]
fn test_memory_efficiency() {
    // ImplicitProjection should be tiny (just 24 bytes on 64-bit)
    let proj = ImplicitProjection::new(1000, 100, Some(42));

    // Verify it can project without storing the matrix
    let query = vec![1.0; 1000];
    let projected = proj.project(&query);

    assert_eq!(projected.len(), 100);

    // The struct should be minimal size
    assert_eq!(std::mem::size_of::<ImplicitProjection>(), 24); // 3 usizes
}

#[test]
fn test_projection_with_various_dimensions() {
    let test_cases = vec![(100, 10), (500, 50), (1000, 100), (5000, 500)];

    for (orig, target) in test_cases {
        let proj = ImplicitProjection::new(orig, target, Some(42));
        let query = vec![1.0; orig];
        let result = proj.project(&query);

        assert_eq!(
            result.len(),
            target,
            "Projection failed for {}→{}",
            orig,
            target
        );
        assert!(
            result.iter().all(|&x| x.is_finite()),
            "Non-finite values in projection"
        );
    }
}
