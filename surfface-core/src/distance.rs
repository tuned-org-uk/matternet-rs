// surfface-core/src/distance.rs
//! Distance metrics for centroid and feature comparisons
//!
//! Implements:
//! - Bhattacharyya distance (diagonal Gaussian) [file:3]
//! - Euclidean and squared Euclidean distances
//! - Affinity conversions for graph edge weights
//!
//! The Bhattacharyya distance measures statistical overlap between
//! Gaussian distributions and is used for:
//! - MST edge weighting (Stage B1) [file:4]
//! - Feature-space Laplacian construction (Stage C) [file:2]

use burn::prelude::*;

/// Diagonal Gaussian Bhattacharyya distance [file:3]
///
/// Computes the Bhattacharyya distance between two Gaussian distributions
/// with diagonal covariance matrices:
///
/// D_B = Σ_k [ 0.25 * (μᵢᵏ - μⱼᵏ)² / (σᵢᵏ² + σⱼᵏ²)
///           + 0.25 * ln((σᵢᵏ² + σⱼᵏ²) / (2√(σᵢᵏ²σⱼᵏ²))) ]
///
/// Where:
/// - μᵢ, μⱼ: mean vectors
/// - σᵢ², σⱼ²: variance vectors (diagonal covariance)
///
/// Returns a scalar distance (sum over all dimensions)
pub fn bhattacharyya_diagonal<B: Backend>(
    mean_i: Tensor<B, 1>,
    var_i: Tensor<B, 1>,
    mean_j: Tensor<B, 1>,
    var_j: Tensor<B, 1>,
) -> Tensor<B, 1> {
    let eps = 1e-10; // Numerical stability

    // Regularize variances to prevent division by zero
    let var_i_reg = var_i.clamp_min(eps);
    let var_j_reg = var_j.clamp_min(eps);

    // Variance sum: σᵢ² + σⱼ²
    let sigma_sum = var_i_reg.clone() + var_j_reg.clone();

    // Variance product: σᵢ² * σⱼ²
    let sigma_prod = var_i_reg * var_j_reg;

    // Mean difference: (μᵢ - μⱼ)²
    let mean_diff = mean_i - mean_j;
    let mean_diff_sq = mean_diff.powf_scalar(2.0);

    // Mahalanobis term: 0.25 * (μᵢ - μⱼ)² / (σᵢ² + σⱼ²)
    let mahalanobis = (mean_diff_sq / sigma_sum.clone()).mul_scalar(0.25);

    // Log-determinant term: 0.25 * ln((σᵢ² + σⱼ²) / (2√(σᵢ²σⱼ²)))
    // Simplified: 0.25 * ln(σ_sum) - 0.25 * ln(2) - 0.25 * 0.5 * ln(σ_prod)
    let log_term = (sigma_sum / (sigma_prod.sqrt().mul_scalar(2.0)))
        .clamp_min(eps)
        .log()
        .mul_scalar(0.25);

    // Sum over feature dimensions
    (mahalanobis + log_term).sum()
}

/// Bhattacharyya distance for slices (CPU-friendly version)
///
/// Used in MST construction where we need to compute distances
/// between pairs of centroids efficiently without full tensor operations.
///
/// # Arguments
/// * `mean_i` - Mean vector for distribution i (length F)
/// * `var_i` - Variance vector for distribution i (length F)
/// * `mean_j` - Mean vector for distribution j (length F)
/// * `var_j` - Variance vector for distribution j (length F)
///
/// # Returns
/// Scalar Bhattacharyya distance
pub fn bhattacharyya_distance_diagonal(
    mean_i: &[f32],
    var_i: &[f32],
    mean_j: &[f32],
    var_j: &[f32],
) -> f32 {
    assert_eq!(mean_i.len(), mean_j.len());
    assert_eq!(var_i.len(), var_j.len());
    assert_eq!(mean_i.len(), var_i.len());

    let eps = 1e-10f32;
    let mut distance = 0.0f32;

    for k in 0..mean_i.len() {
        let sigma_i = var_i[k].max(eps);
        let sigma_j = var_j[k].max(eps);
        let sigma_sum = sigma_i + sigma_j;
        let sigma_prod = sigma_i * sigma_j;

        // Mahalanobis term
        let mean_diff = mean_i[k] - mean_j[k];
        let mahalanobis = 0.25 * (mean_diff * mean_diff) / sigma_sum;

        // Log-determinant term
        let log_term = 0.25 * ((sigma_sum / (2.0 * sigma_prod.sqrt())).max(eps)).ln();

        distance += mahalanobis + log_term;
    }

    distance
}

/// Affinity (for graph edge weights): w = exp(-D_B) [file:3]
///
/// Converts Bhattacharyya distance to an affinity weight:
/// - Distance 0 → Affinity 1 (identical distributions)
/// - Distance ∞ → Affinity 0 (no overlap)
pub fn bhattacharyya_affinity<B: Backend>(
    mean_i: Tensor<B, 1>,
    var_i: Tensor<B, 1>,
    mean_j: Tensor<B, 1>,
    var_j: Tensor<B, 1>,
) -> Tensor<B, 1> {
    let distance = bhattacharyya_diagonal(mean_i, var_i, mean_j, var_j);
    distance.neg().exp()
}

/// Batch Bhattacharyya for pairwise feature distances (F x F) [file:2]
///
/// Computes all pairwise Bhattacharyya distances between features
/// in feature-space (transposed centroids).
///
/// # Arguments
/// * `features` - Feature vectors [F, C] where each row is a feature's
///                values across C centroids
/// * `variances` - Variance vectors [F, C] for each feature
///
/// # Returns
/// Distance matrix [F, F] where entry (i, j) is D_B(feature_i, feature_j)
///
/// # Performance
/// - Memory: O(F² + FC)
/// - Time: O(F²C)
///
/// For large F (e.g., 100K), use sparse k-NN approximation instead [file:2]
pub fn bhattacharyya_pairwise<B: Backend>(
    features: Tensor<B, 2>,
    variances: Tensor<B, 2>,
) -> Tensor<B, 2> {
    let [f, c] = features.dims();
    let eps = 1e-10;

    // Regularize variances
    let variances_reg = variances.clamp_min(eps);

    // Expand for broadcasting: [F, 1, C] and [1, F, C]
    let features_i = features.clone().reshape([f, 1, c]);
    let features_j = features.clone().reshape([1, f, c]);
    let var_i = variances_reg.clone().reshape([f, 1, c]);
    let var_j = variances_reg.clone().reshape([1, f, c]);

    // Vectorized computation [F, F, C]
    let sigma_sum = var_i.clone() + var_j.clone();
    let sigma_prod = var_i * var_j;

    let mean_diff = features_i - features_j;
    let mean_diff_sq = mean_diff.powf_scalar(2.0);

    // Mahalanobis term [F, F, C]
    let mahalanobis = (mean_diff_sq / sigma_sum.clone()).mul_scalar(0.25);

    // Log-determinant term [F, F, C]
    let log_term = (sigma_sum / (sigma_prod.sqrt().mul_scalar(2.0)))
        .clamp_min(eps)
        .log()
        .mul_scalar(0.25);

    // Sum over centroid dimension (C) to get [F, F]
    (mahalanobis + log_term).sum_dim(2).squeeze()
}

/// Euclidean L2 distance between two vectors
pub fn euclidean_distance<B: Backend>(vec_i: Tensor<B, 1>, vec_j: Tensor<B, 1>) -> Tensor<B, 1> {
    let diff = vec_i - vec_j;
    diff.powf_scalar(2.0).sum().sqrt()
}

/// Squared Euclidean distance (avoids sqrt for speed)
pub fn squared_euclidean_distance<B: Backend>(
    vec_i: Tensor<B, 1>,
    vec_j: Tensor<B, 1>,
) -> Tensor<B, 1> {
    let diff = vec_i - vec_j;
    diff.powf_scalar(2.0).sum()
}

/// Euclidean distance for slices (CPU-friendly)
pub fn euclidean_distance_slice(vec_i: &[f32], vec_j: &[f32]) -> f32 {
    assert_eq!(vec_i.len(), vec_j.len());
    vec_i
        .iter()
        .zip(vec_j.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Squared Euclidean distance for slices
pub fn squared_euclidean_distance_slice(vec_i: &[f32], vec_j: &[f32]) -> f32 {
    assert_eq!(vec_i.len(), vec_j.len());
    vec_i
        .iter()
        .zip(vec_j.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum()
}

/// Cosine similarity: cos(θ) = (a·b) / (||a|| ||b||)
pub fn cosine_similarity<B: Backend>(vec_i: Tensor<B, 1>, vec_j: Tensor<B, 1>) -> Tensor<B, 1> {
    let dot_product = (vec_i.clone() * vec_j.clone()).sum();
    let norm_i = vec_i.powf_scalar(2.0).sum().sqrt();
    let norm_j = vec_j.powf_scalar(2.0).sum().sqrt();

    dot_product / (norm_i * norm_j).clamp_min(1e-10)
}

/// Cosine distance: 1 - cos(θ)
pub fn cosine_distance<B: Backend>(vec_i: Tensor<B, 1>, vec_j: Tensor<B, 1>) -> Tensor<B, 1> {
    let sim = cosine_similarity(vec_i, vec_j);
    Tensor::ones_like(&sim) - sim
}

//
// Laplacian compute: Affinity kernels for manifold wiring.
//
// Design contract:
//   - All kernels return a *similarity* (affinity) in [0, 1].
//   - BC(i, j) = exp(-DB(i, j)), where DB is the Bhattacharyya distance.
//   - Variance regularisation is mandatory: σ² is clamped below by `reg`
//     to keep the log term finite and prevent zero-variance features from
//     acting as perfect discriminators.

// Reference:
//   Wikipedia – Bhattacharyya distance (Gaussian case, 1D)
//   file:4 – Diagonal Gaussian Bhattacharyya for FastPair Pipeline

/// Compute the Diagonal Gaussian Bhattacharyya Coefficient BC ∈ [0, 1]
/// between two features, each described by a mean profile and a variance
/// profile over C centroids.
///
/// For each centroid dimension c:
///   DB_1D = (μ_ic - μ_jc)² / (4(σ_ic² + σ_jc²))
///         + 0.5 * ln( (σ_ic² + σ_jc²) / (2 * σ_ic * σ_jc) )
///
///   DB(i, j) = Σ_c DB_1D
///   BC(i, j) = exp(-DB(i, j))
///
/// Arguments:
///   mu_i / mu_j    – mean profiles of features i and j, length C
///   var_i / var_j  – variance profiles, length C (raw Kalman outputs)
///   reg            – variance floor (e.g. 1e-6) applied before log
#[inline]
pub fn bhattacharyya_coefficient(
    mu_i: &[f32],
    var_i: &[f32],
    mu_j: &[f32],
    var_j: &[f32],
    reg: f32,
) -> f32 {
    debug_assert_eq!(mu_i.len(), mu_j.len());
    debug_assert_eq!(var_i.len(), var_j.len());
    debug_assert_eq!(mu_i.len(), var_i.len());

    let mut db = 0.0f32;
    for c in 0..mu_i.len() {
        // Apply variance floor so log is always finite.
        let vi = var_i[c].max(reg);
        let vj = var_j[c].max(reg);
        let v_sum = vi + vj;

        // Mean-difference term: (μ_i - μ_j)² / (4 * (σ_i² + σ_j²))
        let mean_term = (mu_i[c] - mu_j[c]).powi(2) / (4.0 * v_sum);

        // Log-variance term: 0.5 * ln( (σ_i² + σ_j²) / (2 * σ_i * σ_j) )
        // (vi * vj).sqrt() is σ_i * σ_j in the scalar 1D case.
        let log_term = 0.5 * (v_sum / (2.0 * (vi * vj).sqrt())).ln();

        db += mean_term + log_term;
    }

    // BC = exp(-DB) ∈ (0, 1].  Clamp for float hygiene.
    (-db).exp().clamp(0.0, 1.0)
}
