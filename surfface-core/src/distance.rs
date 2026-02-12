// surfface-core/src/distance.rs
use burn::prelude::*;

/// Diagonal Gaussian Bhattacharyya distance [file:3]
/// D_B = Σ_k [ (μᵢᵏ - μⱼᵏ)² / (σᵢᵏ² + σⱼᵏ²) + 0.5 ln((σᵢᵏ² + σⱼᵏ²) / (2σᵢᵏσⱼᵏ)) ]
pub fn bhattacharyya_diagonal<B: Backend>(
    mean_i: Tensor<B, 1>,
    var_i: Tensor<B, 1>,
    mean_j: Tensor<B, 1>,
    var_j: Tensor<B, 1>,
) -> Tensor<B, 1> {
    // Variance sum: σᵢ² + σⱼ²
    let sigma_sum = var_i.clone() + var_j.clone();

    // Variance product: σᵢ² * σⱼ²
    let sigma_prod = var_i.clone() * var_j.clone();

    // Mean difference: (μᵢ - μⱼ)²
    let mean_diff = mean_i - mean_j;
    let mean_diff_sq = mean_diff.powf_scalar(2.0);

    // Mahalanobis term: 0.25 * (μᵢ - μⱼ)² / (σᵢ² + σⱼ²)
    let mahalanobis = (mean_diff_sq / sigma_sum.clone()).mul_scalar(0.25);

    // Log-determinant term: 0.25 * ln((σᵢ² + σⱼ²) / (2√(σᵢ²σⱼ²)))
    let log_term = (sigma_sum / (sigma_prod.sqrt().mul_scalar(2.0)))
        .log()
        .mul_scalar(0.25);

    // Sum over feature dimensions
    (mahalanobis + log_term).sum()
}

/// Affinity (for graph edge weights): w = exp(-D_B) [file:3]
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
/// Inputs: features [F, C], variances [F, C]
/// Output: distance matrix [F, F]
pub fn bhattacharyya_pairwise<B: Backend>(
    features: Tensor<B, 2>,
    variances: Tensor<B, 2>,
) -> Tensor<B, 2> {
    let [f, c] = features.dims();

    // Expand for broadcasting
    let features_i = features.clone().reshape([f, 1, c]); // [F, 1, C]
    let features_j = features.clone().reshape([1, f, c]); // [1, F, C]
    let var_i = variances.clone().reshape([f, 1, c]);
    let var_j = variances.clone().reshape([1, f, c]);

    // Vectorized computation
    let sigma_sum = var_i.clone() + var_j.clone();
    let sigma_prod = var_i * var_j;

    let mean_diff = features_i - features_j;
    let mean_diff_sq = mean_diff.powf_scalar(2.0);

    let mahalanobis = (mean_diff_sq / sigma_sum.clone()).mul_scalar(0.25);
    let log_term = (sigma_sum / (sigma_prod.sqrt().mul_scalar(2.0)))
        .log()
        .mul_scalar(0.25);

    // Sum over centroid dimension (C) to get [F, F]
    (mahalanobis + log_term).sum_dim(2).squeeze()
}
