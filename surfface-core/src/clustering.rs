// surfface-core/src/centroid.rs
use burn::prelude::*;

pub struct CentroidState<B: Backend> {
    pub means: Tensor<B, 2>,       // [C, F]
    pub variances: Tensor<B, 2>,   // [C, F] - Diagonal covariance
    pub counts: Tensor<B, 1, Int>, // [C]
}

impl<B: Backend> CentroidState<B> {
    /// Initialize from clustering output
    pub fn from_clustering(
        centroids: Tensor<B, 2>,
        counts: Tensor<B, 1, Int>,
        initial_variance: f32,
    ) -> Self {
        let [c, f] = centroids.dims();
        let variances = Tensor::ones([c, f], &centroids.device()).mul_scalar(initial_variance);

        Self {
            means: centroids,
            variances,
            counts,
        }
    }

    /// Kalman update with batched observations [file:2]
    pub fn kalman_update(
        &mut self,
        observations: Tensor<B, 2>,     // [N, F]
        assignments: Tensor<B, 1, Int>, // [N]
        obs_noise: f32,
    ) {
        let [c, f] = self.means.dims();

        // For each centroid, aggregate observations
        // (In v0.27+, use scatter operations for full GPU vectorization)
        let assignments_cpu: Vec<i64> = assignments
            .to_data()
            .to_vec::<i64>()
            .expect("Type mismatch");
        let obs_cpu = observations.to_data();

        for c_id in 0..c {
            let mask: Vec<bool> = assignments_cpu.iter().map(|&a| a == c_id as i64).collect();

            let n_obs = mask.iter().filter(|&&m| m).count();
            if n_obs == 0 {
                continue;
            }

            // Gather observations for this centroid
            // ... (Implementation details for masked gather)

            // Kalman gain: K = P / (P + R)
            let r = Tensor::ones([1, f], &self.means.device()).mul_scalar(obs_noise);
            let p_slice = self.variances.clone().slice([c_id..(c_id + 1)]);
            let kalman_gain = p_slice.clone() / (p_slice.clone() + r);

            // Update mean and variance
            // μ ← μ + K(z - μ)
            // P ← (I - K)P
            // ... (Complete update logic)
        }
    }

    /// Thickness proxy for MST weighting
    pub fn get_thickness(&self) -> Tensor<B, 1> {
        // 1. Calculate mean along dimension 1 (results in shape [N, 1])
        // 2. Squeeze dimension 1 to get shape [N] (Tensor<B, 1>)
        self.variances.clone().mean_dim(1).squeeze()
    }

    /// Transpose to feature-space for Laplacian [file:2]
    pub fn to_feature_nodes(&self) -> Tensor<B, 2> {
        self.means.clone().transpose() // [F, C]
    }
}
