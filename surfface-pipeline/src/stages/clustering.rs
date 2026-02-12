// surfface-pipeline/src/stages/clustering.rs
use burn::prelude::*;
use surfface_core::backend::AutoBackend;
use surfface_core::matrix::RowMatrix;

pub struct ClusteringStage {
    pub target_centroids: usize,
    pub radius: f32,
    pub batch_size: usize, // Critical for 10M items to avoid OOM
}

pub struct ClusteringOutput<B: Backend> {
    pub centroids: Tensor<B, 2>,        // [C, F]
    pub assignments: Tensor<B, 1, Int>, // [N]
    pub counts: Tensor<B, 1, Int>,      // [C]
}

impl ClusteringStage {
    pub fn execute(
        &self,
        data: RowMatrix<AutoBackend>,
        device: &<AutoBackend as Backend>::Device,
    ) -> ClusteringOutput<AutoBackend> {
        let [n, f] = data.tensor.dims();

        // Initialize with first item as first centroid
        let mut centroids = data.tensor.clone().slice([0..1]); // [1, F]
        let mut assignments = Vec::with_capacity(n);
        let mut n_centroids = 1usize;

        println!(
            "🎯 Clustering {} items into max {} centroids (batch_size={})",
            n, self.target_centroids, self.batch_size
        );

        // Process in batches to handle N=10M [file:2]
        for batch_start in (0..n).step_by(self.batch_size) {
            let batch_end = (batch_start + self.batch_size).min(n);
            let batch = data.tensor.clone().slice([batch_start..batch_end]); // [B, F]
            let batch_size = batch_end - batch_start;

            // Vectorized distance computation: ||x - c||²
            // Using expansion: ||a-b||² = ||a||² + ||b||² - 2⟨a,b⟩
            let batch_norm = batch
                .clone()
                .powf_scalar(2.0)
                .sum_dim(1)
                .reshape([batch_size, 1]); // [B, 1]

            let cent_norm = centroids
                .clone()
                .powf_scalar(2.0)
                .sum_dim(1)
                .reshape([1, n_centroids]); // [1, C]

            let interaction = batch.clone().matmul(centroids.clone().transpose()); // [B, C]

            // Distance matrix: [B, C]
            let dists = (batch_norm + cent_norm - interaction.mul_scalar(2.0)).sqrt();

            // Find nearest centroid per item
            let min_dists = dists.clone().min_dim(1); // [B]
            let nearest_idx = dists.argmin(1); // [B]

            // Download batch results to CPU for incremental logic
            // (In v0.27+, this could be done fully on GPU via custom kernels)
            let min_dists_cpu: Vec<f32> =
                min_dists.to_data().to_vec::<f32>().expect("Type mismatch");
            let nearest_idx_cpu: Vec<i64> = nearest_idx
                .to_data()
                .to_vec::<i64>()
                .expect("Type mismatch");

            for i in 0..batch_size {
                let dist = min_dists_cpu[i];
                let nearest = nearest_idx_cpu[i] as usize;

                if dist < self.radius {
                    // Assign to existing centroid [file:1]
                    assignments.push(nearest as i64);
                } else if n_centroids < self.target_centroids {
                    // Create new centroid [file:1]
                    let new_centroid = batch.clone().slice([i..(i + 1)]); // [1, F]
                    centroids = Tensor::cat(vec![centroids, new_centroid], 0);
                    assignments.push(n_centroids as i64);
                    n_centroids += 1;
                } else {
                    // Fallback to nearest (cluster is full)
                    assignments.push(nearest as i64);
                }
            }

            if batch_end % 100_000 == 0 {
                println!(
                    "  Processed {}/{} items, {} centroids",
                    batch_end, n, n_centroids
                );
            }
        }

        println!("✓ Clustering complete: {} centroids created", n_centroids);

        // Compute centroid counts
        let assignments_tensor =
            Tensor::<AutoBackend, 1, Int>::from_ints(assignments.as_slice(), device);

        let counts = self.compute_counts(&assignments_tensor, n_centroids, device);

        ClusteringOutput {
            centroids,
            assignments: assignments_tensor,
            counts,
        }
    }

    /// Count items per centroid
    fn compute_counts(
        &self,
        assignments: &Tensor<AutoBackend, 1, Int>,
        n_centroids: usize,
        device: &<AutoBackend as Backend>::Device,
    ) -> Tensor<AutoBackend, 1, Int> {
        let mut counts = vec![0i64; n_centroids];
        let assignments_cpu: Vec<i64> = assignments
            .to_data()
            .to_vec::<i64>()
            .expect("Type mismatch");

        for &c_id in &assignments_cpu {
            counts[c_id as usize] += 1;
        }

        Tensor::from_ints(counts.as_slice(), device)
    }
}
