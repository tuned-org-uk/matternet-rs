pub mod builder;
pub mod stages;

use burn::prelude::*;
use surfface_core::backend::{AutoBackend, SurffaceDevice, dispatch, print_backend_info};

pub fn build(data_vec: Vec<f32>, n_features: usize) {
    // 1. Hardware Telemetry
    print_backend_info();

    // 2. Dispatch to the right backend
    dispatch(|device_variant| match device_variant {
        SurffaceDevice::Cuda(d) => {
            let device = *d.downcast::<<AutoBackend as Backend>::Device>().unwrap();
            execute_stages::<AutoBackend>(data_vec, n_features, device);
        }
        SurffaceDevice::Wgpu(d) => {
            let device = *d.downcast::<<AutoBackend as Backend>::Device>().unwrap();
            execute_stages::<AutoBackend>(data_vec, n_features, device);
        }
        SurffaceDevice::Cpu(d) => {
            let device = *d.downcast::<<AutoBackend as Backend>::Device>().unwrap();
            execute_stages::<AutoBackend>(data_vec, n_features, device);
        }
    });
}

/// 1. Preflight [file:2]
/// 2. Clustering [file:5]
/// 3. Kalman + MST [file:4]
/// 4. Feature-Space Laplacian [file:3]
pub fn execute_stages<B: Backend>(
    data_vec: Vec<f32>,
    n_features: usize,
    device: <AutoBackend as Backend>::Device,
) {
    println!("🚀 Starting Surfface Pipeline on {:?}", device);

    // surfface-pipeline/src/lib.rs
    use crate::stages::clustering::ClusteringStage;
    use surfface_core::clustering::CentroidState;
    use surfface_core::matrix::RowMatrix;

    let n_items = data_vec.len() / n_features;

    let data = RowMatrix::from_vec(data_vec, n_items, n_features, &device);

    // Stage A: Clustering [file:1]
    let clustering = ClusteringStage {
        target_centroids: 10_000,
        radius: 1.5,
        batch_size: 10_000,
    };
    let cluster_output = clustering.execute(data, &device);

    println!(
        "✓ Stage A: {} centroids created",
        cluster_output.centroids.dims()[0]
    );

    // Stage B0: Kalman State [file:2]
    let mut centroid_state = CentroidState::from_clustering(
        cluster_output.centroids,
        cluster_output.counts,
        0.1, // initial variance
    );

    println!("✓ Stage B0: Kalman state initialized");

    // Stage C: Feature-space Laplacian [file:2][file:3]
    let features = centroid_state.to_feature_nodes(); // [F=100K, C=10K]
    // ... (k-NN + Bhattacharyya wiring)

    println!("✓ Pipeline complete");
}
