// surfface-core/src/smoothing_chain.rs
//! Kalman smoothing stage: Regularize centroids along MST order
//!
//! This is Stage B2 of the Surfface pipeline:
//! - Takes MST 1D ordering from Stage B1
//! - Applies Rauch-Tung-Striebel (RTS) smoother along centroid sequence
//! - Produces smoothed centroid means and variances
//! - Reduces noise while preserving manifold structure
//!
//! Mathematical framework:
//! - State space model: x_t = F x_{t-1} + w_t (process model)
//! - Observation model: y_t = H x_t + v_t (observation = raw centroids)
//! - F = I (Identity, random walk) or F = αI (Damped)
//! - Forward pass: Kalman filter
//! - Backward pass: RTS smoothing equations
//!
//! References:
//! - Rauch, Tung, Striebel (1965): "Maximum likelihood estimates of linear dynamic systems"
//! - Särkkä (2013): "Bayesian Filtering and Smoothing"

use crate::centroid::CentroidState;
use crate::mst::MSTOutput;
use burn::prelude::*;

/// Configuration for Kalman smoothing
#[derive(Debug, Clone)]
pub struct SmoothingConfig {
    /// Process noise covariance Q (controls smoothness)
    /// Higher Q → more responsive to observations
    /// Lower Q → smoother trajectory
    pub process_noise: f32,

    /// Observation noise covariance R (controls trust in observations)
    /// Higher R → trust predictions more
    /// Lower R → trust observations more
    pub observation_noise: f32,

    /// State transition model type
    pub transition_model: TransitionModel,

    /// Minimum variance floor (numerical stability)
    pub variance_floor: f32,

    /// Maximum variance ceiling (prevent explosion)
    pub variance_ceiling: f32,
}

/// State transition model for Kalman filter
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TransitionModel {
    /// Identity: x_t = x_{t-1} + w_t (random walk)
    /// P_pred = P_filt + Q
    Identity,

    /// Damped: x_t = α x_{t-1} + w_t (exponential smoothing)
    /// P_pred = α² P_filt + Q
    /// α ∈ (0, 1) controls damping strength
    Damped(f32),

    /// Trunk-aware: Lower process noise along trunk edges,
    /// higher process noise on branch transitions.
    /// trunk_factor ∈ (0, 1]: Q_trunk = trunk_factor * Q
    TrunkAware { trunk_factor: f32 },
}

impl Default for SmoothingConfig {
    fn default() -> Self {
        Self {
            process_noise: 0.01,
            observation_noise: 0.1,
            transition_model: TransitionModel::Identity,
            variance_floor: 1e-6,
            variance_ceiling: 1e3,
        }
    }
}

impl SmoothingConfig {
    /// Conservative smoothing (trust observations more)
    pub fn conservative() -> Self {
        Self {
            process_noise: 0.1,
            observation_noise: 0.01,
            transition_model: TransitionModel::Identity,
            variance_floor: 1e-6,
            variance_ceiling: 1e3,
        }
    }

    /// Aggressive smoothing (smooth heavily)
    pub fn aggressive() -> Self {
        Self {
            process_noise: 0.001,
            observation_noise: 1.0,
            transition_model: TransitionModel::Identity,
            variance_floor: 1e-6,
            variance_ceiling: 1e3,
        }
    }

    /// Trunk-aware smoothing (preserve trunk structure)
    pub fn trunk_aware(trunk_factor: f32) -> Self {
        Self {
            process_noise: 0.01,
            observation_noise: 0.1,
            transition_model: TransitionModel::TrunkAware { trunk_factor },
            variance_floor: 1e-6,
            variance_ceiling: 1e3,
        }
    }
}

/// Output of Kalman smoothing stage
pub struct KalmanOutput<B: Backend> {
    /// Smoothed centroid means [C, F]
    pub smoothed_means: Tensor<B, 2>,

    /// Smoothed centroid variances [C, F]
    pub smoothed_variances: Tensor<B, 2>,

    /// Centroid counts (preserved from input state) [C]
    pub counts: Tensor<B, 1, Int>,

    /// Filtered means (from forward pass, for diagnostics) [C, F]
    pub filtered_means: Tensor<B, 2>,

    /// Filtered variances (from forward pass) [C, F]
    pub filtered_variances: Tensor<B, 2>,

    /// Mean smoothing gain per step (one entry per transition, length = C-1)
    pub smoothing_gains: Vec<f32>,

    /// Mean variance reduction ratio (smoothed vs raw); negative means increase
    pub variance_reduction: f32,
}

impl<B: Backend> KalmanOutput<B> {
    pub fn summary(&self) -> String {
        let n = self.smoothing_gains.len();
        let mean_gain = if n > 0 {
            self.smoothing_gains.iter().sum::<f32>() / n as f32
        } else {
            0.0
        };
        format!(
            "Kalman: variance_reduction={:.2}%, gains_mean={:.4} (over {} transitions)",
            self.variance_reduction * 100.0,
            mean_gain,
            n,
        )
    }

    /// Create smoothed CentroidState from this output
    pub fn to_centroid_state(&self) -> CentroidState<B> {
        CentroidState {
            means: self.smoothed_means.clone(),
            variances: self.smoothed_variances.clone(),
            counts: self.counts.clone(),
        }
    }
}

/// Kalman smoothing stage executor
pub struct SmoothingStage {
    config: SmoothingConfig,
}

impl SmoothingStage {
    pub fn new(config: SmoothingConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(SmoothingConfig::default())
    }

    /// Execute RTS Kalman smoothing along MST order
    pub fn execute<B: Backend>(
        &self,
        state: &CentroidState<B>,
        mst_output: &MSTOutput,
    ) -> KalmanOutput<B> {
        let device = state.means.device();
        let c = state.num_centroids();
        let f = state.feature_dim();

        log::info!("╔═══════════════════════════════════════════════════════╗");
        log::info!("║  STAGE B2: KALMAN SMOOTHING                           ║");
        log::info!("╚═══════════════════════════════════════════════════════╝");
        log::info!("📊 Smoothing {} centroids (F={}) along MST order", c, f);
        log::info!(
            "  • Process noise Q={:.4}, Observation noise R={:.4}",
            self.config.process_noise,
            self.config.observation_noise,
        );

        // Extract data to CPU for sequential processing
        let means_vec: Vec<f32> = state.means.to_data().to_vec().unwrap();
        let variances_vec: Vec<f32> = state.variances.to_data().to_vec().unwrap();

        let order = &mst_output.centroid_order;

        // FORWARD PASS: Kalman filter
        log::debug!("Step 1/2: Forward pass (Kalman filter)...");
        let (filtered_means, filtered_vars, predicted_means, predicted_vars) =
            self.forward_pass(&means_vec, &variances_vec, order, f, mst_output);

        // BACKWARD PASS: RTS smoother
        log::debug!("Step 2/2: Backward pass (RTS smoothing)...");
        let (smoothed_means, smoothed_vars, gains) = self.backward_pass(
            &filtered_means,
            &filtered_vars,
            &predicted_means,
            &predicted_vars,
            f,
        );

        // Diagnostics
        let raw_var_mean: f32 = variances_vec.iter().sum::<f32>() / variances_vec.len() as f32;
        let smoothed_var_mean: f32 = smoothed_vars.iter().sum::<f32>() / smoothed_vars.len() as f32;
        let variance_reduction = if raw_var_mean > 0.0 {
            (raw_var_mean - smoothed_var_mean) / raw_var_mean
        } else {
            0.0
        };

        log::info!("  ✓ Smoothing complete");
        log::info!(
            "    • Variance reduction: {:.2}%",
            variance_reduction * 100.0
        );
        log::info!(
            "    • Mean smoothing gain: {:.4} (over {} transitions)",
            gains.iter().sum::<f32>() / gains.len().max(1) as f32,
            gains.len(),
        );
        log::info!("╔═══════════════════════════════════════════════════════╗");
        log::info!("║  KALMAN SMOOTHING COMPLETE                            ║");
        log::info!("╚═══════════════════════════════════════════════════════╝");

        // Rebuild tensors
        let smoothed_means_tensor = Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(smoothed_means, burn::tensor::Shape::new([c, f])),
            &device,
        );
        let smoothed_vars_tensor = Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(smoothed_vars, burn::tensor::Shape::new([c, f])),
            &device,
        );
        let filtered_means_tensor = Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(filtered_means, burn::tensor::Shape::new([c, f])),
            &device,
        );
        let filtered_vars_tensor = Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(filtered_vars, burn::tensor::Shape::new([c, f])),
            &device,
        );

        KalmanOutput {
            smoothed_means: smoothed_means_tensor,
            smoothed_variances: smoothed_vars_tensor,
            counts: state.counts.clone(),
            filtered_means: filtered_means_tensor,
            filtered_variances: filtered_vars_tensor,
            smoothing_gains: gains,
            variance_reduction,
        }
    }

    // -------------------------------------------------------------------------
    // Forward pass: Kalman filter
    //
    // Returns four flat row-major arrays, all of shape [C, F]:
    //   filtered_means    — x_{t|t}
    //   filtered_vars     — P_{t|t}
    //   predicted_means   — x_{t|t-1}   (needed by backward pass)
    //   predicted_vars    — P_{t|t-1}   (needed by backward pass)
    //
    // Index 0 of predicted_* is intentionally unused (no prior step exists).
    // -------------------------------------------------------------------------
    fn forward_pass(
        &self,
        means: &[f32],
        variances: &[f32],
        order: &[usize],
        f: usize,
        mst_output: &MSTOutput,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let c = order.len();
        let mut filtered_means = vec![0.0f32; c * f];
        let mut filtered_vars = vec![0.0f32; c * f];
        // Slot [0..f] of predicted_* has no meaning; fill with NaN to surface
        // any accidental access in tests.
        let mut predicted_means = vec![f32::NAN; c * f];
        let mut predicted_vars = vec![f32::NAN; c * f];

        // ── t = 0: initialise with first observation ──────────────────────────
        let first_idx = order[0];
        for feat in 0..f {
            filtered_means[feat] = means[first_idx * f + feat];
            // Add process_noise to P_0 so the first Kalman update is not
            // overconfident when centroid variances are near the floor.
            filtered_vars[feat] = (variances[first_idx * f + feat] + self.config.process_noise)
                .clamp(self.config.variance_floor, self.config.variance_ceiling);
        }

        // ── t = 1 .. C-1 ─────────────────────────────────────────────────────
        for t in 1..c {
            let curr_idx = order[t];
            let prev_t = t - 1;

            // Determine whether this transition is along a trunk edge so that
            // TrunkAware can reduce process noise on the main path.
            let is_trunk_edge = matches!(
                self.config.transition_model,
                TransitionModel::TrunkAware { .. }
            ) && mst_output.trunk_edges.contains(&(order[prev_t], order[t]));

            for feat in 0..f {
                let x_filt_prev = filtered_means[prev_t * f + feat];
                let p_filt_prev = filtered_vars[prev_t * f + feat];

                // ── PREDICTION ───────────────────────────────────────────────
                let (x_pred, p_pred) = match self.config.transition_model {
                    TransitionModel::Identity => {
                        let p = p_filt_prev + self.config.process_noise;
                        (x_filt_prev, p)
                    }
                    TransitionModel::Damped(alpha) => {
                        // x_{t|t-1} = α · x_{t-1|t-1}
                        // P_{t|t-1} = α² · P_{t-1|t-1} + Q
                        let p = alpha * alpha * p_filt_prev + self.config.process_noise;
                        (alpha * x_filt_prev, p)
                    }
                    TransitionModel::TrunkAware { trunk_factor } => {
                        // Lower process noise on trunk: Q_eff = trunk_factor * Q
                        // Higher process noise on branches: Q_eff = Q
                        let q_eff = if is_trunk_edge {
                            self.config.process_noise * trunk_factor
                        } else {
                            self.config.process_noise
                        };
                        let p = p_filt_prev + q_eff;
                        (x_filt_prev, p)
                    }
                };

                let p_pred = p_pred.clamp(self.config.variance_floor, self.config.variance_ceiling);

                // Store predictions for the backward pass
                predicted_means[t * f + feat] = x_pred;
                predicted_vars[t * f + feat] = p_pred;

                // ── UPDATE ───────────────────────────────────────────────────
                let y_obs = means[curr_idx * f + feat];
                let r_obs = (variances[curr_idx * f + feat]
                    .clamp(self.config.variance_floor, self.config.variance_ceiling)
                    + self.config.observation_noise)
                    .max(self.config.variance_floor);

                // Innovation covariance S = P_pred + R  (H = I)
                let s = p_pred + r_obs;

                // Kalman gain K = P_pred / S  (clamped to [0, 1])
                let k = (p_pred / s).clamp(0.0, 1.0);

                // x_{t|t} = x_pred + K * (y - x_pred)
                filtered_means[t * f + feat] = x_pred + k * (y_obs - x_pred);

                // P_{t|t} = (1 - K) * P_pred
                filtered_vars[t * f + feat] = ((1.0 - k) * p_pred)
                    .clamp(self.config.variance_floor, self.config.variance_ceiling);
            }
        }

        (
            filtered_means,
            filtered_vars,
            predicted_means,
            predicted_vars,
        )
    }

    // -------------------------------------------------------------------------
    // Backward pass: RTS smoother
    //
    // Inputs:
    //   filtered_means/vars  — x_{t|t}, P_{t|t}  from forward pass
    //   predicted_means/vars — x_{t|t-1}, P_{t|t-1}  from forward pass
    //                          (index 0 is NaN / unused)
    //
    // Returns:
    //   smoothed_means  — x_{t|T}
    //   smoothed_vars   — P_{t|T}
    //   gains           — mean RTS gain per transition, length = C-1
    //
    // RTS equations (scalar per feature, diagonal covariance):
    //   J_t     = P_{t|t} / P_{t+1|t}
    //   x_{t|T} = x_{t|t} + J_t * (x_{t+1|T} - x_{t+1|t})
    //   P_{t|T} = P_{t|t} + J_t² * (P_{t+1|T} - P_{t+1|t})
    // -------------------------------------------------------------------------
    fn backward_pass(
        &self,
        filtered_means: &[f32],
        filtered_vars: &[f32],
        predicted_means: &[f32],
        predicted_vars: &[f32],
        f: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let c = filtered_means.len() / f;
        let mut smoothed_means = filtered_means.to_vec();
        let mut smoothed_vars = filtered_vars.to_vec();
        // gains[i] corresponds to the transition from step i → i+1
        let mut gains: Vec<f32> = Vec::with_capacity(c.saturating_sub(1));

        for t in (0..c - 1).rev() {
            let next_t = t + 1;
            let mut gain_sum = 0.0f32;

            for feat in 0..f {
                let p_filt = filtered_vars[t * f + feat];
                let p_pred_next = predicted_vars[next_t * f + feat]; // P_{t+1|t}

                // RTS gain J_t = P_{t|t} / P_{t+1|t}  (clamped to [0, 1])
                let j = if p_pred_next > self.config.variance_floor {
                    (p_filt / p_pred_next).clamp(0.0, 1.0)
                } else {
                    0.0
                };

                gain_sum += j;

                // x_{t|T} = x_{t|t} + J * (x_{t+1|T} - x_{t+1|t})
                // x_{t+1|t} is the stored predicted mean, NOT filtered_means[t]
                let x_smooth_next = smoothed_means[next_t * f + feat];
                let x_pred_next = predicted_means[next_t * f + feat];
                smoothed_means[t * f + feat] =
                    filtered_means[t * f + feat] + j * (x_smooth_next - x_pred_next);

                // P_{t|T} = P_{t|t} + J² * (P_{t+1|T} - P_{t+1|t})
                let p_smooth_next = smoothed_vars[next_t * f + feat];
                smoothed_vars[t * f + feat] = (p_filt + j * j * (p_smooth_next - p_pred_next))
                    .clamp(self.config.variance_floor, self.config.variance_ceiling);
            }

            gains.push(gain_sum / f as f32);
        }

        // Reverse so gains[0] = transition 0→1, gains[C-2] = transition (C-2)→(C-1)
        gains.reverse();

        (smoothed_means, smoothed_vars, gains)
    }
}
