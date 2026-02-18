//! TauMode: strategy for resolving the scalar τ from the lambda distribution.
//!
//! The τ parameter controls the blend between Rayleigh energy (E) and
//! Dirichlet dispersion (G) in the synthetic index:
//!   S = τ·E_bounded + (1 − τ)·(1 − G_clamped)

use serde::{Deserialize, Serialize};

pub const TAU_FLOOR: f32 = 1e-9;

/// Strategy for resolving the scalar τ from the lambda distribution.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum TauMode {
    /// Use the median of the lambda distribution (recommended for search).
    #[default]
    Median,
    /// Use the arithmetic mean of the lambda distribution.
    Mean,
    /// Use a fixed constant τ value.
    Fixed(f32),
    /// Use the p-th percentile (0.0–1.0) of the lambda distribution.
    Percentile(f32),
}

impl std::fmt::Display for TauMode {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            TauMode::Fixed(v) => write!(f, "Fixed({:.4})", v),
            TauMode::Median => write!(f, "Median"),
            TauMode::Mean => write!(f, "Mean"),
            TauMode::Percentile(p) => write!(f, "Percentile({:.2})", p),
        }
    }
}

/// Resolve a single scalar τ from the lambda distribution and the chosen mode.
pub fn compute_tau(lambdas: &[f32], mode: &TauMode) -> f32 {
    let finite: Vec<f32> = lambdas.iter().copied().filter(|v| v.is_finite()).collect();

    if finite.is_empty() {
        return TAU_FLOOR;
    }

    match mode {
        TauMode::Fixed(t) => {
            if t.is_finite() {
                (*t).max(TAU_FLOOR)
            } else {
                TAU_FLOOR
            }
        }
        TauMode::Mean => (finite.iter().sum::<f32>() / finite.len() as f32).max(TAU_FLOOR),
        TauMode::Median => {
            let mut sorted = finite.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted[sorted.len() / 2].max(TAU_FLOOR)
        }
        TauMode::Percentile(p) => {
            let mut sorted = finite.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let idx = ((sorted.len() as f32 - 1.0) * p.clamp(0.0, 1.0)).round() as usize;
            sorted[idx].max(TAU_FLOOR)
        }
    }
}
