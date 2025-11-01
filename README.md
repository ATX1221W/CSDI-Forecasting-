# CSDI-Forecasting-
================================================================================
CSDI SYSTEM PARAMETERS & WORKFLOW (Updated for steps-only high-frequency mode)
================================================================================
Generated: October 31, 2025
System: 2-Channel Forced ODE Forecasting (state x, binary input u) with CSDI Diffusion
Primary Files: `train_forced_ode_steps_only.py`, `eval_forced_ode_steps_only.py`, `main_model.py`
================================================================================

TABLE OF CONTENTS
-----------------
1. Overview & Goals
2. Physics & Trajectory Generation
3. Forcing Input Modes (`steps_cfg`)
4. Dataset Splits & Seeds
5. Normalization Strategy
6. Masking Strategies (Training)
7. Model & Diffusion Configuration
8. Training Loop & Early Stopping
9. Evaluation Modes & Metrics
10. Uncertainty & Coverage Calibration
11. Saved Artifacts
12. Design Rationale & Future Improvements
13. Environment Setup & Running

QUICKSTART (Concise)
--------------------
1. Create env & install deps:
    python -m venv csdi && csdi\Scripts\Activate.ps1 && pip install -r requirements.txt
2. Train (example):
    `python train_forced_ode_steps_only.py --epochs 12 --flip-prob 0.18 --min-dwell 15`
3. Evaluate (replace <run>):
    `python eval_forced_ode_steps_only.py --n-samples 30 --side-input --run-dir save/steps_only_<run>`
4. Inspect artifacts in `save/steps_only_<run>/` (plots + metrics JSON).
5. Adjust `--flip-prob` / `--min-dwell` and retrain for different input smoothness.

================================================================================
1. OVERVIEW & GOALS
================================================================================
Objective: Learn causal forecasting of a damped forced oscillator under partially observed state trajectories while always observing the forcing input. Provide calibrated uncertainty bands and robust generalization to unseen damping coefficients.

Channels:
- x: system state (position)
- u: exogenous binary forcing (piecewise constant)

================================================================================
2. PHYSICS & TRAJECTORY GENERATION
================================================================================
ODE: m*x'' + c*x' + k*x = d*u
Fixed parameters: m=1.0, k=1.0, d=1.0
Damping c varies by split (train/val/test, disjoint sets).
Integrator: Forward Euler with dt=0.05, length T=200 steps.
State noise: Gaussian additive noise on x with std=0.05.
Initial conditions: x0 ∈ Uniform(0.5,1.5), v0 ∈ Uniform(-0.3,0.3).

================================================================================
3. FORCING INPUT MODES (`steps_cfg`)
================================================================================
Common keys:
- min_steps, max_steps: segment count for coarse mode.
- min_dwell, max_dwell: segment dwell bounds (coarse) or minimum run length (high-frequency).
- level_range: continuous value range if not binary.
- binary: use 0/1 levels.
- binary_high_freq: if True, per-step potential flips after minimum dwell (OFF by default; enable via `--high-freq 1`).
- flip_prob: probability of flipping once dwell ≥ min_dwell.

Modes:
1. Coarse Binary: segmented 0/1 plateaus; dwell sampled in [min_dwell,max_dwell].
2. High-Frequency Binary: enforce run ≥ min_dwell; then geometric tail with flip probability `flip_prob`. Expected run length ≈ min_dwell + (1/flip_prob) - 1.
3. Continuous Steps: like coarse but levels uniform in `level_range`.

Current training defaults (overridable via CLI):
```
min_steps=3, max_steps=6,
min_dwell=15, max_dwell=60,
binary=True, binary_high_freq=False,  # default coarse mode; pass --high-freq 1 to enable high-frequency
flip_prob=0.18, level_range=(0.0,1.0)
```

================================================================================
4. DATASET SPLITS & SEEDS
================================================================================
Disjoint damping coefficients:
- Train c values (12): 0.5,0.7,0.9,1.1,1.3,1.5, 2.2,2.36,2.52,2.68,2.84,3.0
- Validation c values (4): 1.6,1.7,1.8,1.9 (near-critical corridor)
- Test c values (6): 0.6,0.8,1.4,2.0,2.6,3.1 (includes critical c=2.0 & extrapolation)

Seeds:
- train_base, val_base, test_base, norm_base ensure reproducibility of inputs and ODE trajectories.

Sequences per system:
- Train: 10
- Validation: 5
- Test: 5 (generated on-the-fly in evaluation)

================================================================================
5. NORMALIZATION STRATEGY
================================================================================
Generate 500 synthetic (x,u) pairs across train c values to compute:
- x_mean, x_std
- u_mean, u_std
Applied channel-wise: normalized = (raw - mean)/std.
Stored in `config.json` for consistent evaluation.

================================================================================
6. MASKING STRATEGIES (TRAINING)
================================================================================
Mixed strategy sampling per batch item with probabilities:
- prefix: 0.5 (random fraction p ∈ [0.3,0.7])
- random: 0.3 (scattered points ratio ∈ [0.4,0.8])
- half: 0.2 (fixed p = 0.5)

Within prefix/half, dropout applied: rate ∈ [0.05,0.15] randomly removing some conditioning points.
u channel: always fully observed.
Purpose: Encourage robustness across partial histories & enable causal forecasting beyond simply reconstruction.

================================================================================
7. MODEL & DIFFUSION CONFIGURATION
================================================================================
Key model config:
- Conditional (is_unconditional=False)
- Embeddings: timeemb=256, featureemb=32
- Transformer diffusion backbone: layers=6, channels=96, nheads=8, diffusion_embedding_dim=128
- Beta schedule linear: beta_start=1e-4, beta_end=0.02, num_steps=100
- side_dim=289 (aux dimension for diffusion embeddings)

Target dimension: 2 channels (x,u) modeled jointly; u treated as observed (conditioning) but included for context embedding.

================================================================================
8. TRAINING LOOP & EARLY STOPPING
================================================================================
Hyperparameters:
- epochs (cap; e.g. 12 or provided via CLI)
- itr_per_epoch (default 50)
- batch_size=16, lr=1e-3
- patience=4, min_delta=1e-3

Procedure per epoch:
1. Iterate `itr_per_epoch` batches with mixed masking.
2. Compute validation future RMSE across several batches (prefix splits 0.5 & 0.7).
3. If improved > min_delta, save `model_best.pth` and reset patience counter; else increment patience.
4. Stop early when patience exhausted; always save final `model.pth` and history artifacts.

Validation Metric: Future RMSE = RMSE on unobserved portion past cutoff in validation prefix.

================================================================================
9. EVALUATION MODES & METRICS
================================================================================
Executed by `eval_forced_ode_steps_only.py` on unseen test set (generated using stored `steps_cfg`).

Modes:
1. Incremental Evaluation:
    - Split trajectory into n_days (default 3).
    - Each day adds a new window of observed points with internal missing rate (default 0.2).
    - Metrics: per-item per-day RMSE on currently unobserved portion; aggregated mean per day.
    - Visualization: rows = items; columns = days (+ side input panel if `--side-input`).
2. Half-Forecast:
    - Condition on first p fraction (default p=0.5) with dropout (0.1 fixed in eval) then forecast remainder.
    - Metrics: future RMSE, interval coverage for band (percentiles p_low=5, p_high=95).
3. Multi-Input Grid:
    - Multiple sequences per damping coefficient; prefix mask at 50%; shows mean forecast & uncertainty band plus overlapping input.
4. Input Diversity Summary:
    - Overlays several u(t) samples for a given c; histogram of dwell lengths (first sequence) to characterize input run distribution.

Stored Metrics:
- `incremental_rmse.json`: per-item-per-day and mean-per-day RMSE.
- `half_forecast_metrics.json`: per-item future RMSE, mean_future_rmse, interval_coverage.

================================================================================
10. UNCERTAINTY & COVERAGE CALIBRATION
================================================================================
Sampling: Multiple diffusion samples (configurable `--n-samples`).
Band: Percentile interval [p_low, p_high] (default 5–95%).
Coverage: Fraction of true future points inside band; values < nominal (e.g. 0.60 vs 0.90) indicate under-dispersion.
Adjustment levers: increase sample count, increase conditioning dropout, widen band (e.g. 2.5–97.5), or introduce variance inflation / ensemble diversity.

================================================================================
11. SAVED ARTIFACTS
================================================================================
Training run directory: `./save/steps_only_<timestamp>/`
Contents:
- model.pth / model_best.pth
- config.json (full configuration snapshot including `steps_cfg` and normalization)
- training_loss.png, val_future_rmse.png
- metrics.json (loss & validation history)
- input_highfreq_stats.json, input_highfreq_samples.png, input_highfreq_dwell_hist.png
Evaluation additions (same directory after eval):
- eval_incremental.png, incremental_rmse.json
- eval_half_forecast.png, half_forecast_metrics.json
- eval_multi_input_grid.png
- input_diversity_c_<value>.png

================================================================================
12. DESIGN RATIONALE
================================================================================
Rationale:
- Disjoint c splits ensure true generalization beyond training systems.
- High-frequency binary forcing tests model's ability to learn causal response under stochastic plateau durations.
- Mixed masking promotes robust forecasting rather than narrow reconstruction.
- Early stopping guided by near-critical dynamics (validation c values) stabilizes convergence.


================================================================================
END OF UPDATED DOCUMENTATION
================================================================================
