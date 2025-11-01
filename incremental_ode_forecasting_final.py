# TRUE INCREMENTAL/SEQUENTIAL ODE FORECASTING - FINAL VERSION
# Implements exactly what you requested: chronological time progression with realistic missing data

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Ensure we're in the correct directory
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import required modules
try:
    from main_model import CSDI_Forecasting
    print("✅ Successfully imported CSDI modules")
except ImportError as e:
    print(f"❌ Error importing CSDI modules: {e}")
    sys.exit(1)

# Device selection
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')
    print(f"⚠️  CUDA not available, using CPU")

print(f"Using device: {device}")

def generate_ode_trajectory(length, dt, regime, noise_std=0.02, is_training=False):
    """Generate ODE trajectory with separated parameter ranges"""
    m, k = 1.0, 1.0
    
    if is_training:
        # Training parameter ranges
        if regime == 'overdamped':
            c = np.random.uniform(2.5, 3.5)
        elif regime == 'critically_damped':
            c = np.random.uniform(1.8, 2.2)
        elif regime == 'underdamped':
            c = np.random.uniform(0.3, 0.7)
    else:
        # Testing parameter ranges (different from training)
        if regime == 'overdamped':
            c = np.random.uniform(3.6, 4.4)
        elif regime == 'critically_damped':
            c = np.random.uniform(2.3, 2.7)
        elif regime == 'underdamped':
            c = np.random.uniform(0.1, 0.25)
    
    x = np.zeros(length)
    v = np.zeros(length)
    x[0] = np.random.uniform(0.5, 1.5)
    v[0] = np.random.uniform(-0.3, 0.3)

    for i in range(1, length):
        a = -(c * v[i-1] + k * x[i-1]) / m
        v[i] = v[i-1] + a * dt
        x[i] = x[i-1] + v[i] * dt

    x += np.random.randn(length) * noise_std
    return x

def generate_forced_ode_trajectory(length, dt, regime, input_type="steps", noise_std=0.02, is_training=False):
    """Generate forced ODE trajectory with exogenous input u
    
    ODE: m*x'' + c*x' + k*x = d*u
    Rewritten as: a = (-c*v - k*x + d*u) / m
    
    Args:
        length: trajectory length
        dt: time step
        regime: damping regime ('overdamped', 'critically_damped', 'underdamped')
        input_type: type of forcing input ('steps', 'impulses', 'sine', 'random_hold')
        noise_std: noise added to x only
        is_training: whether generating training or test data
        
    Returns:
        x: state trajectory (length,)
        u: input trajectory (length,)
    """
    m, k = 1.0, 1.0
    d = np.random.uniform(0.8, 1.2)  # Forcing coefficient
    
    # Same damping parameter ranges as original
    if is_training:
        if regime == 'overdamped':
            c = np.random.uniform(2.5, 3.5)
        elif regime == 'critically_damped':
            c = np.random.uniform(1.8, 2.2)
        elif regime == 'underdamped':
            c = np.random.uniform(0.3, 0.7)
    else:
        if regime == 'overdamped':
            c = np.random.uniform(3.6, 4.4)
        elif regime == 'critically_damped':
            c = np.random.uniform(2.3, 2.7)
        elif regime == 'underdamped':
            c = np.random.uniform(0.1, 0.25)
    
    # Generate input u based on type
    u = np.zeros(length)
    
    if input_type == "steps":
        # Random piecewise constant
        n_segments = np.random.randint(3, 8)
        segment_length = length // n_segments
        for i in range(n_segments):
            start_idx = i * segment_length
            end_idx = min((i + 1) * segment_length, length)
            u[start_idx:end_idx] = np.random.uniform(-1.0, 1.0)
        # Handle remainder
        if end_idx < length:
            u[end_idx:] = u[end_idx-1]
            
    elif input_type == "impulses":
        # Sparse spikes
        n_impulses = np.random.randint(2, 6)
        impulse_indices = np.random.choice(length, size=n_impulses, replace=False)
        for idx in impulse_indices:
            u[idx] = np.random.uniform(-2.0, 2.0)
            
    elif input_type == "sine":
        # Sinusoidal with random freq/amp
        freq = np.random.uniform(0.02, 0.1)  # Frequency in cycles per time step
        amp = np.random.uniform(0.5, 1.5)
        phase = np.random.uniform(0, 2*np.pi)
        t = np.arange(length) * dt
        u = amp * np.sin(2 * np.pi * freq * t + phase)
        
    elif input_type == "random_hold":
        # Gaussian noise held over 3-8 steps
        hold_length = np.random.randint(3, 9)
        i = 0
        while i < length:
            value = np.random.normal(0, 0.5)
            end_idx = min(i + hold_length, length)
            u[i:end_idx] = value
            i = end_idx
            hold_length = np.random.randint(3, 9)  # New hold length for next segment
    
    # Initialize state
    x = np.zeros(length)
    v = np.zeros(length)
    x[0] = np.random.uniform(0.5, 1.5)
    v[0] = np.random.uniform(-0.3, 0.3)
    
    # Integrate forced ODE
    for i in range(1, length):
        a = (-c * v[i-1] - k * x[i-1] + d * u[i-1]) / m
        v[i] = v[i-1] + a * dt
        x[i] = x[i-1] + v[i] * dt
    
    # Add noise to x only (u should remain exact)
    x += np.random.randn(length) * noise_std
    
    return x, u

class IncrementalODEForecaster:
    """True incremental/sequential forecasting for ODE trajectories"""
    
    def __init__(self, model, device, scaler, mean_scaler):
        self.model = model.to(device)
        self.device = device
        self.scaler = scaler.to(device)
        self.mean_scaler = mean_scaler.to(device)

    def generate_sequential_observation_patterns(self, seq_length, n_days=3, missing_rate=0.2):
        """Generate TRUE incremental observation patterns - sequential through time
        
        Day 1: Observe first ~1/3 of sequence → predict rest
        Day 2: Observe next ~1/3 of sequence → predict rest  
        Day 3: Observe final ~1/3 of sequence → predict rest
        
        Missing_rate: fraction of points missing within each day's observation window
        """
        observations_by_day = []
        
        # Split sequence into roughly equal chunks for each day
        points_per_day = seq_length // n_days
        
        for day in range(n_days):
            obs_mask = np.zeros(seq_length, dtype=float)
            
            # Define the time window for this day
            start_time = day * points_per_day
            end_time = min((day + 1) * points_per_day, seq_length)
            
            # For the last day, include any remaining points
            if day == n_days - 1:
                end_time = seq_length
            
            day_length = end_time - start_time
            
            # Create observations for this day's time window
            day_indices = np.arange(start_time, end_time)
            
            # Randomly missing some points for realism
            n_missing = int(day_length * missing_rate)
            if n_missing > 0 and len(day_indices) > n_missing:
                missing_indices = np.random.choice(day_indices, size=n_missing, replace=False)
                observed_indices = np.setdiff1d(day_indices, missing_indices)
            else:
                observed_indices = day_indices
            
            # Mark observed points
            obs_mask[observed_indices] = 1.0
            
            # For incremental forecasting, we also want to preserve previous observations
            if day > 0:
                # Keep all previous observations
                prev_end = day * points_per_day
                obs_mask[:prev_end] = observations_by_day[-1][:prev_end]
            
            observations_by_day.append(obs_mask)
            
            n_observed_today = len(observed_indices)
            n_total_observed = int(obs_mask.sum())
            print(f"Day {day + 1}: +{n_observed_today} new observations (time {start_time}-{end_time-1}), "
                  f"total observed: {n_total_observed}/{seq_length} ({100*n_total_observed/seq_length:.1f}%)")
        
        return observations_by_day

    def create_batch_with_mask(self, trajectory_data, obs_mask):
        """Create batch with specific observation mask
        
        Args:
            trajectory_data: (T, K) array where K=1 for single channel or K=2 for [x, u]
            obs_mask: (T,) mask for x channel observations
        """
        T, K = trajectory_data.shape
        
        if K == 1:
            # Single channel (original behavior)
            if obs_mask.ndim == 1:
                gt_mask = obs_mask[:, None].astype(np.float32)
            else:
                gt_mask = obs_mask.astype(np.float32)
        elif K == 2:
            # Two channels: x (variable mask) and u (always observed)
            gt_mask = np.zeros((T, K), dtype=np.float32)
            gt_mask[:, 0] = obs_mask  # x channel uses observation pattern
            gt_mask[:, 1] = 1.0       # u channel is always fully observed
        else:
            raise ValueError(f"Unsupported number of channels: {K}")

        observed_mask_tensor = torch.from_numpy(gt_mask[None, :, :]).to(self.device)

        batch = {
            'observed_data': torch.from_numpy(trajectory_data[None, :, :]).to(self.device),
            'observed_mask': observed_mask_tensor,
            'gt_mask': observed_mask_tensor,
            'timepoints': torch.arange(T, dtype=torch.float32)[None, :].to(self.device),
            'feature_id': torch.tensor([[0]], dtype=torch.long).to(self.device),
        }
        return batch

    def safe_evaluate_with_nan_detection(self, batch, n_samples):
        """Evaluate with comprehensive NaN detection"""
        try:
            self.model.eval()
            self.model = self.model.to(self.device)
            
            with torch.inference_mode():
                # Check input batch for NaN
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        if torch.isnan(value).any():
                            print(f"❌ NaN detected in input batch['{key}']")
                            return None
                        if torch.isinf(value).any():
                            print(f"❌ Inf detected in input batch['{key}']")
                            return None
                
                # Evaluate model
                output = self.model.evaluate(batch, n_samples)
                samples, observed_data, target_mask, observed_mask_full, observed_tp = output
                
                # Check output for NaN
                if torch.isnan(samples).any():
                    print(f"❌ NaN detected in model output samples")
                    print(f"   samples shape: {samples.shape}")
                    print(f"   NaN count: {torch.isnan(samples).sum().item()}")
                    return None
                
                if torch.isinf(samples).any():
                    print(f"❌ Inf detected in model output samples")
                    return None
                
                return output
                
        except Exception as e:
            print(f"❌ Error in evaluation: {e}")
            import traceback
            traceback.print_exc()
            return None

    def forecast_incremental_sequential(self, trajectory_data, n_days=3, missing_rate=0.2, n_samples=3):
        """True incremental/sequential forecasting through time"""
        seq_length = len(trajectory_data)
        observation_schedule = self.generate_sequential_observation_patterns(seq_length, n_days, missing_rate)
        results = {}

        for day, obs_mask in enumerate(observation_schedule):
            day_key = f'day_{day + 1}'
            n_observed = int((obs_mask == 1.0).sum())
            n_targets = seq_length - n_observed
            
            # Calculate new observations for this day
            if day == 0:
                new_obs = n_observed
            else:
                prev_observed = int((observation_schedule[day-1] == 1.0).sum())
                new_obs = n_observed - prev_observed
            
            print(f"\nDay {day + 1}: {new_obs} new observations, {n_observed} total observed, {n_targets} to predict")

            # Create batch
            batch = self.create_batch_with_mask(trajectory_data, obs_mask)
            
            # Safe evaluation
            output = self.safe_evaluate_with_nan_detection(batch, n_samples)
            
            if output is None:
                print(f"❌ Failed to get valid output for day {day + 1}")
                results[day_key] = {
                    'predictions': None,
                    'observation_pattern': obs_mask,
                    'n_observed': n_observed,
                    'n_targets': n_targets,
                    'new_observations': new_obs,
                    'success': False
                }
                continue
            
            samples, observed_data, target_mask, observed_mask_full, observed_tp = output
            
            results[day_key] = {
                'predictions': samples.cpu().numpy(),
                'observed_data': observed_data.cpu().numpy(),
                'target_mask': target_mask.cpu().numpy(),
                'observed_mask_full': observed_mask_full.cpu().numpy(),
                'observation_pattern': obs_mask,
                'n_observed': n_observed,
                'n_targets': n_targets,
                'new_observations': new_obs,
                'success': True
            }

        return results

def safe_calculate_metrics(predictions, true_data, obs_pattern, scaler_val, mean_scaler_val, is_multichannel=False):
    """Calculate metrics with comprehensive NaN handling
    
    Args:
        predictions: model predictions
        true_data: ground truth data (for multichannel, only x channel is evaluated)
        obs_pattern: observation pattern (for x channel)
        scaler_val: scaling factor
        mean_scaler_val: mean offset
        is_multichannel: if True, only evaluate channel 0 (x)
    """
    try:
        # Handle predictions shape
        if predictions is None:
            return float('nan'), float('nan')
        
        if len(predictions.shape) == 4 and predictions.shape[0] == 1:
            pred_array = predictions[0, :, :, :]  # (n_samples, K, T)
        else:
            print(f"❌ Unexpected predictions shape: {predictions.shape}")
            return float('nan'), float('nan')
        
        # For multichannel, extract only the x channel (channel 0)
        if is_multichannel and pred_array.shape[1] >= 2:
            pred_array = pred_array[:, 0, :]  # Extract x channel: (n_samples, T)
        elif not is_multichannel:
            # Single channel case
            if pred_array.shape[1] == 1:
                pred_array = pred_array[:, 0, :]  # (n_samples, T)
            else:
                print(f"❌ Expected single channel but got {pred_array.shape[1]} channels")
                return float('nan'), float('nan')
        
        # Check for NaN in predictions
        if np.isnan(pred_array).any():
            print(f"❌ NaN in prediction array")
            return float('nan'), float('nan')
        
        # Calculate median prediction
        pred_median = np.median(pred_array, axis=0)  # (T,)
        
        # Convert to original space
        pred_median_orig = pred_median * scaler_val + mean_scaler_val
        
        # Check for NaN after conversion
        if np.isnan(pred_median_orig).any():
            print(f"❌ NaN after denormalization")
            return float('nan'), float('nan')
        
        # Calculate metrics
        obs_indices = (obs_pattern == 1.0)
        target_indices = (obs_pattern != 1.0)
        
        # Conditioning error
        if np.any(obs_indices):
            cond_error = np.abs(pred_median_orig[obs_indices] - true_data[obs_indices]).mean()
            if np.isnan(cond_error):
                print(f"❌ NaN in conditioning error calculation")
                cond_error = float('nan')
        else:
            cond_error = 0.0
        
        # Target RMSE
        if np.any(target_indices):
            target_diff = pred_median_orig[target_indices] - true_data[target_indices]
            if np.isnan(target_diff).any():
                print(f"❌ NaN in target difference calculation")
                target_rmse = float('nan')
            else:
                target_rmse = np.sqrt(np.mean(target_diff ** 2))
                if np.isnan(target_rmse):
                    print(f"❌ NaN in RMSE calculation")
                    target_rmse = float('nan')
        else:
            target_rmse = 0.0
        
        return cond_error, target_rmse
        
    except Exception as e:
        print(f"❌ Error in metrics calculation: {e}")
        return float('nan'), float('nan')

def evaluate_and_visualize_results(results, true_data, regime_name, scaler_val, mean_scaler_val, true_input=None, p_low=5, p_high=95):
    """Evaluate and visualize incremental forecasting results
    
    Args:
        results: forecasting results
        true_data: ground truth x trajectory  
        regime_name: name of the regime
        scaler_val: scaling factor for x
        mean_scaler_val: mean offset for x
        true_input: ground truth u trajectory (optional, for validation)
    """
    
    # Determine if this is multichannel
    is_multichannel = true_input is not None
    
    # Create visualization
    n_days = len(results)
    fig_height = 8 if is_multichannel else 6
    fig, axes = plt.subplots(2 if is_multichannel else 1, n_days, figsize=(5 * n_days, fig_height))
    
    if n_days == 1:
        if is_multichannel:
            axes = axes[:, None]  # Make it 2D
        else:
            axes = [axes]
    elif not is_multichannel:
        axes = [axes]  # Wrap in list for consistent indexing
    
    performance = {}
    
    for day_idx, (day_key, day_results) in enumerate(results.items()):
        if is_multichannel:
            ax_x = axes[0, day_idx]  # x channel plot
            ax_u = axes[1, day_idx]  # u channel plot
        else:
            ax_x = axes[day_idx] if n_days > 1 else axes[0]
        
        if not day_results['success']:
            ax_x.text(0.5, 0.5, f'{day_key}\nFAILED\n{day_results.get("error", "Unknown")}', 
                     horizontalalignment='center', verticalalignment='center', 
                     transform=ax_x.transAxes, fontsize=12, color='red')
            ax_x.set_title(f'{regime_name} - {day_key} - FAILED')
            
            if is_multichannel:
                ax_u.text(0.5, 0.5, 'FAILED', horizontalalignment='center', 
                         verticalalignment='center', transform=ax_u.transAxes, 
                         fontsize=12, color='red')
            
            performance[day_key] = {
                'conditioning_error': float('nan'),
                'target_rmse': float('nan'),
                'success': False
            }
            continue
        
        # Extract predictions
        predictions_raw = day_results['predictions']
        obs_pattern = day_results['observation_pattern']
        
        # Handle prediction shape
        if len(predictions_raw.shape) == 4 and predictions_raw.shape[0] == 1:
            if predictions_raw.shape[2] >= 1:
                predictions_x = predictions_raw[0, :, 0, :]  # x channel
                if is_multichannel and predictions_raw.shape[2] >= 2:
                    predictions_u = predictions_raw[0, :, 1, :]  # u channel
            else:
                continue
        else:
            continue

        # Calculate statistics for x channel
        pred_mean_x = np.mean(predictions_x, axis=0)
        pred_median_x = np.median(predictions_x, axis=0)
        pred_low_x = np.percentile(predictions_x, p_low, axis=0)
        pred_high_x = np.percentile(predictions_x, p_high, axis=0)

        # Convert x to original space
        pred_mean_x_orig = pred_mean_x * scaler_val + mean_scaler_val
        pred_median_x_orig = pred_median_x * scaler_val + mean_scaler_val
        pred_low_x_orig = pred_low_x * scaler_val + mean_scaler_val
        pred_high_x_orig = pred_high_x * scaler_val + mean_scaler_val

        # Plot x channel
        t = np.arange(len(true_data))
        ax_x.fill_between(t, pred_low_x_orig, pred_high_x_orig, color='tab:green', alpha=0.18, label=f'{p_low}-{p_high}% band')
        ax_x.plot(t, pred_mean_x_orig, color='tab:green', linewidth=2, label='Mean pred')
        ax_x.plot(t, pred_median_x_orig, color='tab:olive', linewidth=1.2, linestyle='--', label='Median')
        ax_x.plot(t, true_data, 'k-', alpha=0.6, label='True x', linewidth=1)
        
        # Show observations for x
        obs_indices = obs_pattern == 1.0
        target_indices = ~obs_indices
        
        # Calculate which observations are new for this day
        if day_idx == 0:
            new_obs_indices = obs_indices
        else:
            prev_day_key = f'day_{day_idx}'
            if prev_day_key in results:
                prev_obs_pattern = results[prev_day_key]['observation_pattern']
                prev_obs_indices = prev_obs_pattern == 1.0
                new_obs_indices = obs_indices & ~prev_obs_indices
            else:
                new_obs_indices = obs_indices
        
        # Plot observations for x
        if np.any(obs_indices):
            ax_x.plot(t[obs_indices], true_data[obs_indices], 'ro', markersize=4, label='x Observed', alpha=0.6)
        
        if np.any(new_obs_indices):
            ax_x.plot(t[new_obs_indices], true_data[new_obs_indices], 'ro', markersize=6, 
                     label=f'New x Day {day_idx+1}', markeredgecolor='darkred', markeredgewidth=2)
        
        if np.any(target_indices):
            ax_x.plot(t[target_indices], true_data[target_indices], 'bo', markersize=2, label='x Targets', alpha=0.5)
        
        # Calculate metrics for x channel only
        cond_error, target_rmse = safe_calculate_metrics(
            predictions_raw, true_data, obs_pattern, scaler_val, mean_scaler_val, is_multichannel=is_multichannel
        )
        
        # Title for x channel
        new_obs_count = np.sum(new_obs_indices)
        total_obs_count = np.sum(obs_indices)
        target_count = np.sum(target_indices)
        
        ax_x.set_title(f'{regime_name} - {day_key} - x channel\n'
                      f'+{new_obs_count} new obs (total: {total_obs_count}), {target_count} targets\n'
                      f'Cond: {cond_error:.2e}, RMSE: {target_rmse:.3f}')
        ax_x.legend(fontsize=8)
        ax_x.grid(True, alpha=0.3)
        
        # Plot u channel if multichannel
        if is_multichannel:
            # u channel predictions (should match true_input exactly)
            pred_median_u = np.median(predictions_u, axis=0)
            
            ax_u.plot(t, pred_median_u, 'b-', label='u Prediction', linewidth=2)
            ax_u.plot(t, true_input, 'k-', alpha=0.6, label='True u', linewidth=1)
            
            # Validation: check if u predictions match inputs
            u_error = np.abs(pred_median_u - true_input).mean()
            ax_u.set_title(f'{day_key} - u channel (always observed)\nMean error: {u_error:.2e}')
            ax_u.legend(fontsize=8)
            ax_u.grid(True, alpha=0.3)
            
            # Validation check
            if u_error > 1e-3:
                print(f"⚠️  Warning: u prediction error {u_error:.2e} is high for {day_key}")
        
        performance[day_key] = {
            'conditioning_error': cond_error,
            'target_rmse': target_rmse,
            'n_observed': total_obs_count,
            'n_new_observed': new_obs_count,
            'n_targets': target_count,
            'success': True
        }
        
        if is_multichannel:
            performance[day_key]['u_error'] = u_error
    
    title_suffix = " with Exogenous Input" if is_multichannel else ""
    plt.suptitle(f'{regime_name} - TRUE Incremental/Sequential ODE Forecasting{title_suffix}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return performance

def parse_args():
    ap = argparse.ArgumentParser(description='Incremental ODE forecasting with uncertainty bands')
    ap.add_argument('--n-samples', type=int, default=10, help='Number of diffusion samples per evaluation')
    ap.add_argument('--p-low', type=float, default=5, help='Lower percentile for band')
    ap.add_argument('--p-high', type=float, default=95, help='Upper percentile for band')
    ap.add_argument('--missing-rate', type=float, default=0.2, help='Within-window missing rate for incremental schedule')
    ap.add_argument('--seq-length', type=int, default=120, help='Sequence length for test trajectories')
    ap.add_argument('--save-dir', type=str, default='', help='Directory to save plots (if empty, auto-create under save/)')
    ap.add_argument('--steps-only', action='store_true', help='Restrict evaluation to step inputs only')
    # Extended multi-system / sparse evaluation arguments
    ap.add_argument('--systems-per-regime', type=int, default=1, help='Number of distinct (c,d) systems to evaluate per regime ( >1 activates multi-system mode when steps-only)')
    ap.add_argument('--variants-per-system', type=int, default=1, help='Number of distinct step input variants per system')
    ap.add_argument('--sparse-missing-rate', type=float, default=0.5, help='Missing rate for sparse (non-incremental) evaluation')
    ap.add_argument('--sparse-repeats', type=int, default=1, help='Number of random sparse masks per variant')
    ap.add_argument('--noise-std-eval', type=float, default=0.02, help='Observation noise std during synthetic evaluation data generation')
    ap.add_argument('--shift-d-range', action='store_true', help='Shift forcing coefficient d range to [1.25,1.5] (away from assumed training range ~[0.8,1.2])')
    ap.add_argument('--force-device', type=str, default='', help='Force device selection: cpu or cuda')
    ap.add_argument('--no-rerun-top', action='store_true', help='Reuse cached incremental predictions for top variants (skip re-sampling)')
    return ap.parse_args()

# ---------------- Extended Multi-System Evaluation Utilities (defined before main) ---------------- #

def sample_c_for_regime(regime, is_training=False):
    if is_training:
        if regime == 'overdamped':
            return np.random.uniform(2.5, 3.5)
        if regime == 'critically_damped':
            return np.random.uniform(1.8, 2.2)
        if regime == 'underdamped':
            return np.random.uniform(0.3, 0.7)
    else:
        if regime == 'overdamped':
            return np.random.uniform(3.6, 4.4)
        if regime == 'critically_damped':
            return np.random.uniform(2.3, 2.7)
        if regime == 'underdamped':
            return np.random.uniform(0.1, 0.25)
    raise ValueError('Unknown regime')

def integrate_forced_system(c, d, u, dt=0.1, noise_std=0.02):
    m = 1.0; k = 1.0
    length = len(u)
    x = np.zeros(length); v = np.zeros(length)
    x[0] = np.random.uniform(0.5, 1.5)
    v[0] = np.random.uniform(-0.3, 0.3)
    for i in range(1, length):
        a = (-c * v[i-1] - k * x[i-1] + d * u[i-1]) / m
        v[i] = v[i-1] + a * dt
        x[i] = x[i-1] + v[i] * dt
    x += np.random.randn(length) * noise_std
    return x

def generate_step_input_variant(T, rng):
    n_segments = rng.integers(4, 9)
    change_points = sorted(rng.choice(np.arange(1, T-1), size=n_segments-1, replace=False)) if n_segments>1 else []
    change_points = [0] + change_points + [T]
    u = np.zeros(T)
    for i in range(n_segments):
        start, end = change_points[i], change_points[i+1]
        level = rng.uniform(-1.0, 1.0)
        u[start:end] = level
    return u

def compute_percentile_stats(samples_array, low_list, high_list):
    stats = {}
    for lo, hi in zip(low_list, high_list):
        stats[(lo, hi)] = (np.percentile(samples_array, lo, axis=0), np.percentile(samples_array, hi, axis=0))
    return stats

def forecast_sparse_variant(forecaster, trajectory_data, missing_rate, n_samples):
    T, K = trajectory_data.shape
    rng = np.random.default_rng()
    obs_mask = np.ones(T, dtype=float)
    missing_indices = rng.choice(T, size=int(T * missing_rate), replace=False)
    obs_mask[missing_indices] = 0.0
    batch = forecaster.create_batch_with_mask(trajectory_data, obs_mask)
    output = forecaster.safe_evaluate_with_nan_detection(batch, n_samples)
    if output is None:
        return None
    samples, observed_data, target_mask, observed_mask_full, observed_tp = output
    return {
        'samples': samples.cpu().numpy(),
        'obs_mask': obs_mask,
        'target_mask': target_mask.cpu().numpy()
    }

def compute_coverage_and_bandwidth(truth, percentiles_dict, scaler_val, mean_val, bands, target_indices):
    results = {}
    truth_orig = truth * scaler_val + mean_val
    for (lo, hi) in bands:
        low_arr, high_arr = percentiles_dict[(lo, hi)]
        low_orig = low_arr * scaler_val + mean_val
        high_orig = high_arr * scaler_val + mean_val
        width = high_orig - low_orig
        if np.any(target_indices):
            cover = np.mean((truth_orig[target_indices] >= low_orig[target_indices]) & (truth_orig[target_indices] <= high_orig[target_indices]))
            width_mean = np.mean(width[target_indices])
        else:
            cover = np.nan; width_mean = np.nan
        results[(lo, hi)] = {'coverage': float(cover), 'mean_width': float(width_mean)}
    return results

def run_multi_system_step_only_evaluation(args, forecaster, x_mean, x_std, u_mean, u_std):
    import time
    regimes = ['overdamped', 'critically_damped', 'underdamped']
    rng_master = np.random.default_rng(12345)
    if getattr(args, 'save_dir', ''):
        base_save_dir = args.save_dir
    else:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        base_save_dir = os.path.join('save', f'multisystem_eval_{timestamp}')
    os.makedirs(base_save_dir, exist_ok=True)
    print(f"📁 Multi-system results will be saved to: {base_save_dir}")

    systems_per_regime = args.systems_per_regime
    variants_per_system = args.variants_per_system
    bands = [(5,95),(10,90)]  # Outer then inner
    metrics_all = {'config':{
        'systems_per_regime': systems_per_regime,
        'variants_per_system': variants_per_system,
        'seq_length': args.seq_length,
        'incremental_missing_rate': args.missing_rate,
        'sparse_missing_rate': args.sparse_missing_rate,
        'n_samples': args.n_samples,
        'bands': bands,
        'd_shifted': bool(args.shift_d_range)
    }, 'regimes':{}}
    d_low, d_high = (1.25, 1.5) if args.shift_d_range else (0.8, 1.2)

    for regime in regimes:
        print(f"\n===== Regime: {regime} =====")
        metrics_all['regimes'][regime] = {'systems': []}
        regime_system_records = []
        for sys_idx in range(systems_per_regime):
            c_val = sample_c_for_regime(regime, is_training=False)
            d_val = rng_master.uniform(d_low, d_high)
            # system_record keeps rich (non-JSON) data; we'll serialize later
            system_record = {
                'system_id': sys_idx,
                'c': float(c_val),
                'd': float(d_val),
                'variants': [],            # metrics only (serializable)
                'sparse_variants': [],      # metrics only (serializable)
                'variant_data': {}          # rich data for later re-eval/plot (NOT serialized)
            }
            print(f" System {sys_idx}: c={c_val:.4f}, d={d_val:.4f}")
            for var_idx in range(variants_per_system):
                var_seed = 1000 + sys_idx*100 + var_idx
                rng_variant = np.random.default_rng(var_seed)
                u = generate_step_input_variant(args.seq_length, rng_variant)
                x = integrate_forced_system(c_val, d_val, u, dt=0.1, noise_std=args.noise_std_eval)
                x_norm = (x - x_mean) / x_std; u_norm = (u - u_mean) / u_std
                trajectory = np.stack([x_norm, u_norm], axis=1).astype(np.float32)
                results_inc = forecaster.forecast_incremental_sequential(trajectory, n_days=3, missing_rate=args.missing_rate, n_samples=args.n_samples)
                variant_metrics = {'variant_id': var_idx, 'day_metrics': {}}
                # store minimal data for potential later plotting (avoid huge sample tensors)
                system_record['variant_data'][var_idx] = {
                    'trajectory': trajectory,  # normalized
                    'x_orig': x,               # original scale
                    'u_orig': u,               # original scale
                    'seed': var_seed,
                    'cached_results': results_inc
                }
                for day_key, day_res in results_inc.items():
                    if not day_res['success']:
                        variant_metrics['day_metrics'][day_key] = {'success': False}
                        continue
                    preds = day_res['predictions']
                    preds_x = preds[0, :, 0, :]
                    percentiles = compute_percentile_stats(preds_x, [5,10],[95,90])
                    obs_pattern = day_res['observation_pattern']
                    target_idx = obs_pattern != 1.0
                    coverage = compute_coverage_and_bandwidth(x_norm, percentiles, x_std, x_mean, bands, target_idx)
                    median = np.median(preds_x, axis=0); median_orig = median * x_std + x_mean; x_orig = x
                    if np.any(target_idx):
                        rmse = float(np.sqrt(np.mean((median_orig[target_idx] - x_orig[target_idx])**2)))
                    else:
                        rmse = float('nan')
                    variant_metrics['day_metrics'][day_key] = {
                        'success': True,
                        'rmse': rmse,
                        'coverage': {f'{lo}-{hi}': coverage[(lo,hi)]['coverage'] for (lo,hi) in bands},
                        'mean_width': {f'{lo}-{hi}': coverage[(lo,hi)]['mean_width'] for (lo,hi) in bands}
                    }
                system_record['variants'].append(variant_metrics)
                sparse_list = []
                for rep in range(args.sparse_repeats):
                    sparse_out = forecast_sparse_variant(forecaster, trajectory, args.sparse_missing_rate, args.n_samples)
                    if sparse_out is None:
                        continue
                    samples = sparse_out['samples']
                    if len(samples.shape)==4 and samples.shape[0]==1:
                        preds_x_sparse = samples[0,:,0,:]
                    elif len(samples.shape)==4:
                        preds_x_sparse = samples[:,0,0,:]
                    else:
                        continue
                    percentiles_sparse = compute_percentile_stats(preds_x_sparse, [5,10],[95,90])
                    obs_mask = sparse_out['obs_mask']
                    target_idx_sparse = obs_mask != 1.0
                    coverage_sparse = compute_coverage_and_bandwidth(x_norm, percentiles_sparse, x_std, x_mean, bands, target_idx_sparse)
                    median_s = np.median(preds_x_sparse, axis=0); median_s_orig = median_s * x_std + x_mean
                    if np.any(target_idx_sparse):
                        rmse_sparse = float(np.sqrt(np.mean((median_s_orig[target_idx_sparse] - x[target_idx_sparse])**2)))
                    else:
                        rmse_sparse = float('nan')
                    sparse_list.append({'rep': rep, 'rmse': rmse_sparse, 'coverage': {f'{lo}-{hi}': coverage_sparse[(lo,hi)]['coverage'] for (lo,hi) in bands}, 'mean_width': {f'{lo}-{hi}': coverage_sparse[(lo,hi)]['mean_width'] for (lo,hi) in bands}})
                system_record['sparse_variants'].append({'variant_id': var_idx, 'sparse': sparse_list})
            day3_rmses = []
            for v in system_record['variants']:
                dm = v['day_metrics'].get('day_3', {})
                if dm.get('success') and dm.get('rmse') is not None:
                    day3_rmses.append(dm['rmse'])
            system_record['aggregate'] = {'day3_rmse_mean': float(np.mean(day3_rmses)) if day3_rmses else float('nan'), 'day3_rmse_std': float(np.std(day3_rmses)) if day3_rmses else float('nan')}
            regime_system_records.append(system_record)
        # Rank systems by mean Day 3 RMSE and select top 3
        ranked = sorted(regime_system_records, key=lambda s: (np.inf if np.isnan(s['aggregate']['day3_rmse_mean']) else s['aggregate']['day3_rmse_mean']))
        selected_system_ids = [s['system_id'] for s in ranked[:3]]

        # For each system, rank its variants for later selection
        for rec in regime_system_records:
            variant_rank_list = []
            for v in rec['variants']:
                day3 = v['day_metrics'].get('day_3', {})
                day2 = v['day_metrics'].get('day_2', {})
                rmse3 = day3.get('rmse', np.inf if not day3.get('success') else np.nan)
                rmse2 = day2.get('rmse', np.inf if not day2.get('success') else np.nan)
                # Prepare tuple for sorting (primary Day3 RMSE then Day2 RMSE then variant id)
                if np.isnan(rmse3):
                    sort_val3 = np.inf
                else:
                    sort_val3 = rmse3
                if np.isnan(rmse2):
                    sort_val2 = np.inf
                else:
                    sort_val2 = rmse2
                variant_rank_list.append((sort_val3, sort_val2, v['variant_id']))
            variant_rank_list.sort()
            rec['variant_ranking'] = [vid for _,__,vid in variant_rank_list]
            # Pick top 5 (or fewer if not available)
            rec['top_variants_for_plot'] = rec['variant_ranking'][:min(5, len(rec['variant_ranking']))]
            rec['selected_for_plot'] = rec['system_id'] in selected_system_ids

        # Produce full uncertainty plots (dual bands) for selected systems (only best 5 variants)
        for rec in regime_system_records:
            if not rec['selected_for_plot']:
                continue
            top_vars = rec['top_variants_for_plot']
            print(f"   ▶ Plotting system {rec['system_id']} top variants: {top_vars}")
            n_rows = len(top_vars)
            fig, axes = plt.subplots(n_rows, 3, figsize=(13, 2.6*n_rows))
            if n_rows == 1:
                axes = np.array([axes])  # make 2D for unified indexing
            fig.suptitle(f'{regime} system {rec["system_id"]}  c={rec["c"]:.3f} d={rec["d"]:.3f}  (Top {n_rows} variants)', fontsize=14, fontweight='bold')

            for row, var_id in enumerate(top_vars):
                # Retrieve stored trajectory & original x/u
                vdata = rec['variant_data'][var_id]
                trajectory = vdata['trajectory']
                x_orig = vdata['x_orig']
                u_orig = vdata['u_orig']
                # Re-run incremental forecasting unless reuse flag
                if args.no_rerun_top and 'cached_results' in vdata:
                    inc_results = vdata['cached_results']
                else:
                    inc_results = forecaster.forecast_incremental_sequential(trajectory, n_days=3, missing_rate=args.missing_rate, n_samples=args.n_samples)
                day_order = ['day_1','day_2','day_3']
                for col, day_key in enumerate(day_order):
                    ax = axes[row, col]
                    day_res = inc_results.get(day_key, {})
                    if not day_res or not day_res.get('success'):
                        ax.text(0.5,0.5,'FAIL',ha='center',va='center',color='red',fontsize=10)
                        ax.set_axis_off()
                        continue
                    preds = day_res['predictions']  # shape (1, n_samples, K, T)
                    if len(preds.shape)==4 and preds.shape[0]==1:
                        preds_x = preds[0,:,0,:]
                    else:
                        ax.text(0.5,0.5,'BAD SHAPE',ha='center',va='center',color='red',fontsize=8)
                        ax.set_axis_off(); continue
                    obs_pattern = day_res['observation_pattern']
                    # Percentiles
                    p5 = np.percentile(preds_x,5,axis=0); p95 = np.percentile(preds_x,95,axis=0)
                    p10 = np.percentile(preds_x,10,axis=0); p90 = np.percentile(preds_x,90,axis=0)
                    mean_pred = np.mean(preds_x,axis=0); median_pred = np.median(preds_x,axis=0)
                    # Denorm
                    p5o = p5 * x_std + x_mean; p95o = p95 * x_std + x_mean
                    p10o = p10 * x_std + x_mean; p90o = p90 * x_std + x_mean
                    meano = mean_pred * x_std + x_mean; mediano = median_pred * x_std + x_mean
                    t = np.arange(len(x_orig))
                    ax.fill_between(t,p5o,p95o,color='tab:orange',alpha=0.14,label='5-95%')
                    ax.fill_between(t,p10o,p90o,color='tab:orange',alpha=0.30,label='10-90%')
                    ax.plot(t, meano, color='tab:orange', lw=2, label='Mean')
                    ax.plot(t, mediano, color='tab:red', lw=1.2, ls='--', label='Median')
                    ax.plot(t, x_orig, 'k-', lw=1.1, alpha=0.7, label='True x')
                    obs_idx = obs_pattern==1.0; tgt_idx = ~obs_idx
                    if np.any(obs_idx):
                        ax.plot(t[obs_idx], x_orig[obs_idx],'go',ms=3,label='Observed',alpha=0.75)
                    if np.any(tgt_idx):
                        ax.plot(t[tgt_idx], x_orig[tgt_idx],'bo',ms=2,label='Targets',alpha=0.55)
                    # Metrics recompute (RMSE Day target region)
                    if np.any(tgt_idx):
                        rmse = np.sqrt(np.mean((mediano[tgt_idx]-x_orig[tgt_idx])**2))
                    else:
                        rmse = np.nan
                    title = f'Var {var_id} {day_key}\nRMSE={rmse:.3f} Obs={obs_idx.sum()} Tgt={tgt_idx.sum()}'
                    ax.set_title(title, fontsize=9)
                    ax.grid(alpha=0.3)
                    if row==0 and col==0:
                        ax.legend(fontsize=7, ncol=2)
            plt.tight_layout()
            out_path = os.path.join(base_save_dir, f'{regime}_system{rec["system_id"]}_top_variants.png')
            plt.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"   💾 Saved full forecast plots: {out_path}")

        # After plotting, prepare serializable records for metrics JSON
        for rec in regime_system_records:
            serializable = {
                'system_id': rec['system_id'],
                'c': rec['c'],
                'd': rec['d'],
                'variants': rec['variants'],
                'sparse_variants': rec['sparse_variants'],
                'aggregate': rec['aggregate'],
                'selected_for_plot': rec['selected_for_plot'],
                'variant_ranking': rec.get('variant_ranking', []),
                'top_variants_for_plot': rec.get('top_variants_for_plot', [])
            }
            metrics_all['regimes'][regime]['systems'].append(serializable)
        sparse_rmses = []
        for rec in regime_system_records:
            for sv in rec['sparse_variants']:
                for rep in sv['sparse']:
                    sparse_rmses.append(rep['rmse'])
        if sparse_rmses:
            fig2, ax2 = plt.subplots(figsize=(5,3))
            ax2.hist([r for r in sparse_rmses if not np.isnan(r)], bins=12, color='steelblue', alpha=0.8)
            ax2.set_title(f'{regime} Sparse RMSE Distribution')
            ax2.set_xlabel('RMSE'); ax2.set_ylabel('Count')
            out_path_hist = os.path.join(base_save_dir, f'{regime}_sparse_rmse_hist.png')
            plt.tight_layout(); plt.savefig(out_path_hist, dpi=140); plt.close(fig2)
            print(f"   💾 Saved sparse RMSE hist: {out_path_hist}")

    import json, csv
    metrics_path = os.path.join(base_save_dir, 'metrics_multisystem.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_all, f, indent=2)
    print(f"📄 Saved metrics JSON: {metrics_path}")
    csv_path = os.path.join(base_save_dir, 'system_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['regime','system_id','c','d','day3_rmse_mean','day3_rmse_std','selected_for_plot'])
        for regime, data in metrics_all['regimes'].items():
            for sys_rec in data['systems']:
                writer.writerow([regime, sys_rec['system_id'], sys_rec['c'], sys_rec['d'], sys_rec['aggregate']['day3_rmse_mean'], sys_rec['aggregate']['day3_rmse_std'], sys_rec['selected_for_plot']])
    print(f"📄 Saved system summary CSV: {csv_path}")
    print("✅ Multi-system step-only evaluation completed.")


def main():
    args = parse_args()
    """Main function for incremental ODE forecasting with exogenous inputs"""
    print("🚀 TRUE INCREMENTAL/SEQUENTIAL FORCED ODE FORECASTING")
    print("=" * 70)
    print("Day 1: Observe first ~1/3 of x → predict rest")
    print("Day 2: Observe next ~1/3 of x → predict rest") 
    print("Day 3: Observe final ~1/3 of x → predict rest")
    print("Input u is ALWAYS fully observed (exogenous)")
    print("Each day includes realistic missing observations for x only")
    print("Sample settings: n_samples={}, band {}-{}%, missing_rate={}".format(
        args.n_samples, args.p_low, args.p_high, args.missing_rate))
    print("=" * 70)
    
    # Check for 2-channel forced ODE model first
    import glob
    forced_model_dirs = glob.glob('./save/forced_ode_2ch_*')
    if forced_model_dirs:
        forced_model_dirs.sort(reverse=True)
        model_path = os.path.join(forced_model_dirs[0], 'model.pth')
        config_path = os.path.join(forced_model_dirs[0], 'config.json')
        if os.path.exists(model_path) and os.path.exists(config_path):
            print(f"✅ Found 2-channel forced ODE model at: {model_path}")
        else:
            print("❌ No valid 2-channel forced ODE model found!")
            return
    else:
        print("❌ No 2-channel forced ODE model found! Run train_forced_ode_2channel.py first.")
        return
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    x_mean = config['x_mean']; x_std = config['x_std']
    u_mean = config['u_mean']; u_std = config['u_std']
    target_dim = 2
    print(f"✅ Using 2-channel forced ODE model")
    print(f"   X normalization: mean={x_mean:.4f}, std={x_std:.4f}")
    print(f"   U normalization: mean={u_mean:.4f}, std={u_std:.4f}")
    config['model']['target_dim'] = 2
    model = CSDI_Forecasting(config, device, target_dim=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()
    x_scaler = torch.tensor([x_std], dtype=torch.float32, device=device)
    x_mean_scaler = torch.tensor([x_mean], dtype=torch.float32, device=device)
    forecaster = IncrementalODEForecaster(model, device, x_scaler, x_mean_scaler)

    # Visualize available input types unless in multi-system step-only mode
    if not (args.steps_only and args.systems_per_regime > 1):
        visualize_input_types()

    # If multi-system step-only mode requested
    if args.steps_only and args.systems_per_regime > 1:
        run_multi_system_step_only_evaluation(args, forecaster, x_mean, x_std, u_mean, u_std)
        return

    # Legacy single-pass evaluation across input types
    regimes = ['overdamped', 'critically_damped', 'underdamped']
    input_types = ['steps', 'impulses', 'sine', 'random_hold']
    if args.steps_only:
        input_types = ['steps']
        print('🔧 Steps-only mode enabled: other input types skipped.')
    all_results = {}
    for regime in regimes:
        all_results[regime] = {}
        for input_type in input_types:
            print(f"\n=== Processing {regime.replace('_',' ').title()} with {input_type} input ===")
            np.random.seed(42 + hash(regime + input_type) % 1000)
            seq_length = args.seq_length
            x_raw, u_raw = generate_forced_ode_trajectory(seq_length, dt=0.1, regime=regime, input_type=input_type, is_training=False)
            x_normalized = (x_raw - x_mean) / x_std
            u_normalized = (u_raw - u_mean) / u_std
            trajectory_data = np.stack([x_normalized, u_normalized], axis=1).astype(np.float32)
            print(f"x range: [{x_raw.min():.3f}, {x_raw.max():.3f}]")
            print(f"u range: [{u_raw.min():.3f}, {u_raw.max():.3f}] ({input_type})")
            print(f"     [Eval] Sampling n={args.n_samples} ...")
            results = forecaster.forecast_incremental_sequential(trajectory_data, n_days=3, missing_rate=args.missing_rate, n_samples=args.n_samples)
            all_results[regime][input_type] = {
                'results': results,
                'x_raw': x_raw,
                'u_raw': u_raw,
                'u_normalized': u_normalized,
                'x_std': x_std,
                'x_mean': x_mean
            }
            print('Quick Results:')
            for day_key, day_results in results.items():
                if day_results['success']:
                    n_new = day_results['new_observations']; n_total = day_results['n_observed']; n_targets = day_results['n_targets']
                    predictions_raw = day_results['predictions']; obs_pattern = day_results['observation_pattern']
                    cond_error, target_rmse = safe_calculate_metrics(predictions_raw, x_raw, obs_pattern, x_std, x_mean, is_multichannel=True)
                    print(f"  {day_key}: +{n_new} new obs, Total={n_total}, Targets={n_targets}, Cond={cond_error:.2e}, RMSE={target_rmse:.3f}")
                else:
                    print(f"  {day_key}: FAILED")

    # Determine save directory
    import time
    if args.save_dir:
        save_dir = args.save_dir
    else:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        save_dir = os.path.join('save', f'incremental_eval_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    print(f"📁 Saving figures to: {save_dir}")

    # Grouped visualization by damping regime
    print("\n🎨 Creating grouped visualizations by damping regime...")
    p_low = args.p_low; p_high = args.p_high
    for regime in regimes:
        regime_title = regime.replace('_',' ').title()
        print(f"\n📊 Displaying all {regime_title} results...")
        n_input_types = len(input_types); n_days = 3
        fig, axes = plt.subplots(n_input_types, n_days, figsize=(5*n_days, 4*n_input_types))
        for input_idx, input_type in enumerate(input_types):
            regime_data = all_results[regime][input_type]
            results = regime_data['results']; x_raw = regime_data['x_raw']; u_normalized = regime_data['u_raw'] if 'u_raw' in regime_data else regime_data['u_normalized']
            x_std_local = regime_data['x_std']; x_mean_local = regime_data['x_mean']
            for day_idx, (day_key, day_results) in enumerate(results.items()):
                ax = axes[input_idx, day_idx]
                if not day_results['success']:
                    ax.text(0.5,0.5,f'{day_key}\nFAILED',ha='center',va='center',transform=ax.transAxes,fontsize=12,color='red')
                    ax.set_title(f'{input_type} - {day_key} - FAILED')
                    continue
                predictions_raw = day_results['predictions']; obs_pattern = day_results['observation_pattern']
                if len(predictions_raw.shape)==4 and predictions_raw.shape[0]==1:
                    predictions = predictions_raw[0,:,0,:]
                else:
                    continue
                pred_mean = np.mean(predictions,axis=0); pred_median = np.median(predictions,axis=0)
                pred_low_outer = np.percentile(predictions,p_low,axis=0); pred_high_outer = np.percentile(predictions,p_high,axis=0)
                # Inner 10-90 band
                pred_low_inner = np.percentile(predictions,10,axis=0); pred_high_inner = np.percentile(predictions,90,axis=0)
                pred_mean_orig = pred_mean * x_std_local + x_mean_local; pred_median_orig = pred_median * x_std_local + x_mean_local
                pred_low_outer_orig = pred_low_outer * x_std_local + x_mean_local; pred_high_outer_orig = pred_high_outer * x_std_local + x_mean_local
                pred_low_inner_orig = pred_low_inner * x_std_local + x_mean_local; pred_high_inner_orig = pred_high_inner * x_std_local + x_mean_local
                t = np.arange(len(x_raw))
                ax.fill_between(t,pred_low_outer_orig,pred_high_outer_orig,color='tab:blue',alpha=0.12,label=f'{p_low}-{p_high}% band')
                ax.fill_between(t,pred_low_inner_orig,pred_high_inner_orig,color='tab:blue',alpha=0.28,label='10-90% band')
                ax.plot(t,pred_mean_orig,color='tab:blue',linewidth=2,label='Mean pred')
                ax.plot(t,pred_median_orig,color='tab:cyan',linewidth=1.2,linestyle='--',label='Median')
                ax.plot(t,x_raw,'k-',alpha=0.7,label='True x',linewidth=1.2)
                obs_indices = obs_pattern==1.0; target_indices = ~obs_indices
                if np.any(obs_indices): ax.plot(t[obs_indices], x_raw[obs_indices],'ro',markersize=3,label='Observed',alpha=0.75)
                if np.any(target_indices): ax.plot(t[target_indices], x_raw[target_indices],'bo',markersize=2,label='Targets',alpha=0.55)
                cond_error, target_rmse = safe_calculate_metrics(predictions_raw,x_raw,obs_pattern,x_std_local,x_mean_local,is_multichannel=True)
                u_error=None
                if predictions_raw.shape[2] >=2:
                    pred_u = predictions_raw[0,:,1,:]; pred_u_median = np.median(pred_u,axis=0); u_error = np.abs(pred_u_median - u_normalized).mean()
                n_observed = np.sum(obs_indices); n_targets = np.sum(target_indices)
                title = f'{input_type} - {day_key}\nObs:{n_observed}, Tgt:{n_targets}\nRMSE:{target_rmse:.3f}'
                if u_error is not None:
                    u_status = '✅' if u_error < 1e-3 else '⚠️'
                    title += f', u:{u_status}'
                ax.set_title(title,fontsize=9); ax.grid(True,alpha=0.3)
                if day_idx==0: ax.legend(fontsize=7,loc='upper right')
                if day_idx==0: ax.set_ylabel(f'{input_type}\nState x',fontsize=9,fontweight='bold')
        plt.suptitle(f'{regime_title} Damping - All Input Types and Days (Dual Bands)',fontsize=15,fontweight='bold')
        plt.tight_layout()
        out_path = os.path.join(save_dir, f'{regime}_grouped.png')
        plt.savefig(out_path, dpi=150)
        print(f"   💾 Saved: {out_path}")
        plt.close(fig)
    print('\n✅ GROUPED VISUALIZATION COMPLETED!')
    print('All plots displayed grouped by damping regime showing:')
    print('  - Rows: Input types (steps, impulses, sine, random_hold)')
    print('  - Columns: Sequential days (Day 1, Day 2, Day 3)')
    print('  - Each regime in separate figure')
    print('Model handles state (x) and input (u) channels. Input u always fully observed.')
def visualize_input_types():
    """Create a comprehensive visualization of all input types and their variations"""
    print("🎨 Creating input types visualization...")
    
    # Generate sample inputs for visualization
    seq_length = 120
    dt = 0.1
    t = np.arange(seq_length) * dt
    
    input_types = ['steps', 'impulses', 'sine', 'random_hold']
    n_variations = 3  # Show 3 variations of each type
    
    fig, axes = plt.subplots(len(input_types), n_variations, figsize=(15, 12))
    fig.suptitle('Exogenous Input Types and Variations\n(Driving Forces for ODE System)', 
                 fontsize=16, fontweight='bold')
    
    for input_idx, input_type in enumerate(input_types):
        for var_idx in range(n_variations):
            ax = axes[input_idx, var_idx]
            
            # Generate different seed for each variation
            np.random.seed(42 + input_idx * 10 + var_idx)
            
            if input_type == 'steps':
                u = np.random.choice([0, 1], size=seq_length).astype(float)
                title_suffix = f"Binary Steps (Variation {var_idx+1})"
                color = 'blue'
                
            elif input_type == 'impulses':
                u = np.zeros(seq_length)
                impulse_times = np.random.choice(seq_length, size=max(1, seq_length//20), replace=False)
                u[impulse_times] = np.random.normal(0, 2, len(impulse_times))
                title_suffix = f"Sparse Impulses (Variation {var_idx+1})"
                color = 'red'
                
            elif input_type == 'sine':
                freq = np.random.uniform(0.1, 0.5)
                phase = np.random.uniform(0, 2*np.pi)
                u = np.sin(2*np.pi*freq*t + phase)
                title_suffix = f"Sinusoidal (f={freq:.2f}, Variation {var_idx+1})"
                color = 'green'
                
            elif input_type == 'random_hold':
                hold_length = np.random.randint(5, 15)
                u = np.zeros(seq_length)
                for i in range(0, seq_length, hold_length):
                    end_idx = min(i + hold_length, seq_length)
                    u[i:end_idx] = np.random.uniform(-1, 1)
                title_suffix = f"Random Hold (Variation {var_idx+1})"
                color = 'purple'
            
            # Plot the input
            ax.plot(t, u, color=color, linewidth=2, alpha=0.8)
            ax.set_title(title_suffix, fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Time (s)')
            if var_idx == 0:  # Only label y-axis for first column
                ax.set_ylabel(f'{input_type.title()}\nInput u(t)', fontweight='bold')
            
            # Add statistics
            u_mean = np.mean(u)
            u_std = np.std(u)
            u_range = np.max(u) - np.min(u)
            ax.text(0.02, 0.98, f'μ={u_mean:.2f}\nσ={u_std:.2f}\nΔ={u_range:.2f}', 
                   transform=ax.transAxes, fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Input types visualization completed!")
    print("These inputs drive the forced ODE: m*x'' + c*x' + k*x = d*u")
    print("Different input patterns will produce different state responses x(t)")
    print("="*70)
    return

if __name__ == "__main__":
    main()
