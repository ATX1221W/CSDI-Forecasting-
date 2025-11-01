# TRUE INCREMENTAL/SEQUENTIAL ODE FORECASTING - FINAL VERSION
# Implements exactly what you requested: chronological time progression with realistic missing data

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json
import datetime
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
        """Create batch with specific observation mask"""
        if obs_mask.ndim == 1:
            gt_mask = obs_mask[:, None].astype(np.float32)
        else:
            gt_mask = obs_mask.astype(np.float32)

        observed_mask_tensor = torch.from_numpy(gt_mask[None, :, :]).to(self.device)

        batch = {
            'observed_data': torch.from_numpy(trajectory_data[None, :, :]).to(self.device),
            'observed_mask': observed_mask_tensor,
            'gt_mask': observed_mask_tensor,
            'timepoints': torch.arange(len(trajectory_data), dtype=torch.float32)[None, :].to(self.device),
            'feature_id': torch.tensor([[0]], dtype=torch.long).to(self.device),
        }
        return batch

    def safe_evaluate_with_nan_detection(self, batch, n_samples):
        """Evaluate with comprehensive NaN detection"""
        try:
            self.model.eval()
            self.model = self.model.to(self.device)
            
            with torch.no_grad():
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

def safe_calculate_metrics(predictions, true_data, obs_pattern, scaler_val, mean_scaler_val):
    """Calculate metrics with comprehensive NaN handling"""
    try:
        # Handle predictions shape
        if predictions is None:
            return float('nan'), float('nan')
        
        if len(predictions.shape) == 4 and predictions.shape[0] == 1:
            if predictions.shape[2] == 1:
                pred_array = predictions[0, :, 0, :]
            elif predictions.shape[3] == 1:
                pred_array = predictions[0, :, :, 0]
            else:
                print(f"❌ Unexpected predictions shape: {predictions.shape}")
                return float('nan'), float('nan')
        else:
            print(f"❌ Unexpected predictions shape: {predictions.shape}")
            return float('nan'), float('nan')
        
        # Check for NaN in predictions
        if np.isnan(pred_array).any():
            print(f"❌ NaN in prediction array")
            return float('nan'), float('nan')
        
        # Calculate median prediction
        pred_median = np.median(pred_array, axis=0)
        
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

def evaluate_and_visualize_results(results, true_data, regime_name, scaler_val, mean_scaler_val):
    """Evaluate and visualize incremental forecasting results"""
    
    # Create visualization
    n_days = len(results)
    fig, axes = plt.subplots(1, n_days, figsize=(5 * n_days, 6))
    
    if n_days == 1:
        axes = [axes]
    
    performance = {}
    
    for day_idx, (day_key, day_results) in enumerate(results.items()):
        ax = axes[day_idx]
        
        if not day_results['success']:
            ax.text(0.5, 0.5, f'{day_key}\nFAILED\n{day_results.get("error", "Unknown")}', 
                   horizontalalignment='center', verticalalignment='center', 
                   transform=ax.transAxes, fontsize=12, color='red')
            ax.set_title(f'{regime_name} - {day_key} - FAILED')
            
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
            if predictions_raw.shape[2] == 1:
                predictions = predictions_raw[0, :, 0, :]
            elif predictions_raw.shape[3] == 1:
                predictions = predictions_raw[0, :, :, 0]
            else:
                continue
        
        # Calculate statistics
        pred_median = np.median(predictions, axis=0)
        pred_5 = np.percentile(predictions, 5, axis=0)
        pred_95 = np.percentile(predictions, 95, axis=0)
        
        # Convert to original space
        pred_median_orig = pred_median * scaler_val + mean_scaler_val
        pred_5_orig = pred_5 * scaler_val + mean_scaler_val
        pred_95_orig = pred_95 * scaler_val + mean_scaler_val
        
        # Plot
        t = np.arange(len(true_data))
        ax.plot(t, pred_median_orig, 'g-', label='Prediction', linewidth=2)
        ax.fill_between(t, pred_5_orig, pred_95_orig, color='g', alpha=0.2, label='90% CI')
        ax.plot(t, true_data, 'k-', alpha=0.6, label='True', linewidth=1)
        
        # Show observations with different colors for each day's contribution
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
        
        # Plot all observations
        if np.any(obs_indices):
            ax.plot(t[obs_indices], true_data[obs_indices], 'ro', markersize=4, label='All Observed', alpha=0.6)
        
        # Highlight new observations for this day
        if np.any(new_obs_indices):
            ax.plot(t[new_obs_indices], true_data[new_obs_indices], 'ro', markersize=6, 
                   label=f'New Day {day_idx+1}', markeredgecolor='darkred', markeredgewidth=2)
        
        # Plot target points
        if np.any(target_indices):
            ax.plot(t[target_indices], true_data[target_indices], 'bo', markersize=2, label='Targets', alpha=0.5)
        
        # Calculate metrics
        cond_error, target_rmse = safe_calculate_metrics(
            predictions_raw, true_data, obs_pattern, scaler_val, mean_scaler_val
        )
        
        # Title with key information
        new_obs_count = np.sum(new_obs_indices)
        total_obs_count = np.sum(obs_indices)
        target_count = np.sum(target_indices)
        
        ax.set_title(f'{regime_name} - {day_key}\n'
                    f'+{new_obs_count} new obs (total: {total_obs_count}), {target_count} targets\n'
                    f'Cond: {cond_error:.2e}, RMSE: {target_rmse:.3f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        performance[day_key] = {
            'conditioning_error': cond_error,
            'target_rmse': target_rmse,
            'n_observed': total_obs_count,
            'n_new_observed': new_obs_count,
            'n_targets': target_count,
            'success': True
        }
    
    plt.suptitle(f'{regime_name} - TRUE Incremental/Sequential ODE Forecasting', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return performance

def main():
    """Main function for incremental ODE forecasting"""
    print("🚀 TRUE INCREMENTAL/SEQUENTIAL ODE FORECASTING")
    print("=" * 60)
    print("Day 1: Observe first ~40 time points → predict rest")
    print("Day 2: Observe next ~40 time points → predict rest") 
    print("Day 3: Observe final ~40 time points → predict rest")
    print("Each day includes realistic missing observations")
    print("=" * 60)
    
    # Load properly trained model (prioritizing the fixed zero-loss model)
    model_paths = [
        ('./save/final_fixed_20250808_080019/model.pth', './save/final_fixed_20250808_080019/config.json'),
        ('./save/incremental_matching_20250808_075333/model.pth', './save/incremental_matching_20250808_075333/config.json'),
        ('./save/clean_training_20250808_072954/model.pth', './save/clean_training_20250808_072954/config.json'),
        ('./save/incremental_sparse_ode_20250730_033305/model.pth', './save/incremental_sparse_ode_20250730_033305/config.json')
    ]
    
    model_path = None
    config_path = None
    
    for mp, cp in model_paths:
        if os.path.exists(mp) and os.path.exists(cp):
            model_path = mp
            config_path = cp
            print(f"✅ Found model at: {mp}")
            break
    
    if model_path is None:
        print("❌ No working model found!")
        print("Available paths checked:")
        for mp, cp in model_paths:
            print(f"  {mp} - {'✅' if os.path.exists(mp) else '❌'}")
        return
    
    # Load model with proper configuration handling
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Check if it's the new clean model format
    if 'train_mean' in config and 'train_std' in config:
        working_mean = config['train_mean']
        working_std = config['train_std']
        print(f"✅ Using clean model normalization: mean={working_mean:.4f}, std={working_std:.4f}")
    else:
        # Fallback to old normalization
        working_mean = 0.1530
        working_std = 0.3310
        print(f"⚠️  Using fallback normalization: mean={working_mean:.4f}, std={working_std:.4f}")
    
    model = CSDI_Forecasting(config, device, target_dim=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()
    
    scaler = torch.tensor([working_std], dtype=torch.float32, device=device)
    mean_scaler = torch.tensor([working_mean], dtype=torch.float32, device=device)
    
    print(f"✅ Model loaded with normalization: mean={working_mean:.4f}, std={working_std:.4f}")
    
    # Create forecaster
    forecaster = IncrementalODEForecaster(model, device, scaler, mean_scaler)
    
    # Test each regime
    regimes = ['overdamped', 'critically_damped', 'underdamped']
    
    for regime in regimes:
        print(f"\n=== Testing {regime.replace('_', ' ').title()} ===")
        
        # Generate test trajectory
        np.random.seed(42)  # Consistent seed
        seq_length = 120
        raw_trajectory = generate_ode_trajectory(seq_length, dt=0.1, regime=regime, is_training=False)
        
        # Normalize
        normalized_trajectory = (raw_trajectory - working_mean) / working_std
        trajectory_data = normalized_trajectory[:, None].astype(np.float32)
        
        print(f"Trajectory range: [{raw_trajectory.min():.3f}, {raw_trajectory.max():.3f}]")
        
        # Run incremental forecasting
        results = forecaster.forecast_incremental_sequential(
            trajectory_data, n_days=3, missing_rate=0.2, n_samples=3
        )
        
        # Evaluate and visualize
        performance = evaluate_and_visualize_results(
            results, raw_trajectory, regime.replace('_', ' ').title(), 
            working_std, working_mean
        )
        
        # Print performance summary
        print(f"\nPerformance Summary:")
        for day_key, metrics in performance.items():
            if metrics['success']:
                print(f"  {day_key}: +{metrics['n_new_observed']} new obs, "
                      f"Total={metrics['n_observed']}, Targets={metrics['n_targets']}, "
                      f"Cond={metrics['conditioning_error']:.2e}, RMSE={metrics['target_rmse']:.3f}")
            else:
                print(f"  {day_key}: FAILED")
    
    print(f"\n✅ TRUE INCREMENTAL ODE FORECASTING COMPLETED!")
    print(f"This is exactly what you requested - sequential time progression with realistic missing data.")

if __name__ == "__main__":
    main()
