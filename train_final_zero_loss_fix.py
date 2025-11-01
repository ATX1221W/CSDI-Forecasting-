# FINAL FIX FOR ZERO LOSS ISSUE - CORRECT UNDERSTANDING OF MASKS
# The key insight: observed_mask = ALL data points, cond_mask = subset for conditioning

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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

def generate_ode_trajectory(length, dt, regime, noise_std=0.02, is_training=True):
    """Generate ODE trajectory with training parameter ranges"""
    m, k = 1.0, 1.0
    
    # Use training parameter ranges
    if regime == 'overdamped':
        c = np.random.uniform(2.5, 3.5)
    elif regime == 'critically_damped':
        c = np.random.uniform(1.8, 2.2)
    elif regime == 'underdamped':
        c = np.random.uniform(0.3, 0.7)
    
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

def create_correct_training_batch(batch_size, seq_length, regimes, mean_val, std_val):
    """Create training batch with CORRECT mask understanding
    
    KEY INSIGHT:
    - observed_mask: ALL points that exist in the sequence (usually all 1s)
    - gt_mask: subset of observed_mask used for conditioning (some points missing)
    - target_mask = observed_mask - cond_mask = points to predict
    """
    
    batch_data = []
    batch_observed_masks = []
    batch_gt_masks = []
    
    for _ in range(batch_size):
        # Random regime
        regime = np.random.choice(regimes)
        
        # Generate trajectory
        raw_trajectory = generate_ode_trajectory(seq_length, dt=0.1, regime=regime, is_training=True)
        
        # Normalize
        normalized_trajectory = (raw_trajectory - mean_val) / std_val
        trajectory_data = normalized_trajectory[:, None].astype(np.float32)
        
        # CORRECT MASK SETUP:
        # observed_mask: all points exist (complete data)
        observed_mask = np.ones(seq_length, dtype=float)
        
        # gt_mask: only some points available for conditioning (what we observe)
        missing_rate = 0.4
        n_missing = int(seq_length * missing_rate)
        missing_indices = np.random.choice(seq_length, size=n_missing, replace=False)
        
        gt_mask = np.ones(seq_length, dtype=float)
        gt_mask[missing_indices] = 0.0  # Missing points
        
        # Create masks with proper dimensions
        observed_mask_2d = observed_mask[:, None].astype(np.float32)
        gt_mask_2d = gt_mask[:, None].astype(np.float32)
        
        batch_data.append(trajectory_data)
        batch_observed_masks.append(observed_mask_2d)
        batch_gt_masks.append(gt_mask_2d)
    
    # Convert to tensors
    observed_data = torch.from_numpy(np.stack(batch_data)).to(device)
    observed_mask = torch.from_numpy(np.stack(batch_observed_masks)).to(device)
    gt_mask = torch.from_numpy(np.stack(batch_gt_masks)).to(device)
    timepoints = torch.arange(seq_length, dtype=torch.float32)[None, :].repeat(batch_size, 1).to(device)
    feature_id = torch.zeros((batch_size, 1), dtype=torch.long).to(device)
    
    # Debug: Verify target_mask will be non-zero
    # cond_mask = gt_mask (approximately, through get_test_pattern_mask)
    # target_mask = observed_mask - cond_mask = 1 - gt_mask
    expected_target_sum = (observed_mask - gt_mask).sum()
    print(f"✅ Expected target_mask sum: {expected_target_sum} (should be > 0)")
    print(f"   observed_mask sum: {observed_mask.sum()}, gt_mask sum: {gt_mask.sum()}")
    
    return {
        'observed_data': observed_data,
        'observed_mask': observed_mask,
        'gt_mask': gt_mask,
        'timepoints': timepoints,
        'feature_id': feature_id,
    }

def train_final_fixed_model(epochs=20, batch_size=8, seq_length=120, lr=1e-3):
    """Train model with FINAL fix for zero loss issue"""
    
    print("🔧 FINAL FIX FOR ZERO LOSS ISSUE")
    print("=" * 70)
    print("CORRECT UNDERSTANDING:")
    print("- observed_mask: ALL points exist (complete data = all 1s)")
    print("- gt_mask: points available for conditioning (some missing = subset)")
    print("- target_mask = observed_mask - cond_mask = points to predict")
    print("=" * 70)
    
    # Create model configuration matching original working model
    config = {
        "train_mean": 0.1530,  # Will be updated with actual data
        "train_std": 0.3310,   # Will be updated with actual data
        "model": {
            "is_unconditional": 0,
            "timeemb": 256,
            "featureemb": 32,
            "target_strategy": "test",
            "num_sample_features": 1
        },
        "diffusion": {
            "layers": 6,
            "channels": 96,
            "nheads": 8,
            "diffusion_embedding_dim": 128,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "num_steps": 100,
            "schedule": "linear",
            "is_linear": True,
            "side_dim": 289
        },
        "train": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "itr_per_epoch": 50
        }
    }
    
    # Generate sample trajectories to calculate normalization
    print("📊 Calculating normalization parameters from training data...")
    regimes = ['overdamped', 'critically_damped', 'underdamped']
    sample_data = []
    
    for _ in range(1000):  # Sample 1000 trajectories for normalization
        regime = np.random.choice(regimes)
        trajectory = generate_ode_trajectory(seq_length, dt=0.1, regime=regime, is_training=True)
        sample_data.append(trajectory)
    
    sample_data = np.concatenate(sample_data)
    train_mean = float(np.mean(sample_data))
    train_std = float(np.std(sample_data))
    
    print(f"✅ Calculated normalization: mean={train_mean:.4f}, std={train_std:.4f}")
    
    # Update config with actual normalization
    config["train_mean"] = train_mean
    config["train_std"] = train_std
    
    # Create model
    model = CSDI_Forecasting(config, device, target_dim=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Create save directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f'./save/final_fixed_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    
    # Save config
    with open(f'{save_dir}/config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"📁 Saving to: {save_dir}")
    print(f"📚 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    model.train()
    train_losses = []
    
    print(f"\n🏋️ Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        epoch_losses = []
        
        # Multiple batches per epoch
        batches_per_epoch = 10
        pbar = tqdm(range(batches_per_epoch), desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch_idx in pbar:
            # Create batch with CORRECT masks
            batch = create_correct_training_batch(batch_size, seq_length, regimes, train_mean, train_std)
            
            # Forward pass
            optimizer.zero_grad()
            loss = model(batch, is_train=1)  # is_train=1 for training mode
            
            # Check for NaN or zero loss
            if torch.isnan(loss):
                print(f"❌ NaN loss detected at epoch {epoch+1}, batch {batch_idx+1}")
                continue
            
            if loss.item() == 0.0:
                print(f"❌ STILL Zero loss at epoch {epoch+1}, batch {batch_idx+1}")
                print(f"   This indicates a fundamental issue!")
                continue
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        # Calculate epoch statistics
        if epoch_losses:
            avg_loss = np.mean(epoch_losses)
            train_losses.append(avg_loss)
            
            print(f"Epoch {epoch+1:3d}/{epochs}: Loss = {avg_loss:.6f}")
            
            # Save model checkpoint periodically
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                checkpoint_path = f'{save_dir}/model_epoch_{epoch+1}.pth'
                torch.save(model.state_dict(), checkpoint_path)
                print(f"💾 Saved checkpoint: {checkpoint_path}")
        else:
            print(f"Epoch {epoch+1:3d}/{epochs}: No valid batches (all NaN or zero)")
    
    # Save final model
    final_model_path = f'{save_dir}/model.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"💾 Saved final model: {final_model_path}")
    
    # Plot training loss
    if train_losses:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.title('Training Loss - FINAL FIXED VERSION')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        loss_plot_path = f'{save_dir}/training_loss.png'
        plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"📊 Saved loss plot: {loss_plot_path}")
    
    print(f"\n✅ TRAINING COMPLETED!")
    print(f"📁 Model saved to: {save_dir}")
    print(f"🔧 ZERO LOSS ISSUE SHOULD BE COMPLETELY FIXED!")
    print(f"📊 Final training loss: {train_losses[-1]:.6f}" if train_losses else "No valid training")
    
    return save_dir, config

def main():
    """Main training function"""
    print("🔧 FINAL FIX FOR ZERO LOSS TRAINING ISSUE")
    
    # Train the model with final fix
    model_dir, config = train_final_fixed_model(
        epochs=15,  # Start with fewer epochs to verify fix
        batch_size=8,
        seq_length=120,
        lr=1e-3
    )
    
    print(f"\n🎉 FINAL ZERO LOSS FIX APPLIED!")
    print(f"📁 Final fixed model saved to: {model_dir}")
    print(f"🚀 Training loss should now be meaningful and non-zero!")

if __name__ == "__main__":
    main()
