"""
Train CSDI model for forced ODE with 2 channels (state x + input u)
Based on train_final_zero_loss_fix.py but extended for exogenous inputs
"""

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

def generate_forced_ode_trajectory(length, dt, regime, input_type="steps", noise_std=0.02, is_training=False):
    """Generate forced ODE trajectory with exogenous input u"""
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
        n_segments = np.random.randint(3, 8)
        segment_length = length // n_segments
        for i in range(n_segments):
            start_idx = i * segment_length
            end_idx = min((i + 1) * segment_length, length)
            u[start_idx:end_idx] = np.random.uniform(-1.0, 1.0)
        if end_idx < length:
            u[end_idx:] = u[end_idx-1]
            
    elif input_type == "impulses":
        n_impulses = np.random.randint(2, 6)
        impulse_indices = np.random.choice(length, size=n_impulses, replace=False)
        for idx in impulse_indices:
            u[idx] = np.random.uniform(-2.0, 2.0)
            
    elif input_type == "sine":
        freq = np.random.uniform(0.02, 0.1)
        amp = np.random.uniform(0.5, 1.5)
        phase = np.random.uniform(0, 2*np.pi)
        t = np.arange(length) * dt
        u = amp * np.sin(2 * np.pi * freq * t + phase)
        
    elif input_type == "random_hold":
        hold_length = np.random.randint(3, 9)
        i = 0
        while i < length:
            value = np.random.normal(0, 0.5)
            end_idx = min(i + hold_length, length)
            u[i:end_idx] = value
            i = end_idx
            hold_length = np.random.randint(3, 9)
    
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

def create_training_batch(batch_size, seq_length, device, x_mean, x_std, u_mean, u_std):
    """Create a training batch with forced ODE trajectories"""
    regimes = ['overdamped', 'critically_damped', 'underdamped']
    input_types = ['steps', 'impulses', 'sine', 'random_hold']
    
    batch_data = []
    batch_observed_masks = []
    batch_gt_masks = []
    
    for _ in range(batch_size):
        regime = np.random.choice(regimes)
        input_type = np.random.choice(input_types)
        
        # Generate trajectory
        x, u = generate_forced_ode_trajectory(seq_length, dt=0.1, regime=regime, 
                                            input_type=input_type, is_training=True)
        
        # Normalize x and u separately
        x_norm = (x - x_mean) / x_std
        u_norm = (u - u_mean) / u_std
        
        # Stack [x, u] as 2-channel data
        trajectory = np.stack([x_norm, u_norm], axis=-1)  # (seq_length, 2)
        
        # CRITICAL FIX: Create proper masks for CSDI training
        # observed_mask: ALL data exists (complete data = all 1s for both channels)
        observed_mask = np.ones((seq_length, 2), dtype=np.float32)
        
        # gt_mask: Conditioning pattern - x partially observed, u fully observed
        observed_ratio = np.random.uniform(0.4, 0.8)
        n_observed = int(seq_length * observed_ratio)
        observed_indices = np.random.choice(seq_length, size=n_observed, replace=False)
        
        gt_mask = np.zeros((seq_length, 2), dtype=np.float32)
        gt_mask[observed_indices, 0] = 1.0  # x channel partially observed for conditioning
        gt_mask[:, 1] = 1.0                 # u channel fully observed for conditioning
        
        batch_data.append(trajectory)
        batch_observed_masks.append(observed_mask)
        batch_gt_masks.append(gt_mask)
    
    # Convert to tensors
    batch_data = torch.from_numpy(np.array(batch_data)).to(device)  # (B, T, 2)
    batch_observed_masks = torch.from_numpy(np.array(batch_observed_masks)).to(device)  # (B, T, 2)
    batch_gt_masks = torch.from_numpy(np.array(batch_gt_masks)).to(device)  # (B, T, 2)
    timepoints = torch.arange(seq_length, dtype=torch.float32)[None, :].expand(batch_size, -1).to(device)
    
    batch = {
        'observed_data': batch_data,
        'observed_mask': batch_observed_masks,  # All data exists (complete data)
        'gt_mask': batch_gt_masks,              # Conditioning pattern (subset)
        'timepoints': timepoints,
        'feature_id': torch.zeros((batch_size, 1), dtype=torch.long).to(device)
    }
    
    return batch

def train_forced_ode_model():
    """Train CSDI model for forced ODE with 2 channels"""
    print("🔧 TRAINING 2-CHANNEL FORCED ODE MODEL WITH ZERO LOSS FIX")
    print("=" * 70)
    print("CORRECT MASK UNDERSTANDING:")
    print("- observed_mask: ALL points exist (complete data = all 1s)")
    print("- gt_mask: points available for conditioning (some missing)")
    print("- target_mask = observed_mask - cond_mask = points to predict")
    print("=" * 70)
    print("TRAINING SETUP:")
    print("- Channel 0: State x (partially observed during training)")
    print("- Channel 1: Input u (always fully observed)")
    print("- Various input types: steps, impulses, sine, random_hold")
    print("=" * 70)
    
    # Training parameters
    epochs = 50  # Increased for better convergence
    batch_size = 16
    seq_length = 120
    lr = 1e-3
    patience = 8  # Early stopping: stop if no improvement for 8 epochs
    min_delta = 0.001  # Minimum improvement threshold
    
    # Model configuration for 2 channels
    config = {
        "model": {
            "is_unconditional": False,
            "timeemb": 256,
            "featureemb": 32,
            "target_strategy": "test",
            "num_sample_features": 2  # 2 channels: x and u
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
    
    # Generate sample trajectories to calculate normalization for both channels
    print("📊 Calculating normalization parameters for x and u channels...")
    regimes = ['overdamped', 'critically_damped', 'underdamped']
    input_types = ['steps', 'impulses', 'sine', 'random_hold']
    
    x_samples = []
    u_samples = []
    
    for _ in range(1000):  # Sample 1000 trajectories for normalization
        regime = np.random.choice(regimes)
        input_type = np.random.choice(input_types)
        x, u = generate_forced_ode_trajectory(seq_length, dt=0.1, regime=regime, 
                                            input_type=input_type, is_training=True)
        x_samples.append(x)
        u_samples.append(u)
    
    x_data = np.concatenate(x_samples)
    u_data = np.concatenate(u_samples)
    
    x_mean = float(np.mean(x_data))
    x_std = float(np.std(x_data))
    u_mean = float(np.mean(u_data))
    u_std = float(np.std(u_data))
    
    print(f"✅ X channel normalization: mean={x_mean:.4f}, std={x_std:.4f}")
    print(f"✅ U channel normalization: mean={u_mean:.4f}, std={u_std:.4f}")
    
    # Update config with normalization
    config["x_mean"] = x_mean
    config["x_std"] = x_std
    config["u_mean"] = u_mean
    config["u_std"] = u_std
    
    # Create model with 2 channels
    model = CSDI_Forecasting(config, device, target_dim=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Create save directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"./save/forced_ode_2ch_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"📁 Saving to: {save_dir}")
    print(f"📚 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop with early stopping
    print(f"🏋️ Starting training for up to {epochs} epochs with early stopping...")
    print(f"⏹️ Early stopping: patience={patience}, min_delta={min_delta:.4f}")
    losses = []
    best_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        
        with tqdm(range(config["train"]["itr_per_epoch"]), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for batch_idx in pbar:
                optimizer.zero_grad()
                
                # Create training batch
                batch = create_training_batch(batch_size, seq_length, device, x_mean, x_std, u_mean, u_std)
                
                # Forward pass
                loss = model(batch, is_train=1)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                loss_val = loss.item()
                epoch_losses.append(loss_val)
                pbar.set_postfix(loss=f"{loss_val:.6f}")
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1:3d}/{epochs}: Loss = {avg_loss:.6f}")
        
        # Early stopping check
        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            best_epoch = epoch + 1
            patience_counter = 0
            print(f"🎯 New best loss: {best_loss:.6f}")
        else:
            patience_counter += 1
            print(f"⏳ No improvement for {patience_counter}/{patience} epochs")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"💾 Saved checkpoint: {checkpoint_path}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"⏹️ Early stopping triggered at epoch {epoch+1}")
            print(f"🏆 Best loss {best_loss:.6f} achieved at epoch {best_epoch}")
            break
    
    # Save final model and config
    model_path = os.path.join(save_dir, "model.pth")
    config_path = os.path.join(save_dir, "config.json")
    
    torch.save(model.state_dict(), model_path)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(losses, 'b-', linewidth=2)
    plt.title('2-Channel Forced ODE Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "training_loss.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"💾 Saved final model: {model_path}")
    print(f"📊 Saved loss plot: {os.path.join(save_dir, 'training_loss.png')}")
    print("✅ 2-CHANNEL FORCED ODE TRAINING COMPLETED!")
    print(f"📁 Final model saved to: {save_dir}")
    print(f"🚀 Training loss should show meaningful learning for both channels!")
    print(f"📊 Final training loss: {losses[-1]:.6f}")
    if patience_counter >= patience:
        print(f"🏁 Training stopped early at epoch {len(losses)}/{epochs}")
        print(f"🏆 Best loss achieved: {best_loss:.6f} at epoch {best_epoch}")
    else:
        print(f"🏁 Training completed all {epochs} epochs")
    
    return save_dir

if __name__ == "__main__":
    save_dir = train_forced_ode_model()
    print(f"\n🎉 TRAINING COMPLETE! Model saved to: {save_dir}")
