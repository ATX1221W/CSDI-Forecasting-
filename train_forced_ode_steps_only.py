"""
Train CSDI model for forced ODE with 2 channels (state x + input u), steps-only inputs.
Disjoint train/val/test systems (c values) and input sequences. GPU if available.

Artifacts:
- Saves model and config to ./save/steps_only_<timestamp>/

Notes:
- observed_mask: all ones (data exists)
- gt_mask: conditioning mask (x partially observed, u fully observed)
"""

import os
import sys
import json
import math
import datetime
import argparse
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from tqdm import tqdm

# Ensure we're in the correct directory
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
if current_dir not in sys.path:
    sys.path.append(current_dir)

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
    print("⚠️  CUDA not available, using CPU")

print(f"Using device: {device}")


# ----------------------------- Physics and Data Gen -----------------------------

def create_c_splits():
    """Create disjoint damping coefficient (c) splits for train/val/test."""
    # m=k=1 => critical damping c_crit = 2.0
    train_c = list(np.linspace(0.5, 1.5, 6)) + list(np.linspace(2.2, 3.0, 6))  # 12 systems
    val_c = [1.6, 1.7, 1.8, 1.9]  # near-critical but not exactly
    test_c = [0.6, 0.8, 1.4, 2.0, 2.6, 3.1]  # includes critical and unseen ranges
    return {
        'train': {'c_values': train_c},
        'val': {'c_values': val_c},
        'test': {'c_values': test_c},
    }


def generate_steps_input(T, rng, min_steps=3, max_steps=6, min_dwell=10, max_dwell=60,
                         level_range=(-1.0, 1.0), binary=False, binary_high_freq=False, flip_prob=0.2):
    """Generate a piecewise-constant steps input of length T.

    Modes:
      - Continuous (default): uniform random levels per segment.
      - Binary (binary=True): standard coarse step segments of 0/1.
      - High-frequency binary (binary_high_freq=True): start from an initial bit and flip with probability flip_prob at each time step (after an optional minimum dwell) to create patterns like 111000111010...

    Args follow previous version; new args:
        binary_high_freq: if True, ignore segment logic and use Bernoulli flips.
        flip_prob: probability of flipping (0->1 or 1->0) at each step once past minimum dwell.
    Precedence: binary_high_freq overrides binary; if both False => continuous.
    """
    if binary_high_freq:
        u = np.zeros(T, dtype=np.float32)
        # choose initial level 0/1
        current = int(rng.randint(0, 2)) if binary else int(rng.rand() > 0.5)
        u[0] = current
        # dwell control: now use the provided min_dwell directly (no compression) so user intent matches runtime
        dwell_counter = 1
        min_local_dwell = max(1, min_dwell)
        for t in range(1, T):
            if dwell_counter >= min_local_dwell and rng.rand() < flip_prob:
                current = 1 - current
                dwell_counter = 0
            u[t] = current
            dwell_counter += 1
        return u

    n_segments = rng.randint(min_steps, max_steps + 1)
    u = np.zeros(T, dtype=np.float32)
    t = 0
    for _ in range(n_segments):
        dwell = int(rng.randint(min_dwell, max_dwell + 1))
        if binary:
            level = float(rng.randint(0, 2))
        else:
            level = float(rng.uniform(level_range[0], level_range[1]))
        end = min(t + dwell, T)
        u[t:end] = level
        t = end
        if t >= T:
            break
    if t < T:
        u[t:] = u[t - 1] if t > 0 else 0.0
    return u


def simulate_forced_ode(u, dt, m=1.0, c=1.0, k=1.0, d=1.0, x0=None, v0=None, noise_std=0.05, rng=None):
    """Simulate m*x'' + c*x' + k*x = d*u using Euler integration.

    Args:
        u: (T,) input array
        dt: time-step
        m, c, k, d: ODE parameters
        x0, v0: initial conditions (random if None)
        noise_std: Gaussian noise std added to x only
        rng: RandomState
    Returns:
        x: (T,) noisy state trajectory
    """
    T = len(u)
    if rng is None:
        rng = np.random.RandomState()
    if x0 is None:
        x0 = rng.uniform(0.5, 1.5)
    if v0 is None:
        v0 = rng.uniform(-0.3, 0.3)
    x = np.zeros(T, dtype=np.float32)
    v = np.zeros(T, dtype=np.float32)
    x[0] = float(x0)
    v[0] = float(v0)
    inv_m = 1.0 / m
    for t in range(1, T):
        a = (-c * v[t - 1] - k * x[t - 1] + d * u[t - 1]) * inv_m
        v[t] = v[t - 1] + a * dt
        x[t] = x[t - 1] + v[t] * dt
    if noise_std and noise_std > 0:
        x = x + rng.randn(T).astype(np.float32) * float(noise_std)
    return x


# ----------------------------- Batching -----------------------------

class StepsOnlyDataset:
    def __init__(self, c_values, seqs_per_system, T, dt, seeds_base, steps_cfg, phys_cfg, noise_std=0.05):
        self.items = []  # list of dicts with x, u, c, seed
        idx = 0
        for c in c_values:
            for s in range(seqs_per_system):
                seed = int(seeds_base + idx)
                rng = np.random.RandomState(seed)
                u = generate_steps_input(T, rng, **steps_cfg)
                x = simulate_forced_ode(u, dt, m=phys_cfg['m'], c=c, k=phys_cfg['k'], d=phys_cfg['d'], noise_std=noise_std, rng=rng)
                self.items.append({'x': x, 'u': u, 'c': float(c), 'seed': seed})
                idx += 1

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def _make_base_arrays(batch_size, seqs, x_mean, x_std, u_mean, u_std):
    T = len(seqs[0]['x'])
    batch_data = np.zeros((batch_size, T, 2), dtype=np.float32)
    for b, item in enumerate(seqs):
        batch_data[b, :, 0] = (item['x'] - x_mean) / x_std
        batch_data[b, :, 1] = (item['u'] - u_mean) / u_std
    return batch_data

def _assemble_batch(batch_data, gt_mask, device):
    batch_size, T, _ = batch_data.shape
    observed_mask = np.ones_like(gt_mask, dtype=np.float32)
    timepoints = torch.arange(T, dtype=torch.float32)[None, :].repeat(batch_size, 1).to(device)
    return {
        'observed_data': torch.from_numpy(batch_data).to(device),
        'observed_mask': torch.from_numpy(observed_mask).to(device),
        'gt_mask': torch.from_numpy(gt_mask.astype(np.float32)).to(device),
        'timepoints': timepoints,
        'feature_id': torch.zeros((batch_size, 1), dtype=torch.long).to(device)
    }

def create_training_batch(batch_size, dataset, device, x_mean, x_std, u_mean, u_std, rng, strategy_probs: Dict[str,float], prefix_dropout_range: Tuple[float,float]):
    """Create a training batch with mixed masking strategies.

    Strategies:
      - prefix: choose split p; condition on prefix (optional dropout) only
      - random: scattered points (legacy)
      - half: fixed p=0.5 with optional dropout inside prefix
    """
    idxs = rng.choice(len(dataset), size=batch_size, replace=True)
    seqs = [dataset[i] for i in idxs]
    T = len(seqs[0]['x'])
    batch_data = _make_base_arrays(batch_size, seqs, x_mean, x_std, u_mean, u_std)
    gt_mask = np.zeros((batch_size, T, 2), dtype=np.float32)

    # Precompute strategy selection
    strategies = list(strategy_probs.keys())
    probs = np.array([strategy_probs[s] for s in strategies], dtype=float)
    probs = probs / probs.sum()
    chosen = rng.choice(strategies, size=batch_size, p=probs)

    for b, strat in enumerate(chosen):
        if strat == 'random':
            observed_ratio = float(rng.uniform(0.4, 0.8))
            n_obs = int(T * observed_ratio)
            obs_idx = rng.choice(T, size=n_obs, replace=False)
            gt_mask[b, obs_idx, 0] = 1.0
        else:
            # prefix or half
            if strat == 'half':
                p = 0.5
            else:  # prefix
                p = float(rng.uniform(0.3, 0.7))
            cutoff = int(p * T)
            gt_mask[b, :cutoff, 0] = 1.0
            # optional dropout inside prefix
            drop_low, drop_high = prefix_dropout_range
            if drop_high > 0 and cutoff > 5:
                dropout_rate = float(rng.uniform(drop_low, drop_high))
                n_drop = int(cutoff * dropout_rate)
                if n_drop > 0:
                    drop_idx = rng.choice(cutoff, size=n_drop, replace=False)
                    gt_mask[b, drop_idx, 0] = 0.0
        # u channel always observed / conditioned
        gt_mask[b, :, 1] = 1.0

    batch = _assemble_batch(batch_data, gt_mask, device)
    batch['mask_strategy'] = chosen  # for optional debugging
    return batch


# ----------------------------- Training -----------------------------

def train_steps_only(args=None):
    print("🔧 TRAINING 2-CHANNEL FORCED ODE (STEPS-ONLY)")
    print("=" * 72)

    # Fast-to-iterate defaults (can be overridden by CLI)
    T = 200
    dt = 0.05
    noise_std = 0.05

    phys_cfg = {'m': 1.0, 'k': 1.0, 'd': 1.0}
    steps_cfg = {
        'min_steps': 3, 'max_steps': 6,
        'min_dwell': 10, 'max_dwell': 60,
        'level_range': (0.0, 1.0),
        'binary': True,
        'binary_high_freq': False,  # default OFF; enable with --high-freq 1
        'flip_prob': 0.25  # only used if high-frequency mode is enabled
    }

    if args is not None:
        if args.flip_prob is not None:
            steps_cfg['flip_prob'] = float(args.flip_prob)
        if args.high_freq is not None:
            steps_cfg['binary_high_freq'] = bool(args.high_freq)
        if args.coarse_binary:
            # force coarse mode
            steps_cfg['binary_high_freq'] = False
            steps_cfg['binary'] = True
        if getattr(args, 'min_dwell', None) is not None:
            steps_cfg['min_dwell'] = int(args.min_dwell)

    # Splits and sizes
    splits = create_c_splits()
    train_systems = splits['train']['c_values']
    val_systems = splits['val']['c_values']
    test_systems = splits['test']['c_values']

    train_seqs_per_sys = 10
    val_seqs_per_sys = 5
    # test will be handled by eval script (5 per system)

    seeds = {
        'train_base': 12345,
        'val_base': 22345,
        'test_base': 32345,
        'norm_base': 42345,
    }

    # Build datasets (train/val) with disjoint seeds and systems
    train_ds = StepsOnlyDataset(train_systems, train_seqs_per_sys, T, dt, seeds['train_base'], steps_cfg, phys_cfg, noise_std)
    val_ds = StepsOnlyDataset(val_systems, val_seqs_per_sys, T, dt, seeds['val_base'], steps_cfg, phys_cfg, noise_std)

    # Normalization from synthetic pool (train domain only for realism)
    print("📊 Computing per-channel normalization from training distribution...")
    norm_rng = np.random.RandomState(seeds['norm_base'])
    x_pool = []
    u_pool = []
    # 500 samples across train systems
    for i in range(500):
        c = float(norm_rng.choice(train_systems))
        u = generate_steps_input(T, norm_rng, **steps_cfg)
        x = simulate_forced_ode(u, dt, m=phys_cfg['m'], c=c, k=phys_cfg['k'], d=phys_cfg['d'], noise_std=noise_std, rng=norm_rng)
        x_pool.append(x)
        u_pool.append(u)
    x_all = np.concatenate(x_pool).astype(np.float32)
    u_all = np.concatenate(u_pool).astype(np.float32)
    x_mean, x_std = float(np.mean(x_all)), float(np.std(x_all) + 1e-6)
    u_mean, u_std = float(np.mean(u_all)), float(np.std(u_all) + 1e-6)
    print(f"✅ X normalization: mean={x_mean:.4f}, std={x_std:.4f}")
    print(f"✅ U normalization: mean={u_mean:.4f}, std={u_std:.4f}")

    # Model config
    epochs = args.epochs if (args and args.epochs is not None) else 12
    batch_size = 16
    lr = 1e-3
    patience = 4
    min_delta = 1e-3
    itr_per_epoch = args.iters if (args and args.iters is not None) else 50

    config = {
        'model': {
            'is_unconditional': False,
            'timeemb': 256,
            'featureemb': 32,
            'target_strategy': 'test',
            'num_sample_features': 2,
        },
        'diffusion': {
            'layers': 6,
            'channels': 96,
            'nheads': 8,
            'diffusion_embedding_dim': 128,
            'beta_start': 0.0001,
            'beta_end': 0.02,
            'num_steps': 100,
            'schedule': 'linear',
            'is_linear': True,
            'side_dim': 289,
        },
        'train': {
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'itr_per_epoch': itr_per_epoch,
            'mask_strategy_probs': {'prefix':0.5,'random':0.3,'half':0.2},
            'prefix_dropout_range': [0.05,0.15],
        },
        'validation': {
            'n_batches': 4,
            'n_samples': 8,
            'prefix_splits': [0.5,0.7],
            'band': [5,95]
        },
        'physics': {**phys_cfg, 'dt': dt, 'T': T, 'noise_std': noise_std},
        'steps_cfg': steps_cfg,
        'splits': splits,
        'seeds': seeds,
        'dataset_sizes': {
            'train_systems': len(train_systems),
            'val_systems': len(val_systems),
            'train_seqs_per_sys': train_seqs_per_sys,
            'val_seqs_per_sys': val_seqs_per_sys,
            'test_seqs_per_sys': 5,
        },
        'x_mean': x_mean,
        'x_std': x_std,
        'u_mean': u_mean,
        'u_std': u_std,
    }

    # Instantiate model
    model = CSDI_Forecasting(config, device, target_dim=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Save directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(current_dir, 'save', f'steps_only_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    print(f"📁 Saving to: {save_dir}")
    print(f"📚 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training loop with validation-based early stopping (future RMSE)
    rng = np.random.RandomState(seeds['train_base'])
    losses = []
    val_history = []
    best_metric = float('inf')
    best_epoch = 0
    patience_counter = 0

    def run_validation():
        model.eval()
        with torch.no_grad():
            rmses = []
            for _ in range(config['validation']['n_batches']):
                # build a deterministic prefix batch from val set
                idxs = rng.choice(len(val_ds), size=batch_size, replace=True)
                seqs = [val_ds[i] for i in idxs]
                batch_data = _make_base_arrays(batch_size, seqs, x_mean, x_std, u_mean, u_std)
                Tloc = batch_data.shape[1]
                gt_mask = np.zeros((batch_size, Tloc, 2), dtype=np.float32)
                p = float(np.random.choice(config['validation']['prefix_splits']))
                cutoff = int(p * Tloc)
                gt_mask[:, :cutoff, 0] = 1.0
                gt_mask[:, :, 1] = 1.0
                val_batch = _assemble_batch(batch_data, gt_mask, device)
                samples, obs_data, target_mask, obs_mask, tpts = model.evaluate(val_batch, config['validation']['n_samples'])
                preds = samples[0, :, 0, :].cpu().numpy()  # (n_samples, T)
                pred_mean = preds.mean(axis=0) * x_std + x_mean
                # Recover true x
                true_x = batch_data[0,:,0]*x_std + x_mean  # representative; could average over batch
                rmse_future = np.sqrt(np.mean((pred_mean[cutoff:] - true_x[cutoff:])**2)) if cutoff < Tloc else 0.0
                rmses.append(rmse_future)
            return float(np.mean(rmses)) if rmses else float('inf')

    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        with tqdm(range(config['train']['itr_per_epoch']), desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for _ in pbar:
                batch = create_training_batch(batch_size, train_ds, device, x_mean, x_std, u_mean, u_std, rng,
                                              config['train']['mask_strategy_probs'], tuple(config['train']['prefix_dropout_range']))
                optimizer.zero_grad()
                loss = model(batch, is_train=1)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                val = float(loss.item())
                epoch_losses.append(val)
                pbar.set_postfix(loss=f"{val:.6f}")

        avg_loss = float(np.mean(epoch_losses))
        losses.append(avg_loss)
        val_rmse = run_validation()
        val_history.append(val_rmse)
        print(f"Epoch {epoch+1:02d}/{epochs} - train_loss={avg_loss:.6f}  val_future_RMSE={val_rmse:.5f}")

        improved = val_rmse < (best_metric - min_delta)
        if improved:
            best_metric = val_rmse
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir,'model_best.pth'))
        else:
            patience_counter += 1
            print(f"⏳ No val improvement for {patience_counter}/{patience} epochs (best {best_metric:.5f} @ {best_epoch})")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            ckpt = os.path.join(save_dir, f'model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), ckpt)
            print(f"💾 Saved checkpoint: {ckpt}")

        if patience_counter >= patience:
            print(f"⏹️ Early stopping at epoch {epoch+1}; best epoch {best_epoch} val_future_RMSE={best_metric:.6f}")
            break

    # Save final model and config
    model_path = os.path.join(save_dir, 'model.pth')
    config_path = os.path.join(save_dir, 'config.json')
    torch.save(model.state_dict(), model_path)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Save loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(losses, 'b-', lw=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Steps-Only 2-Channel Training Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_loss.png'), dpi=150)
    plt.close()

    print("✅ Training complete")
    print(f"🏆 Best future RMSE {best_metric:.6f} @ epoch {best_epoch}")
    print(f"💾 Saved model: {model_path}")
    print(f"🧾 Saved config: {config_path}")
    # Save validation curve
    plt.figure(figsize=(8,5))
    plt.plot(val_history, 'r-', lw=2, label='Val Future RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('Validation Future RMSE')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,'val_future_rmse.png'), dpi=150)
    plt.close()

    # Persist metric history
    with open(os.path.join(save_dir,'metrics.json'),'w') as f:
        json.dump({'train_loss': losses, 'val_future_rmse': val_history, 'best_epoch': best_epoch, 'best_future_rmse': best_metric}, f, indent=2)

    return save_dir


def _compute_binary_highfreq_stats(dataset, max_items=30):
    flips = []
    dwells_all = []
    sample_us = []
    for i, item in enumerate(dataset.items[:max_items]):
        u = item['u']
        sample_us.append(u)
        # flips
        flips.append(int((u[1:] != u[:-1]).sum()))
        # dwell lengths
        runs = []
        current = u[0]
        rl = 1
        for val in u[1:]:
            if val == current:
                rl += 1
            else:
                runs.append(rl)
                current = val
                rl = 1
        runs.append(rl)
        dwells_all.extend(runs)
    stats = {
        'num_sequences': len(sample_us),
        'mean_flips': float(np.mean(flips)) if flips else None,
        'std_flips': float(np.std(flips)) if flips else None,
        'mean_dwell': float(np.mean(dwells_all)) if dwells_all else None,
        'std_dwell': float(np.std(dwells_all)) if dwells_all else None,
    }
    return stats, sample_us, dwells_all


def add_input_stats_artifacts(train_ds, save_dir):
    stats, sample_us, dwells = _compute_binary_highfreq_stats(train_ds)
    # Save JSON
    with open(os.path.join(save_dir, 'input_highfreq_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    # Plot sample sequences (up to 8)
    n_show = min(8, len(sample_us))
    plt.figure(figsize=(10, 4))
    for i in range(n_show):
        plt.step(range(len(sample_us[i])), sample_us[i], where='post', label=f'seq{i}')
    plt.title('High-Frequency Binary Inputs (samples)')
    plt.xlabel('t')
    plt.ylabel('u')
    plt.ylim(-0.2, 1.2)
    if n_show <= 10:
        plt.legend(fontsize=7, ncol=4)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'input_highfreq_samples.png'), dpi=150)
    plt.close()
    # Dwell histogram
    if dwells:
        plt.figure(figsize=(6,4))
        plt.hist(dwells, bins=min(30, max(5, len(set(dwells)))) , color='tab:gray', alpha=0.8)
        plt.title('Dwell Length Distribution (binary high-freq)')
        plt.xlabel('length (steps)')
        plt.ylabel('count')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'input_highfreq_dwell_hist.png'), dpi=150)
        plt.close()


def parse_args():
    ap = argparse.ArgumentParser(description='Train steps-only forced ODE model (with high-frequency binary option)')
    ap.add_argument('--epochs', type=int, default=None, help='Override number of epochs (default 12)')
    ap.add_argument('--iters', type=int, default=None, help='Override iterations per epoch (default 50)')
    ap.add_argument('--flip-prob', type=float, default=None, help='Flip probability for high-frequency binary mode')
    ap.add_argument('--high-freq', type=int, default=None, help='1 to enable high-frequency mode, 0 to disable')
    ap.add_argument('--coarse-binary', action='store_true', help='Force coarse binary segments (disables high-frequency)')
    ap.add_argument('--min-dwell', type=int, default=None, help='Override min_dwell (also affects high-frequency internal min_local_dwell)')
    return ap.parse_args()


if __name__ == '__main__':
    args = parse_args()
    out = train_steps_only(args)
    # After training gather and save input stats from training dataset if available
    # (Recreate small dataset to compute stats based on config saved)
    try:
        with open(os.path.join(out, 'config.json'),'r') as f:
            cfg = json.load(f)
        # Recreate train dataset for stats
        splits = cfg['splits']
        train_systems = splits['train']['c_values']
        ds_for_stats = StepsOnlyDataset(train_systems, cfg['dataset_sizes']['train_seqs_per_sys'], cfg['physics']['T'], cfg['physics']['dt'], cfg['seeds']['train_base'], cfg['steps_cfg'], {'m':cfg['physics']['m'],'k':cfg['physics']['k'],'d':cfg['physics']['d']}, cfg['physics']['noise_std'])
        add_input_stats_artifacts(ds_for_stats, out)
        print('📝 Saved high-frequency input statistics and plots.')
    except Exception as e:
        print(f'⚠️ Could not generate input stats post-training: {e}')
    print(f"\n🎉 TRAINING COMPLETE! Model saved to: {out}")
