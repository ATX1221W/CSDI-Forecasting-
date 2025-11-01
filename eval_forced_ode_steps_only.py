"""
Evaluate a trained steps-only 2-channel forced ODE CSDI model on unseen systems and inputs.
Evaluation modes now focus on causal-style forecasting and input diversity:
    1. Incremental (day-by-day) masking with input overlay.
    2. Half-forecast (first 50% conditioned, second 50% predicted) with input overlay.
    3. Multi-input grid per system: multiple distinct step inputs for each damping coefficient c.
    4. Input diversity summary: overlays of several u(t) samples + simple distribution view.

Legacy sparse random masking evaluation removed (redundant with new modes).
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
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


# ----------------------------- Shared generation -----------------------------

def generate_steps_input(T, rng, min_steps=3, max_steps=6, min_dwell=10, max_dwell=60, level_range=(-1.0, 1.0),
                         binary=False, binary_high_freq=False, flip_prob=0.2):
    if binary_high_freq:
        u = np.zeros(T, dtype=np.float32)
        current = int(rng.randint(0, 2))
        u[0] = current
        # Option B: use min_dwell directly (no internal compression) for clarity & user control
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


def build_test_set(config):
    T = int(config['physics']['T'])
    dt = float(config['physics']['dt'])
    steps_cfg = config['steps_cfg']
    phys = config['physics']
    test_c_values = config['splits']['test']['c_values']
    seeds_base = int(config['seeds']['test_base'])
    seqs_per_sys = int(config['dataset_sizes']['test_seqs_per_sys'])

    items = []
    idx = 0
    for c in test_c_values:
        for _ in range(seqs_per_sys):
            seed = seeds_base + idx
            rng = np.random.RandomState(seed)
            # Backward compatibility: allow absence of binary key
            # Pass through high-frequency or binary flags if present
            u = generate_steps_input(T, rng, **steps_cfg)
            x = simulate_forced_ode(u, dt, m=phys['m'], c=float(c), k=phys['k'], d=phys['d'], noise_std=phys['noise_std'], rng=rng)
            items.append({'x': x, 'u': u, 'c': float(c), 'seed': int(seed)})
            idx += 1
    return items


def make_batch(trajectory, x_mean, x_std, u_mean, u_std):
    x = (trajectory['x'] - x_mean) / x_std
    u = (trajectory['u'] - u_mean) / u_std
    data = np.stack([x, u], axis=1).astype(np.float32)  # (T, 2)
    T = data.shape[0]
    observed_mask = np.ones((1, T, 2), dtype=np.float32)
    gt_mask = observed_mask.copy()
    batch = {
        'observed_data': torch.from_numpy(data[None, ...]).to(device),
        'observed_mask': torch.from_numpy(observed_mask).to(device),
        'gt_mask': torch.from_numpy(gt_mask).to(device),
        'timepoints': torch.arange(T, dtype=torch.float32)[None, :].to(device),
        'feature_id': torch.zeros((1, 1), dtype=torch.long).to(device)
    }
    return batch


def apply_sparse_mask(batch, missing_rate=0.3):
    T = batch['observed_mask'].shape[1]
    obs = batch['observed_mask'].clone()
    # x channel masked sparsely; u fully observed
    keep = torch.rand((1, T), device=obs.device) > missing_rate
    obs[:, :, 0] = keep.float()
    gt = obs.clone()
    batch['observed_mask'] = obs
    batch['gt_mask'] = gt
    return batch


def apply_incremental_masks(T, n_days=3, missing_rate=0.2):
    per_day = T // n_days
    patterns = []
    prev = np.zeros(T, dtype=np.float32)
    for day in range(n_days):
        start = day * per_day
        end = T if day == n_days - 1 else (day + 1) * per_day
        mask = prev.copy()
        window = np.ones(end - start, dtype=np.float32)
        # introduce within-window missing
        n_missing = int((end - start) * missing_rate)
        if n_missing > 0 and (end - start) > n_missing:
            miss_idx = np.random.choice(np.arange(start, end), size=n_missing, replace=False)
            window_idx = np.setdiff1d(np.arange(start, end), miss_idx)
        else:
            window_idx = np.arange(start, end)
        mask[window_idx] = 1.0
        patterns.append(mask)
        prev = mask
    return patterns


## Removed legacy sparse evaluation


def evaluate_incremental(model, test_items, config, save_dir, n_samples=10, n_days=3, missing_rate=0.2, p_low=5, p_high=95, input_side=True):
    print(f"[INCR] Starting incremental evaluation with n_samples={n_samples}, days={n_days}, band={p_low}-{p_high}% (input_side={input_side})")
    x_mean, x_std = config['x_mean'], config['x_std']
    u_mean, u_std = config['u_mean'], config['u_std']
    # Layout: each row = one item; day columns + optional input column at end
    nrows = len(test_items)
    ncols = n_days + (1 if input_side else 0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows), squeeze=False)

    per_day_rmses = []  # list of lists
    for i, item in enumerate(test_items):
        print(f"  [INCR] Item {i+1}/{len(test_items)} (c={item['c']:.2f})")
        T = len(item['x'])
        patterns = apply_incremental_masks(T, n_days=n_days, missing_rate=missing_rate)
        item_rmses = []
        for d, mask in enumerate(patterns):
            print(f"    [INCR] Day {d+1}/{n_days} applying mask & sampling...")
            batch = make_batch(item, x_mean, x_std, u_mean, u_std)
            # apply mask: x uses mask; u observed fully
            obs = batch['observed_mask'].clone()
            obs[:, :, 0] = torch.from_numpy(mask[None, :]).to(obs.device)
            batch['observed_mask'] = obs
            batch['gt_mask'] = obs.clone()
            with torch.no_grad():
                samples, observed_data, target_mask, observed_mask_full, observed_tp = model.evaluate(batch, n_samples)
            preds_norm = samples[0, :, 0, :].cpu().numpy()
            pred_mean = preds_norm.mean(axis=0) * x_std + x_mean
            pred_med = np.median(preds_norm, axis=0) * x_std + x_mean
            pred_low = np.percentile(preds_norm, p_low, axis=0) * x_std + x_mean
            pred_high = np.percentile(preds_norm, p_high, axis=0) * x_std + x_mean
            t = np.arange(T)
            # RMSE only on currently unobserved points (target area)
            full_mask = obs[0,:,0].cpu().numpy() > 0.5
            rmse_pred = np.sqrt(np.mean((pred_mean[~full_mask] - item['x'][~full_mask])**2)) if (~full_mask).any() else 0.0
            item_rmses.append(rmse_pred)
            ax = axes[i, d]
            ax.plot(t, item['x'], 'k-', lw=1.0, label='True x')
            ax.fill_between(t, pred_low, pred_high, color='tab:blue', alpha=0.18, label=f'{p_low}-{p_high}% band')
            ax.plot(t, pred_mean, color='tab:blue', lw=1.6, label='Mean pred')
            ax.plot(t, pred_med, color='tab:cyan', lw=1.0, ls='--', label='Median')
            # handle (B,T,K) vs (B,K,T) for observed markers
            omf = observed_mask_full
            if omf.dim() == 3:
                B, D1, D2 = omf.shape
                if D1 == T:
                    obs_idx = omf[0, :, 0].cpu().numpy() > 0.5
                elif D2 == T:
                    obs_idx = omf[0, 0, :].cpu().numpy() > 0.5
                else:
                    obs_idx = omf[0, :, 0].cpu().numpy() > 0.5
            else:
                obs_idx = np.zeros(T, dtype=bool)
            ax.plot(t[obs_idx], item['x'][obs_idx], 'ro', ms=3, label='Observed x', alpha=0.7)
            ax.set_title(f"Item {i+1} Day {d+1}\nRMSE={rmse_pred:.3f}")
            ax.grid(True, alpha=0.3)
            if d == 0:
                ax.legend(fontsize=8)
        # Side input panel (once per item)
        if input_side:
            u_ax = axes[i, -1]
            u_ax.step(np.arange(T), item['u'], where='post', color='tab:orange')
            u_ax.set_ylim(-0.2, 1.2)
            u_ax.set_title(f"Input u\n(c={item['c']:.2f})")
            u_ax.set_yticks([0,1])
            u_ax.grid(True, alpha=0.3)
        per_day_rmses.append(item_rmses)

    plt.suptitle('Incremental evaluation (steps-only, test systems)')
    plt.tight_layout()
    out = os.path.join(save_dir, 'eval_incremental.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"📊 Saved incremental grouped plot: {out}")
    # Save RMSE summary
    rmse_path = os.path.join(save_dir,'incremental_rmse.json')
    # Ensure all numeric types are native Python floats for JSON serialization
    mean_per_day = []
    if per_day_rmses:
        arr = np.array(per_day_rmses, dtype=float)
        mean_per_day = [float(x) for x in np.mean(arr, axis=0)]
    serializable = {
        'per_item_per_day_rmse': [[float(x) for x in row] for row in per_day_rmses],
        'mean_per_day': mean_per_day
    }
    with open(rmse_path,'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"🧾 Saved incremental RMSE JSON: {rmse_path}")

def evaluate_half_forecast(model, test_items, config, save_dir, n_samples=10, p=0.5, dropout=0.1, p_low=5, p_high=95, input_side=True):
    print(f"[HALF] Evaluating half-forecast p={p} dropout={dropout} (input_side={input_side})")
    x_mean, x_std = config['x_mean'], config['x_std']
    u_mean, u_std = config['u_mean'], config['u_std']
    # Each item gets a row: left prediction, optional right input
    nrows = len(test_items)
    ncols = 2 if input_side else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 2.4*nrows), squeeze=False)
    rmses = []
    cover_counts = []
    total_points = 0
    for i, item in enumerate(test_items):
        batch = make_batch(item, x_mean, x_std, u_mean, u_std)
        T = batch['observed_mask'].shape[1]
        cutoff = int(p*T)
        obs = batch['observed_mask'].clone()
        # condition on first half (with dropout) for x
        keep = np.ones(cutoff, dtype=bool)
        n_drop = int(cutoff*dropout)
        if n_drop>0:
            drop_idx = np.random.choice(cutoff, size=n_drop, replace=False)
            keep[drop_idx] = False
        x_mask = np.zeros(T, dtype=np.float32)
        x_mask[:cutoff][keep] = 1.0
        obs[:, :, 0] = torch.from_numpy(x_mask[None,:]).to(obs.device)
        batch['observed_mask'] = obs
        batch['gt_mask'] = obs.clone()
        with torch.no_grad():
            samples, observed_data, target_mask, observed_mask_full, observed_tp = model.evaluate(batch, n_samples)
        preds_norm = samples[0, :, 0, :].cpu().numpy()
        pred_mean = preds_norm.mean(axis=0) * x_std + x_mean
        pred_low = np.percentile(preds_norm, p_low, axis=0) * x_std + x_mean
        pred_high = np.percentile(preds_norm, p_high, axis=0) * x_std + x_mean
        true_x = item['x']
        rmse_future = np.sqrt(np.mean((pred_mean[cutoff:] - true_x[cutoff:])**2)) if cutoff < T else 0.0
        rmses.append(rmse_future)
        # coverage
        inside = (true_x[cutoff:] >= pred_low[cutoff:]) & (true_x[cutoff:] <= pred_high[cutoff:])
        cover_counts.append(inside.sum())
        total_points += len(inside)
        t = np.arange(T)
        ax = axes[i,0]
        ax.plot(t, true_x, 'k-', lw=1.0, label='True x')
        ax.plot(t, pred_mean, color='tab:purple', lw=1.8, label='Mean pred')
        ax.fill_between(t, pred_low, pred_high, color='tab:purple', alpha=0.18, label=f'{p_low}-{p_high}% band')
        ax.axvline(cutoff, color='gray', ls='--', lw=1)
        obs_idx = x_mask>0.5
        ax.plot(t[obs_idx], true_x[obs_idx], 'ro', ms=3, label='Observed x', alpha=0.7)
        ax.set_title(f"Item {i+1}  c={item['c']:.2f}\nRMSEf={rmse_future:.3f}")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8, loc='upper right')
        if input_side:
            u_ax = axes[i,1]
            u_ax.step(t, item['u'], where='post', color='tab:orange')
            u_ax.set_ylim(-0.2,1.2)
            u_ax.axvline(cutoff, color='gray', ls='--', lw=1)
            u_ax.set_title('Input u')
            u_ax.set_yticks([0,1])
            u_ax.grid(True, alpha=0.3)
    plt.suptitle(f'Half Forecast (p={p}) Mean Future RMSE={np.mean(rmses):.4f}, Coverage={(np.sum(cover_counts)/max(total_points,1)):.2f}')
    plt.tight_layout()
    out = os.path.join(save_dir,'eval_half_forecast.png')
    plt.savefig(out,dpi=150)
    plt.close()
    # metrics json
    metrics_path = os.path.join(save_dir,'half_forecast_metrics.json')
    with open(metrics_path,'w') as f:
        json.dump({'future_rmse_per_item': [float(r) for r in rmses],
                   'mean_future_rmse': float(np.mean(rmses)) if rmses else None,
                   'interval_coverage': float(np.sum(cover_counts)/total_points) if total_points else None,
                   'cutoff_fraction': float(p),
                   'dropout': float(dropout),
                   'band':[float(p_low), float(p_high)]}, f, indent=2)
    print(f"📊 Saved half-forecast plot: {out}")
    print(f"🧾 Saved half-forecast metrics: {metrics_path}")


def parse_args():
    ap = argparse.ArgumentParser(description='Evaluate steps-only forced ODE CSDI model')
    ap.add_argument('--n-samples', type=int, default=10, help='Number of diffusion samples')
    # Removed sparse mode argument
    ap.add_argument('--incr-missing', type=float, default=0.2, help='Incremental within-window missing rate')
    ap.add_argument('--band-low', type=float, default=5, help='Lower percentile for uncertainty band')
    ap.add_argument('--band-high', type=float, default=95, help='Upper percentile for uncertainty band')
    ap.add_argument('--incremental-items', type=int, default=6, help='Number of test items to plot incrementally')
    ap.add_argument('--grid-items-per-system', type=int, default=3, help='Items per system for multi-input grid')
    ap.add_argument('--diversity-systems', type=int, default=2, help='Number of systems to summarize input diversity')
    ap.add_argument('--diversity-samples', type=int, default=8, help='Number of inputs to overlay per diversity figure')
    ap.add_argument('--side-input', action='store_true', help='Show input in separate side panels instead of overlay (incremental & half modes)')
    ap.add_argument('--run-dir', type=str, default=None, help='Explicit path to a steps_only_* run directory to evaluate')
    return ap.parse_args()


def main():
    args = parse_args()
    # Find most recent steps-only training run
    if args.run_dir:
        run_dir = args.run_dir
        if not os.path.isdir(run_dir):
            print(f'❌ Provided run-dir does not exist: {run_dir}')
            return
    else:
        candidates = sorted(glob.glob(os.path.join(current_dir, 'save', 'steps_only_*')), reverse=True)
        if not candidates:
            print('❌ No steps-only training run found. Please run train_forced_ode_steps_only.py first.')
            return
        run_dir = candidates[0]
    model_path = os.path.join(run_dir, 'model.pth')
    config_path = os.path.join(run_dir, 'config.json')
    if not (os.path.exists(model_path) and os.path.exists(config_path)):
        print('❌ Missing model or config in latest steps-only run.')
        return
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Load model
    model = CSDI_Forecasting(config, device, target_dim=2).to(device)
    state = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    print(f"✅ Loaded model from {model_path}")

    # Build unseen test set (disjoint c and RNG seeds; 5 sequences/system)
    test_items = build_test_set(config)
    os.makedirs(run_dir, exist_ok=True)

    # Evaluate incremental (grouped)
    evaluate_incremental(model, test_items[:args.incremental_items], config, run_dir,
                         n_samples=args.n_samples,
                         missing_rate=args.incr_missing,
                         p_low=args.band_low,
                         p_high=args.band_high,
                         input_side=args.side_input)

    # Half forecast evaluation on a subset equal to incremental items
    evaluate_half_forecast(model, test_items[:args.incremental_items], config, run_dir,
                           n_samples=args.n_samples, p=0.5, dropout=0.1,
                           p_low=args.band_low, p_high=args.band_high,
                           input_side=args.side_input)

    # Multi-input grid per system (new)
    systems = {}
    for item in test_items:
        systems.setdefault(item['c'], []).append(item)
    # Build subset respecting items-per-system
    subset = []
    for c_val, lst in systems.items():
        subset.extend(lst[:args.grid_items_per_system])
    if subset:
        evaluate_multi_input_grid(model, subset, config, run_dir,
                                  n_samples=args.n_samples,
                                  p_low=args.band_low, p_high=args.band_high)

    # Input diversity summary (new)
    unique_cs = list(systems.keys())[:args.diversity_systems]
    for c_val in unique_cs:
        evaluate_input_diversity_summary(systems[c_val][:args.diversity_samples], config, run_dir, c_val)

    print('✅ Evaluation finished. Plots saved in run directory.')


# ---------------------- New Evaluation Utilities ----------------------

def evaluate_multi_input_grid(model, items, config, save_dir, n_samples=6, p_low=5, p_high=95):
    print(f"[GRID] Multi-input grid evaluation (items={len(items)})")
    x_mean, x_std = config['x_mean'], config['x_std']
    u_mean, u_std = config['u_mean'], config['u_std']
    # Group by c
    grouped = {}
    for it in items:
        grouped.setdefault(it['c'], []).append(it)
    systems = list(grouped.keys())
    ncols = len(systems)
    max_rows = max(len(v) for v in grouped.values())
    fig, axes = plt.subplots(max_rows, ncols, figsize=(4.8*ncols, 3.0*max_rows), squeeze=False)
    for col, c_val in enumerate(systems):
        lst = grouped[c_val]
        for row in range(max_rows):
            ax = axes[row, col]
            if row >= len(lst):
                ax.axis('off')
                continue
            item = lst[row]
            batch = make_batch(item, x_mean, x_std, u_mean, u_std)
            # simple prefix mask (50%) for forecasting context
            T = batch['observed_mask'].shape[1]
            cutoff = T//2
            obs = batch['observed_mask'].clone()
            x_mask = torch.zeros(T)
            x_mask[:cutoff] = 1.0
            obs[:, :, 0] = x_mask
            batch['observed_mask'] = obs
            batch['gt_mask'] = obs.clone()
            with torch.no_grad():
                samples, observed_data, target_mask, observed_mask_full, observed_tp = model.evaluate(batch, n_samples)
            preds_norm = samples[0, :, 0, :].cpu().numpy()
            pred_mean = preds_norm.mean(axis=0) * x_std + x_mean
            pred_low = np.percentile(preds_norm, p_low, axis=0) * x_std + x_mean
            pred_high = np.percentile(preds_norm, p_high, axis=0) * x_std + x_mean
            true_x = item['x']
            rmse_future = np.sqrt(np.mean((pred_mean[cutoff:] - true_x[cutoff:])**2))
            t = np.arange(T)
            ax.plot(t, true_x, 'k-', lw=1.0)
            ax.plot(t, pred_mean, color='tab:blue', lw=1.8)
            ax.fill_between(t, pred_low, pred_high, color='tab:blue', alpha=0.15)
            ax.axvline(cutoff, color='gray', ls='--', lw=1)
            ax2 = ax.twinx()
            ax2.plot(t, item['u'], color='tab:orange', lw=0.9, alpha=0.55)
            ax2.set_yticks([])
            if row == 0:
                ax.set_title(f"c={c_val:.2f}")
            ax.text(0.02,0.92,f"seed={item['seed']}\nRMSEf={rmse_future:.3f}", transform=ax.transAxes, fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5, lw=0.3))
            ax.grid(True, alpha=0.25)
            if col==0:
                ax.set_ylabel('x')
    plt.tight_layout()
    out = os.path.join(save_dir,'eval_multi_input_grid.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"📊 Saved multi-input grid: {out}")

def evaluate_input_diversity_summary(items, config, save_dir, c_val):
    print(f"[DIVERSITY] Input diversity summary for c={c_val:.2f} (n={len(items)})")
    fig, axes = plt.subplots(2,1, figsize=(8,5), gridspec_kw={'height_ratios':[3,1]})
    ax_u, ax_hist = axes
    for it in items:
        ax_u.step(np.arange(len(it['u'])), it['u'], where='post', alpha=0.5)
    ax_u.set_title(f"Input Step Overlays (c={c_val:.2f})")
    ax_u.set_ylabel('u')
    # dwell length distribution approximation via counting constant runs of first curve
    if items:
        u0 = items[0]['u']
        runs = []
        current_val = u0[0]
        run_len = 1
        for val in u0[1:]:
            if val == current_val:
                run_len += 1
            else:
                runs.append(run_len)
                current_val = val
                run_len = 1
        runs.append(run_len)
        ax_hist.hist(runs, bins=min(15,len(runs)), color='tab:gray', alpha=0.7)
        ax_hist.set_ylabel('count')
        ax_hist.set_xlabel('dwell length (steps)')
    plt.tight_layout()
    out = os.path.join(save_dir, f'input_diversity_c_{c_val:.2f}.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"📊 Saved diversity summary: {out}")


if __name__ == '__main__':
    main()
