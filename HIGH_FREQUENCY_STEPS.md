High-Frequency Binary Step Inputs
=================================

Overview
--------
The training (`train_forced_ode_steps_only.py`) and evaluation (`eval_forced_ode_steps_only.py`) generators now support a high-frequency binary mode to create patterns like `1110001110100100`.

Configuration Keys (in `steps_cfg`)
-----------------------------------
- `binary` (bool): Use coarse piecewise-constant 0/1 segments when `True` (original binary behavior).
- `binary_high_freq` (bool): When `True`, overrides segment logic and produces a per-time-step binary flip process.
- `flip_prob` (float): Probability of flipping (0→1 or 1→0) at each time step once a minimal local dwell is satisfied.
- `min_dwell`, `max_dwell`: For coarse mode these bound segment dwell lengths. In high-frequency mode `min_dwell` is now used directly as the strict minimum number of steps before a flip (no internal compression factor).

Generation Logic (High-Frequency)
---------------------------------
1. Draw initial bit uniformly from {0,1}.
2. For each subsequent time step t:
  - If the local dwell counter >= `min_dwell` and `rng.rand() < flip_prob`, flip the bit and reset the dwell counter.
  - Assign current bit to `u[t]` and increment dwell counter.

Choosing Parameters
-------------------
- More rapid variation: increase `flip_prob` (e.g. 0.35–0.5) or lower `min_dwell`.
- Smoother (longer constant runs): lower `flip_prob` and/or raise `min_dwell` (now an exact lower bound on run length).
- To revert to original coarse binary segments, set `binary_high_freq=False` and keep `binary=True`.

Example `steps_cfg` Variants
----------------------------
High-frequency (default added):
```
steps_cfg = {
  'min_steps': 3, 'max_steps': 6,
  'min_dwell': 10, 'max_dwell': 60,
  'level_range': (0.0, 1.0),
  'binary': True,
  'binary_high_freq': True,
  'flip_prob': 0.25
}
```
Coarse binary segments:
```
steps_cfg = {
  'min_steps': 3, 'max_steps': 6,
  'min_dwell': 10, 'max_dwell': 60,
  'level_range': (0.0, 1.0),
  'binary': True,
  'binary_high_freq': False
}
```
Continuous (non-binary):
```
steps_cfg = {
  'min_steps': 3, 'max_steps': 6,
  'min_dwell': 10, 'max_dwell': 60,
  'level_range': (-1.0, 1.0),
  'binary': False,
  'binary_high_freq': False
}
```

Notes
-----
- `binary_high_freq` has precedence; if it is `True`, coarse segments are ignored even if `binary=True`.
- Normalization statistics include the generated high-frequency patterns; if you toggle modes later, consider regenerating stats.
- For deterministic experiments, rely on seeds already passed to RNG; flipping process is reproducible given the same seed.

Adjusting After Training
------------------------
If you change from coarse to high-frequency (or vice versa), you should retrain because the input distribution shift may degrade performance.

Happy experimenting!
