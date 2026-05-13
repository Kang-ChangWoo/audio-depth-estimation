#!/usr/bin/env python3
"""Paired bootstrap CI for two test runs.

Each input is a per-sample npz produced by Round-4 test.py at
``eval/<dataset>/<eval_on>/per_sample_<exp>.npz``. Sample order is
identical across runs (shuffle=False), so we pair sample i of A with
sample i of B and bootstrap 10k times to get a 95 % CI on the mean
delta (B − A) for each metric.

Usage:
    paired_bootstrap.py A.npz B.npz [--metrics abs_rel,rmse,...]
                                    [--n-boot 10000] [--seed 0]

Reports `mean_delta` (B - A) so a *negative* abs_rel/rmse/log10/mae
delta means **B is better** on that metric, and a *positive* delta1
means **B is better** on δ1.
"""

import argparse
import sys

import numpy as np


# Metrics where lower is better; rest of mapping is "higher is better".
LOWER_IS_BETTER = {'abs_rel', 'rmse', 'log10', 'mae',
                   'abs_rel_sphere', 'rmse_sphere',
                   'log10_sphere', 'mae_sphere'}

DEFAULT_METRICS = (
    'abs_rel', 'rmse', 'delta1', 'log10', 'mae',
    'abs_rel_sphere', 'rmse_sphere', 'delta1_sphere',
)


def paired_bootstrap(delta, n_boot, rng):
    """delta: (N,) ndarray. Returns (mean, ci_lo, ci_hi)."""
    N = delta.shape[0]
    if N == 0:
        return float('nan'), float('nan'), float('nan')
    idx = rng.integers(0, N, size=(n_boot, N))
    means = delta[idx].mean(axis=1)
    return float(delta.mean()), float(np.percentile(means, 2.5)), \
        float(np.percentile(means, 97.5))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('a_npz', help='per_sample npz for run A (baseline)')
    p.add_argument('b_npz', help='per_sample npz for run B (candidate)')
    p.add_argument('--metrics', default=','.join(DEFAULT_METRICS),
                   help='comma-separated metric names to test')
    p.add_argument('--n-boot', type=int, default=10000,
                   help='bootstrap resamples (default 10000)')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args(argv)

    a = np.load(args.a_npz, allow_pickle=True)
    b = np.load(args.b_npz, allow_pickle=True)

    # Pair check: same number of samples and (when scene_id present)
    # same scene order.
    n_a = a['sample_id'].shape[0]
    n_b = b['sample_id'].shape[0]
    if n_a != n_b:
        print(f'ERROR: sample counts differ: A={n_a}  B={n_b}', file=sys.stderr)
        return 2
    if 'scene_id' in a.files and 'scene_id' in b.files:
        sa = a['scene_id'].astype(str)
        sb = b['scene_id'].astype(str)
        if not np.array_equal(sa, sb):
            print('ERROR: scene_id order differs between A and B; cannot pair',
                  file=sys.stderr)
            return 2

    rng = np.random.default_rng(args.seed)
    metrics = [m.strip() for m in args.metrics.split(',') if m.strip()]
    available = set(a.files) & set(b.files)

    print(f'Paired bootstrap (n_boot={args.n_boot}, n_samples={n_a})')
    print(f'  A = {args.a_npz}')
    print(f'  B = {args.b_npz}')
    print(f"{'metric':>20s}  {'mean_A':>9s}  {'mean_B':>9s}  "
          f"{'mean_Δ(B−A)':>12s}  {'ci_lo':>9s}  {'ci_hi':>9s}  verdict")
    print('-' * 95)

    for m in metrics:
        if m not in available:
            print(f'{m:>20s}  (missing in npz, skipped)')
            continue
        va = a[m].astype(np.float64)
        vb = b[m].astype(np.float64)
        if va.shape != vb.shape:
            print(f'{m:>20s}  shape mismatch {va.shape} vs {vb.shape}, skipped')
            continue
        delta = vb - va
        mean_d, lo, hi = paired_bootstrap(delta, args.n_boot, rng)
        better_when_lower = m in LOWER_IS_BETTER
        if better_when_lower:
            verdict = ('B better' if hi < 0 else
                       'A better' if lo > 0 else
                       'tie / inside CI')
        else:
            verdict = ('B better' if lo > 0 else
                       'A better' if hi < 0 else
                       'tie / inside CI')
        print(f'{m:>20s}  {va.mean():9.4f}  {vb.mean():9.4f}  '
              f'{mean_d:12.5f}  {lo:9.5f}  {hi:9.5f}  {verdict}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
