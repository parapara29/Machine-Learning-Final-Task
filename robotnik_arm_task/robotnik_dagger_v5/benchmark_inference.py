"""Time one control step of the V6 policy (feature transform + wheel net + one arm net) on CPU.

Single process, batch size 1, 1 torch thread: the setting of a deployment control loop.
Usage:  python benchmark_inference.py [--model-dir trained_model_dagger_v6] [--steps 20000]
Writes eval_v6/inference_benchmark.json.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from dagger_common import FEATURES
from v6_features import V6Policy

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', default='trained_model_dagger_v6')
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--out', default='eval_v6/inference_benchmark.json')
    args = ap.parse_args()
    torch.set_num_threads(1)
    policy = V6Policy(args.model_dir, device='cpu')
    rng = np.random.default_rng(0)
    xs = rng.normal(0, 0.5, size=(args.steps, len(FEATURES))).astype(np.float32)
    xs[:, FEATURES.index('navigation_complete')] = (np.arange(args.steps) > args.steps // 2)  # HOLD, then REACH
    for x in xs[:500]:                                                                          # warm-up
        policy.predict(x)
    times = np.empty(args.steps)
    for i, x in enumerate(xs):
        t0 = time.perf_counter(); policy.predict(x); times[i] = time.perf_counter() - t0
    res = {'mean_ms': 1e3 * times.mean(), 'median_ms': 1e3 * np.median(times),
           'p99_ms': 1e3 * np.percentile(times, 99), 'steps': args.steps, 'torch_threads': 1}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
