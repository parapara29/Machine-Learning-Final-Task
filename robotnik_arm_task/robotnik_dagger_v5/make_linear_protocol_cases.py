"""Write the 1,000 test cases of the linear-regression evaluation (test_combined_random.py).

The random draws, their order and the rounding are copied from that script (seed 42), so the
DAgger policy is tested on exactly the same spawn poses and targets as the linear model:
  spawn x, y ~ U(-2, 2) m, yaw ~ U(-pi, pi); target at a uniform direction and a distance of
  0.9-3.5 m from the spawn, height z ~ U(0.3, 0.9) m.
Usage:  python make_linear_protocol_cases.py [--out eval_v6/linear_protocol_cases.csv]
        python make_linear_protocol_cases.py --radius-min 2.0 --radius-max 4.0 --out eval_v6/cases_2.0-4.0.csv
The radius range must match RADIUS_MIN / RADIUS_MAX of the test_combined_random.py version being compared
(0.9-3.5 m until commit 79736c3 of shivansh/linear_regression, 2.0-4.0 m from commit 957d57e).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

NUM_TESTS, SEED, RADIUS_MIN, RADIUS_MAX = 1000, 42, 0.9, 3.5


def make_cases(n=NUM_TESTS, seed=SEED, radius_min=RADIUS_MIN, radius_max=RADIUS_MAX):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        sx = float(np.round(rng.uniform(-2.0, 2.0), 3))
        sy = float(np.round(rng.uniform(-2.0, 2.0), 3))
        syaw = float(np.round(rng.uniform(-np.pi, np.pi), 3))
        angle = rng.uniform(-np.pi, np.pi)
        radius = rng.uniform(radius_min, radius_max)
        tx = float(np.round(sx + radius * np.cos(angle), 3))
        ty = float(np.round(sy + radius * np.sin(angle), 3))
        tz = float(np.round(rng.uniform(0.3, 0.9), 3))
        rows.append(dict(episode=i, spawn_x=sx, spawn_y=sy, spawn_yaw=syaw,
                         target_x=tx, target_y=ty, target_z=tz, spawn_target_distance=float(radius)))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='eval_v6/linear_protocol_cases.csv')
    ap.add_argument('--radius-min', type=float, default=RADIUS_MIN)
    ap.add_argument('--radius-max', type=float, default=RADIUS_MAX)
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    make_cases(radius_min=args.radius_min, radius_max=args.radius_max).to_csv(out, index=False)
    print('wrote', out)
