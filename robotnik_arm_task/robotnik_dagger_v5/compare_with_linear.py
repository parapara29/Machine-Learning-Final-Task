"""Compare the V6 DAgger policy with the linear-regression model on the linear model's own test.

The linear model is evaluated by test_combined_random.py on branch shivansh/linear_regression:
1000 random spawns (x, y in [-2, 2] m, any yaw) with a target 0.9-3.5 m away (seed 42), stopped
when the EE error is <= 1 cm, after a 1.5 s stall, or at 40000 steps; success = final EE error
<= 5 cm. The DAgger results come from run_v6_evaluation.sh, which uses exactly the same cases
(make_linear_protocol_cases.py), simulation settings and stop rule (--stop-rule linear).

This script computes, for both models, the summary metrics that test_combined_random.py prints
(success rate, mean / median / std / min / max EE error, average episode time over all episodes),
draws its summary figure (error histogram + targets relative to the start) for both, and adds a
paired comparison, since both models faced the same 1000 cases.

Inputs:  --linear-log  the console output of test_combined_random.py (one line per episode)
         eval_v6/linear_protocol/v6/deploy_summary.csv
Outputs: figures/linear_vs_dagger.{pdf,png} (4 panels), figures/linear_vs_dagger_distance.{pdf,png}
         (success per distance band, one column) and figures/linear_vs_dagger.json
Usage:   python compare_with_linear.py --linear-log linear_test.log
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from make_linear_protocol_cases import make_cases

HERE = Path(__file__).resolve().parent
SUCCESS_TOLERANCE = 0.05   # m, same as test_combined_random.py
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
GREEN, RED = '#1baf7a', '#d03b3b'

# One episode line of test_combined_random.py, e.g.
# [  12/1000] Start: ( 0.82,  1.12, yaw=-0.26) | Target: [1.968 1.652 0.369] | EE Error: 0.0425 m | Time:  8.06s | SUCCESS
LINE = re.compile(r'Start:\s*\(\s*(?P<sx>[-\d.]+),\s*(?P<sy>[-\d.]+),\s*yaw=\s*(?P<syaw>[-\d.]+)\)\s*\|\s*'
                  r'Target:\s*\[(?P<target>[^\]]+)\]\s*\|\s*EE Error:\s*(?P<err>[-\d.]+)\s*m\s*\|\s*'
                  r'Time:\s*(?P<t>[-\d.]+)s\s*\|\s*(?P<status>SUCCESS|FAIL)')


def load_linear(log_path, cases):
    """Parse the per-episode lines of the linear test and match them to the case table.

    The log prints episodes in completion order, so each line is matched to its case by the
    target (printed with all three decimals) and the start position (printed rounded to 2 decimals).
    """
    rows = []
    for line in Path(log_path).read_text().splitlines():
        m = LINE.search(line)
        if m:
            rows.append(dict(sx=float(m['sx']), sy=float(m['sy']), syaw=float(m['syaw']),
                             target=np.array(m['target'].split(), dtype=float),
                             final_error_m=float(m['err']), time_s=float(m['t']),
                             success=m['status'] == 'SUCCESS'))
    if len(rows) != len(cases):
        raise SystemExit(f'{log_path}: found {len(rows)} episode lines, expected {len(cases)}')
    episode = np.full(len(rows), -1)
    targets = cases[['target_x', 'target_y', 'target_z']].to_numpy()
    starts = cases[['spawn_x', 'spawn_y', 'spawn_yaw']].to_numpy()
    for i, r in enumerate(rows):
        match = np.flatnonzero((np.abs(targets - r['target']).max(1) < 1e-6)
                               & (np.abs(starts - [r['sx'], r['sy'], r['syaw']]).max(1) < 0.006))
        if len(match) != 1:
            raise SystemExit(f'could not match linear log line {i} to a unique case')
        episode[i] = match[0]
    if len(set(episode)) != len(cases):
        raise SystemExit('linear log does not cover every case exactly once')
    df = pd.DataFrame(rows).drop(columns=['sx', 'sy', 'syaw', 'target'])
    df['episode'] = episode
    return df.sort_values('episode').reset_index(drop=True)


def load_dagger(cases, summary_csv=HERE / 'eval_v6' / 'linear_protocol' / 'v6' / 'deploy_summary.csv'):
    df = pd.read_csv(summary_csv)
    df = df.sort_values('episode').reset_index(drop=True)
    if df['episode'].tolist() != cases['episode'].tolist():
        raise SystemExit('DAgger summary does not cover the 1000 linear-protocol cases')
    # The DAgger runner records the target of each episode; check it is the same case.
    if not np.allclose(df[['target_x', 'target_y', 'target_z']], cases[['target_x', 'target_y', 'target_z']]):
        raise SystemExit('DAgger targets differ from the linear-protocol cases')
    df['success'] = df['final_error_m'] <= SUCCESS_TOLERANCE
    return df[['episode', 'final_error_m', 'time_s', 'success']]


def summary(df):
    """The metrics printed by test_combined_random.py (errors in m, times in s)."""
    e, n, k = df['final_error_m'].to_numpy(), len(df), int(df['success'].sum())
    return {
        'total_points_tested': n, 'successful_reaches': k, 'failed_reaches': n - k,
        'success_rate_percent': 100.0 * k / n,
        'mean_ee_error_m': float(e.mean()), 'median_ee_error_m': float(np.median(e)),
        'std_ee_error_m': float(e.std()), 'min_ee_error_m': float(e.min()), 'max_ee_error_m': float(e.max()),
        'average_episode_time_s': float(df['time_s'].mean()),
    }


def wilson(k, n, z=1.96):
    """95 % Wilson score interval of a success rate, in percent."""
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [100 * (c - h), 100 * (c + h)]


def plot(results, cases, out, radius_min=0.9, radius_max=3.5):
    """The two-panel summary figure of test_combined_random.py, one row per model, shared axes."""
    rel = cases[['target_x', 'target_y']].to_numpy() - cases[['spawn_x', 'spawn_y']].to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), gridspec_kw={'width_ratios': [1.25, 1]})
    emax = max(df['final_error_m'].max() for df in results.values())
    bins = np.linspace(0, emax, 31)
    ymax = max(np.histogram(df['final_error_m'], bins)[0].max() for df in results.values()) * 1.08
    for row, (name, df) in enumerate(results.items()):
        s = summary(df)
        ok = df['success'].to_numpy()
        ax = axes[row, 0]
        ax.hist(df['final_error_m'], bins=bins, color='#2a78d6', edgecolor='white', linewidth=0.6)
        ax.axvline(SUCCESS_TOLERANCE, color=INK, ls='--', lw=1, label=f'success ≤ {SUCCESS_TOLERANCE} m')
        ax.set_ylim(0, ymax)
        ax.set_xlabel('final EE error (m)')
        ax.set_ylabel('episodes')
        ax.set_title(f'{name}: final EE error (success rate {s["success_rate_percent"]:.1f} %)', loc='left')
        ax.legend()
        ax = axes[row, 1]
        ax.scatter(rel[ok, 0], rel[ok, 1], s=10, color=GREEN, label=f'success ({ok.sum()})', linewidths=0)
        ax.scatter(rel[~ok, 0], rel[~ok, 1], s=22, color=RED, marker='x', label=f'fail ({(~ok).sum()})', linewidths=1.2)
        ax.set_aspect('equal')
        ax.set_xlabel('target offset dX from start (m)')
        ax.set_ylabel('target offset dY from start (m)')
        ax.set_title(f'{name}: target offset from the start ({radius_min}–{radius_max} m)', loc='left')
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))
    for ax in axes.flat:
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out.with_suffix('.pdf'))
    fig.savefig(out.with_suffix('.png'), dpi=150)


def plot_distance(report, out):
    """One-column figure for the report: success rate per spawn-target distance band, both models."""
    bands = list(report['success_by_distance_percent'])
    names = list(report['success_by_distance_percent'][bands[0]])
    colors = ['#eb6834', '#2a78d6']          # validated categorical slots: linear orange, DAgger blue
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    x, w = np.arange(len(bands)), 0.38
    for k, (name, color) in enumerate(zip(names, colors)):
        vals = [report['success_by_distance_percent'][b][name] for b in bands]
        bars = ax.bar(x + (k - 0.5) * (w + 0.02), vals, w, color=color, label=f'{name} ({report[name]["success_rate_percent"]:.1f} %)')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f'{v:.0f}', ha='center', va='bottom', fontsize=7, color=INK2)
    ax.set_xticks(x, [b.replace(' m (n=', ' m\n(n=') for b in bands], fontsize=7)
    ax.set_ylabel('success rate (%)', fontsize=8)
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.grid(axis='y', color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel('distance from spawn to target', fontsize=8)
    fig.tight_layout(pad=0.3)
    fig.savefig(out.with_suffix('.pdf'))
    fig.savefig(out.with_suffix('.png'), dpi=200)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--linear-log', required=True, help='console output of test_combined_random.py')
    ap.add_argument('--out', default=str(HERE / 'figures' / 'linear_vs_dagger'))
    ap.add_argument('--dagger-summary', default=str(HERE / 'eval_v6' / 'linear_protocol' / 'v6' / 'deploy_summary.csv'),
                    help='deploy_summary.csv of the DAgger run on the same cases')
    ap.add_argument('--radius-min', type=float, default=0.9, help='RADIUS_MIN of the test_combined_random.py run')
    ap.add_argument('--radius-max', type=float, default=3.5, help='RADIUS_MAX of the test_combined_random.py run')
    args = ap.parse_args()
    cases = make_cases(radius_min=args.radius_min, radius_max=args.radius_max)
    results = {'Linear regression': load_linear(args.linear_log, cases),
               'DAgger V6': load_dagger(cases, args.dagger_summary)}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    report = {}
    for name, df in results.items():
        s = summary(df)
        s['success_rate_95ci_percent'] = wilson(s['successful_reaches'], s['total_points_tested'])
        s['within_2cm_percent'] = float(100 * (df['final_error_m'] <= 0.02).mean())
        s['within_1cm_percent'] = float(100 * (df['final_error_m'] <= 0.01).mean())
        s['average_episode_time_successes_s'] = float(df.loc[df['success'], 'time_s'].mean())
        report[name] = s

    # Paired view: both models faced the same 1000 cases.
    lin, dag = (results[k]['success'].to_numpy() for k in results)
    report['paired'] = {
        'both_succeed': int((lin & dag).sum()), 'only_linear_succeeds': int((lin & ~dag).sum()),
        'only_dagger_succeeds': int((~lin & dag).sum()), 'both_fail': int((~lin & ~dag).sum()),
    }
    # Success by spawn-target distance (both models fail mostly on the near targets).
    edges = [e for e in [1.3, 2.0, 2.5, 3.0] if args.radius_min < e < args.radius_max]
    edges = [args.radius_min, *edges, args.radius_max]
    dist = cases['spawn_target_distance'].to_numpy()
    report['success_by_distance_percent'] = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        band = (dist >= lo) & ((dist < hi) | (hi == args.radius_max))
        report['success_by_distance_percent'][f'{lo}-{hi} m (n={int(band.sum())})'] = {
            name: float(100 * df['success'].to_numpy()[band].mean()) for name, df in results.items()}
    # Success for all targets at or beyond a distance (e.g. "100 % above 1.3 m").
    report['success_at_or_beyond_percent'] = {
        f'>= {t} m (n={int((dist >= t).sum())})': {
            name: float(100 * df['success'].to_numpy()[dist >= t].mean()) for name, df in results.items()}
        for t in [1.3, 1.8, 2.0, 2.5] if args.radius_min <= t < args.radius_max}
    out.with_suffix('.json').write_text(json.dumps(report, indent=2))
    plot(results, cases, out, args.radius_min, args.radius_max)
    plot_distance(report, out.with_name(out.name + '_distance'))

    names = list(results)
    rows = [('Success rate (%)', 'success_rate_percent', '{:.2f}'),
            ('Successful / failed', None, None),
            ('Mean EE error (m)', 'mean_ee_error_m', '{:.4f}'),
            ('Median EE error (m)', 'median_ee_error_m', '{:.4f}'),
            ('Std EE error (m)', 'std_ee_error_m', '{:.4f}'),
            ('Min EE error (m)', 'min_ee_error_m', '{:.4f}'),
            ('Max EE error (m)', 'max_ee_error_m', '{:.4f}'),
            ('Average episode time (s)', 'average_episode_time_s', '{:.2f}')]
    print(f'{"metric":28s}' + ''.join(f'{n:>20s}' for n in names))
    for label, key, fmt in rows:
        if key is None:
            vals = [f'{report[n]["successful_reaches"]} / {report[n]["failed_reaches"]}' for n in names]
        else:
            vals = [fmt.format(report[n][key]) for n in names]
        print(f'{label:28s}' + ''.join(f'{v:>20s}' for v in vals))
    print('paired:', report['paired'])
    for key in ('success_by_distance_percent', 'success_at_or_beyond_percent'):
        for band, rates in report[key].items():
            print(f'success {band:22s}' + ''.join(f'{v:>20.1f}' for v in rates.values()))
    print('wrote', out.with_suffix('.json'), out.with_suffix('.png'), out.with_suffix('.pdf'))


if __name__ == '__main__':
    main()
