"""Figures + metrics for the V6 DAgger policy (run by run_v6_evaluation.sh, last stage).

Inputs:  eval_v6/linear_protocol/v6/deploy_summary.csv       V6 on the 1,000 linear-protocol cases
         eval_v6/linear_protocol/{r0,r1}/deploy_summary.csv  DAgger round checkpoints, cases 0-199
         eval_v6/examples/deploy_*.csv                       saved example trajectories
         trained_model_dagger_v6{_r0,_r1,}/*_training_history.csv
Outputs: figures/v6_*.pdf (+ .png previews) and figures/v6_results.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE / 'figures'
EVAL = HERE / 'eval_v6' / 'linear_protocol'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'          # validated categorical slots 1-3
ROUND_COLORS = ['#86b6ef', '#2a78d6', '#184f95']            # sequential blue: round order
ROUNDS = [('r0', 'trained_model_dagger_v6_r0', 'Round 0\n(expert data)'),
          ('r1', 'trained_model_dagger_v6_r1', 'Round 1\n($\\beta=0.5$)'),
          ('v6', 'trained_model_dagger_v6', 'Round 2, final\n($\\beta=0$)')]
TOL, STRICT, REACH = 0.05, 0.02, 0.01
N_ROUND_CASES = 200

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 9.5, 'axes.labelsize': 9, 'legend.fontsize': 8,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
    'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'legend.frameon': False,
    'pdf.fonttype': 42,
})


def load(path):
    """Read a deploy_summary.csv (one row per episode); keep the last row per episode."""
    if not Path(path).is_file():
        return None
    df = pd.read_csv(path)
    df['success'] = df['success'].astype(str).str.lower().eq('true')
    return df.drop_duplicates('episode', keep='last').sort_values('episode').reset_index(drop=True)


def wilson(k, n, z=1.96):
    """95 % Wilson score interval for k successes out of n (better than +-1.96*sqrt(p(1-p)/n) near 0 or 1)."""
    p = k / n; d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return c - h, c + h


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(OUT / f'{name}.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def n_params(folder):
    """Number of trainable parameters of each network in a checkpoint folder."""
    return {name: int(sum(v.numel() for v in torch.load(HERE / folder / f'{name}.pth', map_location='cpu',
                                                         weights_only=True).values()))
            for name in ('wheel', 'arm_hold', 'arm_reach')}


def metrics(df):
    """The metrics of the linear-regression table (success <= 5 cm, <= 2 cm, mean/median/max error, ...)."""
    e = df['final_error_m'].to_numpy(float)
    k, n = int(df['success'].sum()), len(df)
    succ = df[df['success']]
    return {'episodes': n, 'successes': k, 'success_rate': k / n, 'wilson95': list(wilson(k, n)),
            'strict_success_rate_2cm': float(np.mean(e <= STRICT)),
            'within_1cm_rate': float(np.mean(e <= REACH)),
            'mean_error_m': float(e.mean()), 'median_error_m': float(np.median(e)),
            'max_error_m': float(e.max()), 'p95_error_m': float(np.percentile(e, 95)),
            'successes_mean_error_m': float(succ['final_error_m'].mean()) if k else None,
            'successes_max_error_m': float(succ['final_error_m'].max()) if k else None,
            'mean_episode_time_s': float(df['time_s'].mean()),
            'mean_time_successes_s': float(succ['time_s'].mean()) if k else None,
            'inference_ms_per_step': float(df['inference_ms_per_step'].mean()),
            'stop_reasons': df['stop_reason'].value_counts().to_dict()}


# ----------------------------------------------------------------------------- figures
def fig_rounds(rounds):
    """Closed-loop success of each DAgger round's checkpoint on the same 200 cases."""
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    for i, ((key, _, label), color) in enumerate(zip(ROUNDS, ROUND_COLORS)):
        df = rounds.get(key)
        if df is None:
            continue
        k, n = int(df['success'].sum()), len(df)
        lo, hi = wilson(k, n)
        ax.bar(i, 100 * k / n, 0.62, color=color, edgecolor='white', linewidth=2)
        ax.errorbar(i, 100 * k / n, yerr=[[100 * (k / n - lo)], [100 * (hi - k / n)]], color=INK2, lw=1, capsize=3)
        ax.text(i, 100 * hi + 2, f'{100*k/n:.1f}%\n({k}/{n})', ha='center', va='bottom', fontsize=7.5, color=INK)
    ax.set_xticks(range(len(ROUNDS)), [r[2] for r in ROUNDS], fontsize=7.5)
    ax.set_ylim(0, 120); ax.set_yticks(range(0, 101, 20)); ax.grid(axis='x', visible=False)
    ax.set_ylabel('Success rate (error $\\leq$ 5 cm) [%]')
    fig.tight_layout(); save(fig, 'v6_dagger_rounds')


def fig_error(df):
    """Final-error distribution on the 1,000 cases (CDF, log x) with the 1/2/5 cm thresholds."""
    e = np.sort(np.clip(df['final_error_m'].to_numpy(float), 1e-3, None))
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.step(e, np.arange(1, len(e) + 1) / len(e) * 100, where='post', color=BLUE, lw=2)
    for thr, lab in ((REACH, '1 cm'), (STRICT, '2 cm'), (TOL, '5 cm')):
        pct = 100 * np.mean(e <= thr)
        ax.axvline(thr, color=INK2, lw=0.9, ls='--')
        ax.text(thr * 1.07, max(pct - 12, 3), f'{lab}:\n{pct:.1f}%', fontsize=7.5, color=INK)
    ax.set_xscale('log'); ax.set_ylim(0, 102)
    ax.set_xlabel('Final end-effector error [m]'); ax.set_ylabel('Episodes with error $\\leq x$ [%]')
    fig.tight_layout(); save(fig, 'v6_final_error_cdf')


def fig_map(df, cases):
    """Targets in the robot's initial body frame (as in the linear test), success vs. failure,
    and success rate against the spawn-target distance."""
    c = cases.set_index('episode').loc[df['episode']]
    dx, dy = c['target_x'] - c['spawn_x'], c['target_y'] - c['spawn_y']
    cy, sy = np.cos(c['spawn_yaw']), np.sin(c['spawn_yaw'])
    fwd, left = (cy * dx + sy * dy).to_numpy(), (-sy * dx + cy * dy).to_numpy()
    ok = df['success'].to_numpy()
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={'width_ratios': [1.05, 1]})
    a.scatter(left[ok], fwd[ok], s=9, color=BLUE, alpha=0.55, lw=0, label=f'success ({ok.sum()})')
    a.scatter(left[~ok], fwd[~ok], s=22, color=ORANGE, marker='x', lw=1.3, label=f'failure ({(~ok).sum()})')
    a.scatter([0], [0], s=60, marker='^', color=INK, label='robot at spawn (facing up)')
    a.set_aspect('equal'); a.invert_xaxis()
    a.set_xlabel('Target to the left of the robot [m]'); a.set_ylabel('Target ahead of the robot [m]')
    a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=7.5)
    a.set_title('(a) Target position at spawn', loc='left', color=INK)
    d = spawn_goal_distance(c)
    bins = np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5, 4.5])
    centers = 0.5 * (bins[:-1] + bins[1:])
    rate, lo_, hi_ = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (d >= lo) & (d <= hi)
        k, n = int(ok[m].sum()), int(m.sum())
        rate.append(100 * k / n); w = wilson(k, n); lo_.append(100 * w[0]); hi_.append(100 * w[1])
    b.fill_between(centers, lo_, hi_, color=BLUE, alpha=0.15, lw=0, label='95% Wilson interval')
    b.plot(centers, rate, '-o', color=BLUE, lw=2, ms=6, markeredgecolor='white', markeredgewidth=1.5,
           label='success rate')
    b.set_xticks(centers, [f'{x:g}-\n{y:g}' for x, y in zip(bins[:-1], bins[1:])], fontsize=7.5)
    b.set_ylim(0, 105); b.set_xlabel('Distance from spawn to base goal [m]'); b.set_ylabel('Success rate [%]')
    b.legend(loc='lower left', fontsize=7.5)
    b.set_title('(b) Success against spawn-goal distance', loc='left', color=INK)
    fig.tight_layout(); save(fig, 'v6_success_map')


def spawn_goal_distance(c):
    """Distance from the spawn to the expert's base goal (standoff pose on the origin-target ray)."""
    standoff = np.interp(c['target_z'], [.1, .5, 1.1, 1.25], [.8, .65, .65, .35])
    bearing = np.arctan2(c['target_y'], c['target_x'])
    gx, gy = c['target_x'] - standoff * np.cos(bearing), c['target_y'] - standoff * np.sin(bearing)
    return np.hypot(gx - c['spawn_x'], gy - c['spawn_y']).to_numpy()


def fig_training():
    """Validation MSE per epoch for each DAgger round: demonstration hold-out (solid) and
    DAgger-rollout hold-out (dashed)."""
    names = [('wheel', 'Wheel network'), ('arm_hold', 'Arm HOLD network'), ('arm_reach', 'Arm REACH network')]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for ax, (name, title) in zip(axes, names):
        for (key, folder, label), color in zip(ROUNDS, ROUND_COLORS):
            p = HERE / folder / f'{name}_training_history.csv'
            if not p.is_file():
                continue
            h = pd.read_csv(p)
            ax.plot(h['epoch'], h['val_demo_mse'], color=color, lw=2, label=label.replace('\n', ' '))
            if 'val_dagger_mse' in h:
                ax.plot(h['epoch'], h['val_dagger_mse'], color=color, lw=1.5, ls='--')
        ax.set_yscale('log'); ax.set_title(title, loc='left', color=INK); ax.set_xlabel('Epoch')
    axes[0].set_ylabel('Validation MSE')
    axes[0].plot([], [], color=INK2, lw=2, label='demo hold-out')
    axes[0].plot([], [], color=INK2, lw=1.5, ls='--', label='DAgger hold-out')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=5, fontsize=7,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.08, 1, 1)); save(fig, 'v6_training_curves')


def fig_examples():
    files = sorted((HERE / 'eval_v6' / 'examples').glob('deploy_*.csv'))[:3]
    if not files:
        return
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for f, color in zip(files, (BLUE, ORANGE, AQUA)):
        t = pd.read_csv(f)
        eid = int(f.stem.split('_')[1])
        a.plot(t['base_x'], t['base_y'], color=color, lw=2, label=f'episode {eid}')
        a.scatter(t['base_x'].iloc[0], t['base_y'].iloc[0], s=40, color=color, edgecolor='white', zorder=3)
        a.scatter(t['goal_x'].iloc[0], t['goal_y'].iloc[0], s=55, marker='s', facecolor='white',
                  edgecolor=color, lw=1.6, zorder=3)
        a.scatter(t['target_x'].iloc[0], t['target_y'].iloc[0], s=90, marker='*', color=color,
                  edgecolor='white', lw=0.6, zorder=4)
        b.plot(t['time'], t['ee_error'], color=color, lw=2, label=f'episode {eid}')
        nav = t.index[t['navigation_complete'] >= 0.5]
        if len(nav):
            b.axvline(t['time'].iloc[nav[0]], color=color, lw=0.9, ls=':')
    a.set_aspect('equal', adjustable='datalim')
    a.set_xlabel('World x [m]'); a.set_ylabel('World y [m]')
    a.scatter([], [], s=40, color=INK2, label='spawn'); a.scatter([], [], s=55, marker='s', facecolor='white',
                                                                  edgecolor=INK2, label='base goal')
    a.scatter([], [], s=90, marker='*', color=INK2, label='target (x, y)')
    a.legend(fontsize=7, loc='best'); a.set_title('(a) Base path (top view)', loc='left', color=INK)
    b.axhline(TOL, color=INK2, lw=0.9, ls='--'); b.text(0.3, TOL * 1.15, '5 cm', fontsize=7.5, color=INK2)
    b.set_yscale('log'); b.set_xlabel('Time [s]'); b.set_ylabel('End-effector error [m]')
    b.set_title('(b) Error (dotted: navigation complete)', loc='left', color=INK)
    fig.tight_layout(); save(fig, 'v6_example_episodes')


def main():
    OUT.mkdir(exist_ok=True)
    cases = pd.read_csv(HERE / 'eval_v6' / 'linear_protocol_cases.csv')
    final = load(EVAL / 'v6' / 'deploy_summary.csv')
    rounds = {k: load(EVAL / k / 'deploy_summary.csv') for k, _, _ in ROUNDS[:2]}
    rounds['v6'] = final[final['episode'] < N_ROUND_CASES]
    rounds = {k: v[v['episode'] < N_ROUND_CASES] for k, v in rounds.items() if v is not None}
    fig_rounds(rounds); fig_error(final); fig_map(final, cases); fig_training(); fig_examples()

    params = n_params('trained_model_dagger_v6')
    res = {'linear_protocol_1000': metrics(final),
           'dagger_rounds_first_200_cases': {k: metrics(v) for k, v in rounds.items()},
           'parameters': {**params, 'total': sum(params.values())},
           'best_validation_mse': {}}
    bench = HERE / 'eval_v6' / 'inference_benchmark.json'
    if bench.is_file():
        res['inference_benchmark'] = json.loads(bench.read_text())
    for key, folder, _ in ROUNDS:
        for name in ('wheel', 'arm_hold', 'arm_reach'):
            h = pd.read_csv(HERE / folder / f'{name}_training_history.csv')
            i = int(h['val_mean_mse'].idxmin())
            res['best_validation_mse'].setdefault(key, {})[name] = {
                'demo': float(h['val_demo_mse'][i]), 'dagger': float(h.get('val_dagger_mse', pd.Series([np.nan] * len(h)))[i]),
                'epochs_run': int(len(h))}
    (OUT / 'v6_results.json').write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))


if __name__ == '__main__':
    main()
