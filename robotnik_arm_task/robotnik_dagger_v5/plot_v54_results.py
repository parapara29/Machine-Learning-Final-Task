"""Figures + summary numbers comparing the V5.3-A and V5.4 DAgger checkpoints.

Inputs (produced by run_v54_evaluation.sh and train_arm_v5.py):
  trained_model_dagger_{v5_3_A,v5_4}/arm_{hold,reach}_v4_training_history.csv
  eval_v53_validation_fresh/deploy_summary.csv          (V5.3-A, 300 validation targets)
  eval_v54_validation/shard_*/deploy_summary.csv         (V5.4,   300 validation targets)
  eval_random/{v5_3_A,v5_4}/shard_*/deploy_summary.csv   (both, same 100 random targets+spawns)
Outputs: figures/*.pdf (+ .png previews) and figures/v54_results.json
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.ticker
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / 'figures'
MODELS = {'v5_3_A': ('V5.3-A', '#2a78d6'), 'v5_4': ('V5.4', '#eb6834')}  # validated categorical slots 1, 2
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
TOL = 0.05

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 9.5, 'axes.labelsize': 9, 'legend.fontsize': 8.5,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
    'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'legend.frameon': False,
    'pdf.fonttype': 42,
})


def load_summary(paths):
    frames = [pd.read_csv(p) for p in paths if Path(p).is_file()]
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df['success'] = df['success'].astype(str).str.lower().eq('true')
    # Some collector versions append; keep the last row per episode label + target.
    return df.drop_duplicates(['episode', 'target_x', 'target_y', 'target_z'], keep='last').reset_index(drop=True)


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (c - h, c + h)


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(OUT / f'{name}.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def fig_training():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for ax, phase in zip(axes, ('hold', 'reach')):
        for key, (label, color) in MODELS.items():
            p = HERE / f'trained_model_dagger_{key}' / f'arm_{phase}_v4_training_history.csv'
            if not p.is_file():
                continue
            h = pd.read_csv(p)
            ax.plot(h['epoch'], h['validation_mse'], color=color, lw=2, label=label)
            i = int(h['validation_mse'].idxmin())
            ax.plot(h['epoch'][i], h['validation_mse'][i], 'o', ms=6, color=color,
                    markeredgecolor='white', markeredgewidth=1.5, zorder=3)
            ax.annotate(f"{h['validation_mse'][i]:.3f}", (h['epoch'][i], h['validation_mse'][i]),
                        xytext=(4, 6), textcoords='offset points', fontsize=8, color=INK)
        ax.set_yscale('log')
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1.0, 3.0)))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
        ax.set_title(f'{phase.upper()} network', loc='left', color=INK)
        ax.set_xlabel('Epoch')
    axes[0].set_ylabel(r'Validation MSE  [(N$\cdot$m)$^2$]')
    axes[1].legend(loc='upper right')
    fig.tight_layout()
    save(fig, 'v54_training_curves')


def fig_success(evals):
    groups = [g for g in evals if any(evals[g].get(k) is not None for k in MODELS)]
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    width = 0.36
    for j, (key, (label, color)) in enumerate(MODELS.items()):
        for i, g in enumerate(groups):
            df = evals[g].get(key)
            if df is None:
                continue
            k, n = int(df['success'].sum()), len(df)
            lo, hi = wilson(k, n)
            x = i + (j - 0.5) * (width + 0.04)
            ax.bar(x, 100 * k / n, width, color=color, label=label if i == 0 else None,
                   edgecolor='white', linewidth=2)
            ax.errorbar(x, 100 * k / n, yerr=[[100 * (k / n - lo)], [100 * (hi - k / n)]],
                        color=INK2, lw=1, capsize=3)
            ax.text(x, 100 * hi + 1.5, f'{100*k/n:.1f}%\n({k}/{n})', ha='center', va='bottom',
                    fontsize=7.5, color=INK)
    ax.set_xticks(range(len(groups)), groups)
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel('Success rate [%]')
    ax.grid(axis='x', visible=False)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout()
    save(fig, 'v54_success_rate')


def fig_error_cdf(evals):
    groups = [g for g in evals if any(evals[g].get(k) is not None for k in MODELS)]
    fig, axes = plt.subplots(1, len(groups), figsize=(3.5 * len(groups), 2.7), squeeze=False)
    for ax, g in zip(axes[0], groups):
        for key, (label, color) in MODELS.items():
            df = evals[g].get(key)
            if df is None:
                continue
            e = np.sort(np.clip(df['final_error_m'].to_numpy(float), 1e-3, None))
            ax.step(e, np.arange(1, len(e) + 1) / len(e) * 100, where='post', color=color, lw=2,
                    label=f'{label} (median {1000*np.median(e):.0f} mm)')
        ax.axvline(TOL, color=INK2, lw=1, ls='--')
        ax.text(TOL * 1.08, 4, 'tolerance\n5 cm', fontsize=7.5, color=INK2)
        ax.set_xscale('log')
        ax.set_xlabel('Final end-effector error [m]')
        ax.set_title(g, loc='left', color=INK)
        ax.set_ylim(0, 102)
        ax.legend(loc='center right', fontsize=7.5)
    axes[0, 0].set_ylabel('Episodes with error $\\leq x$ [%]')
    fig.tight_layout()
    save(fig, 'v54_final_error_cdf')


def fig_stop_reasons(evals):
    groups = [g for g in evals if any(evals[g].get(k) is not None for k in MODELS)]
    reasons = ['reached', 'step_cap', 'velocity_guard', 'joint_guard']
    names = {'reached': 'Reached', 'step_cap': 'Step cap', 'velocity_guard': 'Velocity guard',
             'joint_guard': 'Joint guard'}
    fig, axes = plt.subplots(1, len(groups), figsize=(3.5 * len(groups), 2.4), squeeze=False)
    h = 0.36
    for ax, g in zip(axes[0], groups):
        for j, (key, (label, color)) in enumerate(MODELS.items()):
            df = evals[g].get(key)
            if df is None:
                continue
            counts = [int((df['stop_reason'] == r).sum()) for r in reasons]
            y = np.arange(len(reasons)) + (j - 0.5) * (h + 0.04)
            ax.barh(y, counts, h, color=color, label=label, edgecolor='white', linewidth=2)
            for yy, c in zip(y, counts):
                ax.text(c + 1, yy, str(c), va='center', fontsize=7.5, color=INK)
        ax.set_yticks(range(len(reasons)), [names[r] for r in reasons])
        ax.invert_yaxis()
        ax.set_xlabel('Episodes')
        ax.set_title(g, loc='left', color=INK)
        ax.grid(axis='y', visible=False)
    axes[0, -1].legend(loc='lower right')
    fig.tight_layout()
    save(fig, 'v54_stop_reasons')


def stats(df):
    if df is None:
        return None
    k, n = int(df['success'].sum()), len(df)
    lo, hi = wilson(k, n)
    succ = df[df['success']]
    return {'episodes': n, 'successes': k, 'success_rate': k / n, 'wilson95': [lo, hi],
            'median_final_error_m': float(df['final_error_m'].median()),
            'mean_final_error_m': float(df['final_error_m'].mean()),
            'mean_time_to_success_s': float(succ['time_s'].mean()) if len(succ) else None,
            'stop_reasons': df['stop_reason'].value_counts().to_dict(),
            'velocity_jump_anomaly_episodes': int(df['velocity_jump_anomaly'].notna().sum())}


def main():
    evals = {
        'Validation targets': {
            'v5_3_A': load_summary([HERE / 'eval_v53_validation_fresh/deploy_summary.csv']),
            'v5_4': load_summary(sorted(HERE.glob('eval_v54_validation/shard_*/deploy_summary.csv'))),
        },
        'Random target + spawn': {
            k: load_summary(sorted(HERE.glob(f'eval_random/{k}/shard_*/deploy_summary.csv'))) for k in MODELS
        },
    }
    fig_training()
    fig_success(evals)
    fig_error_cdf(evals)
    fig_stop_reasons(evals)
    res = {g: {k: stats(v) for k, v in d.items()} for g, d in evals.items()}
    # Paired comparison on the validation targets (same episode IDs).
    a, b = evals['Validation targets']['v5_3_A'], evals['Validation targets']['v5_4']
    if a is not None and b is not None:
        m = a[['episode', 'success']].merge(b[['episode', 'success']], on='episode', suffixes=('_a', '_b'))
        res['validation_paired'] = {
            'both_success': int((m.success_a & m.success_b).sum()),
            'fixed_by_v5_4': int((~m.success_a & m.success_b).sum()),
            'regressed_in_v5_4': int((m.success_a & ~m.success_b).sum()),
            'both_fail': int((~m.success_a & ~m.success_b).sum())}
    for key in MODELS:
        p = HERE / f'trained_model_dagger_{key}/arm_v4_validation.json'
        if p.is_file():
            res.setdefault('best_validation_mse', {})[key] = json.loads(p.read_text())
    (OUT / 'v54_results.json').write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))


if __name__ == '__main__':
    main()
