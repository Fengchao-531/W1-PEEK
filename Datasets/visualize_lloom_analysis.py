"""Rebuild Domain stability and the current consolidated topic reports."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

def save(fig, out, name):
    fig.savefig(out / f'{name}.png', dpi=180, bbox_inches='tight')
    fig.savefig(out / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def stability(root, plan, out):
    summaries, emails = [], []
    for batch in sorted(root.glob('b[0-9][0-9][0-9]')):
        folder = batch / '03_stability'
        for i in range(1, plan['repeats'] + 1):
            if json.loads((folder / f'run{i:02}' / 'status.json').read_text())['state'] != 'complete':
                raise ValueError(f'Incomplete stability run in {folder}')
        summary = pd.read_csv(folder / 'summary.csv')
        by_email = pd.read_csv(folder / 'by_email.csv', dtype={'row_id': str})
        if len(by_email) != plan['sizes']['stable50'] or by_email.row_id.duplicated().any():
            raise ValueError(f'Invalid fixed stability sample: {folder}')
        if len(summary.loc[(summary.kind == 'pairwise') & (summary.comparison != 'overall')]) != plan['repeats'] * (plan['repeats'] - 1) // 2:
            raise ValueError(f'Missing run-pair comparisons: {folder}')
        summaries.append(summary.assign(batch_id=batch.name))
        emails.append(by_email.assign(batch_id=batch.name))
    summaries = pd.concat(summaries, ignore_index=True)
    emails = pd.concat(emails, ignore_index=True)
    summaries.to_csv(out / 'stability_all_summaries.csv', index=False)
    emails.to_csv(out / 'stability_all_emails.csv', index=False)
    overall = summaries.loc[summaries.comparison.eq('overall')]
    batches = list(overall.batch_id.unique())
    x = np.arange(len(batches))
    fig, axes = plt.subplots(2, 1, figsize=(19, 10), sharex=True)
    for ax, metric, title in zip(axes, ['mean_jaccard', 'exact_match_rate'], ['Mean label-set Jaccard', 'Exact label-set match rate']):
        for kind, color, label in [('pairwise', '#2878b5', 'Between five repeat runs'), ('vs_initial', '#e1812c', 'Repeat runs vs initial annotation')]:
            group = overall.loc[overall.kind.eq(kind)].set_index('batch_id').loc[batches]
            ax.plot(x, group[metric], 'o-', markersize=3, color=color, label=label, linewidth=1)
        if metric == 'mean_jaccard':
            group = overall.loc[overall.kind.eq('pairwise')].set_index('batch_id').loc[batches]
            ax.plot(x, group.mean_jaccard_excluding_both_unknown, '--', color='#3d9855', label='Between repeats, excluding both-Unknown pairs', linewidth=1)
        ax.set(ylabel=title, ylim=(0, 1.02))
        ax.grid(axis='y', alpha=.2)
        ax.legend(fontsize=9, loc='lower left')
    axes[-1].set(xticks=x, xticklabels=batches, xlabel='Batch ID (independent email groups; not training steps)')
    axes[-1].tick_params(axis='x', labelrotation=90)
    fig.suptitle('LLooM stability across all batches | same fixed emails in all five repeats', fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .96))
    save(fig, out, 'stability_batch_curves')

    fig, axes = plt.subplots(2, 1, figsize=(21, 11), sharex=True)
    for ax, metric, title in zip(axes, ['mean_pairwise_jaccard', 'mean_vs_initial_jaccard'],
                               ['Between repeats: per-email mean of 10 pairwise Jaccards', 'Against initial annotation: per-email mean of 5 Jaccards']):
        data = [emails.loc[emails.batch_id.eq(batch), metric].to_numpy() for batch in batches]
        ax.boxplot(data, positions=x, widths=.65, showmeans=True, meanline=True, whis=1.5,
                   flierprops={'marker': '.', 'markersize': 3, 'alpha': .5},
                   medianprops={'color': '#2878b5'}, meanprops={'color': '#e1812c'})
        ax.set(ylabel='Jaccard', title=title, ylim=(-.02, 1.02))
        ax.grid(axis='y', alpha=.2)
    axes[-1].set(xticks=x, xticklabels=batches, xlabel='Each box contains 50 email-level means; blue = median, dashed orange = mean')
    axes[-1].tick_params(axis='x', labelrotation=90)
    fig.suptitle('Email-level stability distributions | both Unknown gives Jaccard = 1\nRun pairs reuse the same emails; they are not independent samples', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .94))
    save(fig, out, 'stability_batch_boxplots')

    repeat_table = summaries.loc[summaries.kind.eq('vs_initial') & ~summaries.comparison.eq('overall')].pivot(
        index='batch_id', columns='comparison', values='mean_jaccard').loc[batches]
    fig, ax = plt.subplots(figsize=(8, 16))
    im = ax.imshow(repeat_table.to_numpy(), vmin=0, vmax=1, aspect='auto', cmap='YlGnBu')
    for i in range(len(batches)):
        for j in range(plan['repeats']):
            value = repeat_table.iloc[i, j]
            ax.text(j, i, f'{value:.3f}', ha='center', va='center', fontsize=7, color='white' if value > .65 else 'black')
    ax.set(xticks=range(plan['repeats']), xticklabels=repeat_table.columns, yticks=range(len(batches)),
           yticklabels=batches, title='Mean Jaccard vs initial annotation\nSame 50 emails per batch')
    fig.colorbar(im, ax=ax, label='Mean Jaccard', shrink=.5)
    fig.tight_layout()
    save(fig, out, 'stability_repeat_heatmap')
    metrics = {}
    for kind in ['pairwise', 'vs_initial']:
        data = overall.loc[overall.kind.eq(kind)]
        weights = data.email_comparisons.to_numpy()
        metrics[kind] = {metric: float(np.average(data[metric], weights=weights))
                         for metric in ['mean_jaccard', 'exact_match_rate']}
        valid = data.mean_jaccard_excluding_both_unknown.notna()
        weights_nonempty = (data.email_comparisons - data.both_unknown_count)[valid]
        metrics[kind]['mean_jaccard_excluding_both_unknown'] = float(np.average(
            data.loc[valid, 'mean_jaccard_excluding_both_unknown'], weights=weights_nonempty)) if weights_nonempty.sum() else None
        metrics[kind]['lowest_batch'] = data.loc[data.mean_jaccard.idxmin(), 'batch_id']
        metrics[kind]['min_batch_mean_jaccard'] = float(data.mean_jaccard.min())
        metrics[kind]['max_batch_mean_jaccard'] = float(data.mean_jaccard.max())
    (out / 'stability_metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotation-root', type=Path, default=ROOT / 'Annotation')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'lloom_analysis')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--stability-only', action='store_true')
    modes.add_argument('--domain-only', action='store_true')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.domain_only:
        plan = json.loads((args.annotation_root / 'plan.json').read_text())
        stability(args.annotation_root, plan, args.output_dir)
    if not args.stability_only:
        scripts = ROOT / 'lloom_analysis/domain_two_topics_only/code'
        destination = args.output_dir / 'domain_two_topics_only'
        for filename in ['visualize_domain_two_topics_only.py', 'visualize_exact_label_combinations.py']:
            subprocess.run([sys.executable, str(scripts / filename),
                '--annotation-root', str(args.annotation_root.resolve()),
                '--output-dir', str(destination.resolve())], check=True)
    print(f'Updated reports: {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
