"""Project all completed LLooM main annotations; labels are only used for coloring."""
import argparse
import ast
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits

from visualize_tsne import text_features

ROOT = Path(__file__).resolve().parent


def load_labels(root):
    plan = json.loads((root / 'plan.json').read_text())
    frames = []
    for batch in sorted(root.glob('b[0-9][0-9][0-9]')):
        folder = batch / '02_labels'
        status = json.loads((folder / 'status.json').read_text())
        if status['state'] != 'complete':
            raise ValueError(f'Incomplete annotation: {folder}')
        frame = pd.read_csv(folder / 'labels.csv', dtype={'row_id': str})
        if len(frame) != status['emails']:
            raise ValueError(f'Row count mismatch: {folder}')
        frame['batch_id'] = batch.name
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    if len(frames) != plan['batch_count'] or len(frame) != plan['eligible_emails'] or frame.row_id.duplicated().any():
        raise ValueError('Main annotations must cover every eligible email exactly once.')
    domains = frame.domains.map(ast.literal_eval)
    if not all(isinstance(ds, (tuple, list)) and all(isinstance(d, str) for d in ds) for ds in domains):
        raise ValueError('Invalid domain labels.')
    if not np.array_equal(domains.map(len).to_numpy(), frame.domain_count.to_numpy()):
        raise ValueError('Domain count mismatch.')
    if not np.array_equal(domains.map(len).eq(0).to_numpy(), frame.status.eq('Unknown').to_numpy()):
        raise ValueError('Unknown status mismatch.')
    if frame.text.isna().any() or frame.text.str.strip().eq('').any():
        raise ValueError('Empty email text.')
    return frame, domains, plan


def save(fig, out, name):
    fig.savefig(out / f'{name}.png', dpi=200, bbox_inches='tight')
    fig.savefig(out / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotation-root', type=Path, default=ROOT / 'Annotation')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'tsne_results/lloom_all')
    parser.add_argument('--perplexity', type=float, default=30)
    parser.add_argument('--max-iter', type=int, default=1500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--svd-components', type=int, default=50)
    parser.add_argument('--max-features', type=int, default=20000)
    args = parser.parse_args()
    frame, domains, plan = load_labels(args.annotation_root)
    if not 0 < args.perplexity < len(frame) or args.max_iter <= 250 or args.threads < 1 or min(args.svd_components, args.max_features) < 2:
        parser.error('Invalid perplexity, iterations, thread count, or feature dimensions.')
    args.sample_seed = args.seed
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    print(f'Loaded {len(frame):,} emails from {frame.batch_id.nunique()} batches.', flush=True)
    with threadpool_limits(limits=args.threads):
        _, reduced, feature_info = text_features(frame, args)
        print(f'Features ready: {reduced.shape}; running t-SNE.', flush=True)
        model = TSNE(n_components=2, perplexity=args.perplexity, learning_rate='auto',
                     init='pca', random_state=args.seed, max_iter=args.max_iter,
                     method='barnes_hut', n_jobs=args.threads, verbose=1)
        xy = model.fit_transform(reduced)
    topics = [d['name'] for d in plan['definitions']]
    colors = {name: plt.get_cmap('tab20')(i) for i, name in enumerate(topics)}
    colors.update({'Multiple topics': '#222222', 'Unknown / Outlier': '#aaaaaa'})
    groups = domains.map(lambda ds: ds[0] if len(ds) == 1 else 'Multiple topics' if ds else 'Unknown / Outlier')
    coordinates = frame.drop(columns=['reasons_json'], errors='ignore').copy()
    coordinates['plot_group'] = groups
    coordinates['tsne_1'], coordinates['tsne_2'] = xy.T
    coordinates.to_csv(out / 'coordinates.csv', index=False)
    fig, ax = plt.subplots(figsize=(15, 9))
    for name, color in colors.items():
        mask = groups.eq(name).to_numpy()
        ax.scatter(*xy[mask].T, s=5, alpha=0.55, c=[color], rasterized=True, linewidths=0)
    handles = [Line2D([], [], marker='o', linestyle='', color=c,
                     label=f'{name} (n={int(groups.eq(name).sum()):,})') for name, c in colors.items()]
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1, .5), fontsize=8)
    ax.set(title=f'LLooM topics: all {len(frame):,} emails\nTF-IDF + SVD + t-SNE | perplexity={args.perplexity:g}, seed={args.seed}',
           xlabel='t-SNE 1', ylabel='t-SNE 2', xticks=[], yticks=[])
    save(fig, out, 'tsne_topics_overview')
    fig, axes = plt.subplots(5, 4, figsize=(20, 22))
    counts = []
    for ax, name in zip(axes.flat, topics + ['Unknown / Outlier']):
        mask = domains.map(lambda ds: name in ds if name != 'Unknown / Outlier' else not ds).to_numpy()
        ax.scatter(*xy.T, s=2, c='#dddddd', alpha=.3, rasterized=True, linewidths=0)
        ax.scatter(*xy[mask].T, s=5, c=[colors[name]], alpha=.75, rasterized=True, linewidths=0)
        count = int(mask.sum())
        ax.set(title=f'{name}\nn={count:,} ({count / len(frame):.2%})', xticks=[], yticks=[])
        counts.append({'topic': name, 'count': count, 'percentage': count / len(frame) * 100})
    for ax in list(axes.flat)[len(topics) + 1:]:
        ax.set_visible(False)
    fig.suptitle('Topic membership on the SAME t-SNE coordinates\nMulti-topic emails appear in every matching panel', fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, .96))
    save(fig, out, 'tsne_topic_panels')
    pd.DataFrame(counts).to_csv(out / 'topic_counts.csv', index=False)
    metadata = {'parameters': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'emails': len(frame), 'features': feature_info, 'kl_divergence': float(model.kl_divergence_),
                'note': 'Text features only; labels used for coloring. Main annotations only; stability repeats excluded.'}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(f'Saved plots and coordinates to {out.resolve()}', flush=True)


if __name__ == '__main__':
    main()
