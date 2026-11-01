"""Visualize sampled emails using TF-IDF -> normalized SVD -> two-dimensional t-SNE.

Examples (run from this directory):
    python visualize_tsne.py --dataset phishing --n-samples 100 --sweep
    python visualize_tsne.py --dataset benign --n-samples 100
    python visualize_tsne.py --dataset full_set --n-samples 100 --color-by label

Each point represents one complete email. This lexical baseline measures wording
similarity, not predefined domains or semantic embedding similarity. Labels and
source datasets are used only for coloring, never as input features.
"""
import argparse
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', choices=('full_set', 'benign', 'phishing'), default='phishing')
    parser.add_argument('--n-samples', type=int, default=100)
    parser.add_argument('--sample-seed', type=int, default=42)
    parser.add_argument('--perplexity', type=float, default=15,
                        help='Local-neighborhood scale; start at 15 for this 100-email pilot.')
    parser.add_argument('--tsne-seed', type=int, default=42)
    parser.add_argument('--learning-rate', default='auto', help="'auto' or a positive number.")
    parser.add_argument('--max-iter', type=int, default=1500)
    parser.add_argument('--early-exaggeration', type=float, default=12)
    parser.add_argument('--init', choices=('pca', 'random'), default='pca')
    parser.add_argument('--svd-components', type=int, default=50)
    parser.add_argument('--max-features', type=int, default=20000)
    parser.add_argument('--color-by', choices=('auto', 'label', 'Dataset'), default='auto')
    parser.add_argument('--threads', type=int, default=1,
                        help='One thread avoids parallel overhead for the small pilot sample.')
    parser.add_argument('--sweep', action='store_true', help='Compare parameters on the SAME sample.')
    parser.add_argument('--perplexities', nargs='+', type=float, default=[5, 15, 30])
    parser.add_argument('--tsne-seeds', nargs='+', type=int, default=[42, 7, 21])
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    if args.n_samples < 4:
        parser.error('--n-samples must be at least 4.')
    if not 0 < args.perplexity < args.n_samples:
        parser.error('--perplexity must be positive and less than --n-samples.')
    if args.sweep and any(not 0 < p < args.n_samples for p in args.perplexities):
        parser.error('Every sweep perplexity must be positive and less than --n-samples.')
    if args.max_iter <= 250:
        parser.error('--max-iter must exceed 250 to include optimization after early exaggeration.')
    if args.svd_components < 2 or args.max_features < 2 or args.threads < 1:
        parser.error('Use at least 2 SVD components/features and at least 1 thread.')
    if args.early_exaggeration < 1:
        parser.error('--early-exaggeration must be at least 1.')
    if args.learning_rate != 'auto':
        try:
            args.learning_rate = float(args.learning_rate)
        except ValueError:
            parser.error('--learning-rate must be auto or a positive number.')
        if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
            parser.error('--learning-rate must be auto or a positive finite number.')
    return args


def load_sample(args):
    source = ROOT / f'{args.dataset}.csv'
    df = pd.read_csv(source)
    if 'text' not in df:
        raise ValueError(f'{source.name} must contain a text column.')
    # Preserve the original row identifier across the full and split datasets.
    if 'row_id' not in df:
        df.insert(0, 'row_id', np.arange(len(df)))
    valid = df.text.notna() & df.text.fillna('').astype(str).str.strip().ne('')
    available = df.loc[valid].copy()
    if len(available) < args.n_samples:
        raise ValueError(f'Only {len(available)} nonempty emails; requested {args.n_samples}.')
    sample = available.sample(n=args.n_samples, random_state=args.sample_seed).reset_index(drop=True)
    sample['text'] = sample.text.astype(str)
    return sample, {'path': str(source), 'total_rows': len(df),
                    'nonempty_rows': len(available), 'empty_rows_removed': int((~valid).sum())}


def text_features(sample, args):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2),
                                 sublinear_tf=True, max_features=args.max_features,
                                 strip_accents='unicode', dtype=np.float64)
    original = vectorizer.fit_transform(sample.text)
    if np.any(np.asarray(original.getnnz(axis=1)) == 0):
        raise ValueError('At least one sampled email has no usable word features; inspect sample.csv.')
    dimensions = min(args.svd_components, original.shape[0] - 1, original.shape[1] - 1)
    if dimensions < 2:
        raise ValueError('The sample needs at least three distinct usable word features.')
    svd = TruncatedSVD(n_components=dimensions, n_iter=7, random_state=args.sample_seed)
    reduced = normalize(svd.fit_transform(original))
    return original, reduced, {'representation': 'word TF-IDF + L2-normalized TruncatedSVD',
        'ngram_range': [1, 2], 'stop_words': 'english', 'sublinear_tf': True,
        'vocabulary_size': original.shape[1], 'svd_components': dimensions,
        'svd_explained_variance_ratio_sum': float(svd.explained_variance_ratio_.sum()),
        'exact_duplicate_texts_in_sample': int(sample.text.duplicated().sum())}


def neighbors(distances, k):
    distances = distances.copy()
    np.fill_diagonal(distances, np.inf)
    return np.argsort(distances, axis=1)[:, :k]


def project(original, reduced, args, perplexity, seed, reference_neighbors):
    model = TSNE(n_components=2, perplexity=perplexity, learning_rate=args.learning_rate,
                 max_iter=args.max_iter, early_exaggeration=args.early_exaggeration,
                 init=args.init, random_state=seed, metric='euclidean',
                 method='barnes_hut', angle=0.5, n_jobs=args.threads)
    xy = model.fit_transform(reduced)
    k = reference_neighbors.shape[1]
    projected_neighbors = neighbors(pairwise_distances(xy), k)
    overlap = np.mean([len(set(a) & set(b)) / k
                       for a, b in zip(reference_neighbors, projected_neighbors)])
    quality = {'perplexity': float(perplexity), 'tsne_seed': seed,
        'learning_rate_effective': float(model.learning_rate_),
        'iterations_completed': int(model.n_iter_) + 1,
        'kl_divergence': float(model.kl_divergence_),
        'trustworthiness_original_tfidf': float(trustworthiness(
            original, xy, n_neighbors=k, metric='cosine')),
        'trustworthiness_svd_input': float(trustworthiness(reduced, xy, n_neighbors=k)),
        'neighbor_overlap_original_tfidf': float(overlap), 'evaluation_neighbors': k}
    print(f'perplexity={perplexity:g}, seed={seed}: '
          f'trustworthiness(TF-IDF)={quality["trustworthiness_original_tfidf"]:.3f}, '
          f'neighbor overlap={overlap:.3f}', flush=True)
    return xy, quality


def draw(ax, xy, sample, color_by, colors):
    values = sample[color_by].fillna('Unknown').astype(str)
    for name, color in colors.items():
        selected = values.eq(name).to_numpy()
        ax.scatter(xy[selected, 0], xy[selected, 1], s=35, alpha=0.8,
                   color=color, edgecolors='white', linewidths=0.4,
                   label=f'{name} (n={int(selected.sum())})')
    ax.set(xlabel='t-SNE 1', ylabel='t-SNE 2')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[['top', 'right']].set_visible(False)


def save_figure(fig, path):
    fig.savefig(path.with_suffix('.png'), dpi=180, bbox_inches='tight')
    fig.savefig(path.with_suffix('.svg'), bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()
    sample, source_info = load_sample(args)
    out = args.output_dir or ROOT / 'tsne_results' / f'{args.dataset}_{args.n_samples}_seed{args.sample_seed}'
    out.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out / 'sample.csv', index=False)
    color_by = args.color_by
    if color_by == 'auto':
        color_by = 'label' if sample.get('label', pd.Series(dtype=str)).nunique() > 1 else 'Dataset'
    if color_by not in sample:
        raise ValueError(f'The dataset has no {color_by!r} column for coloring.')
    categories = sorted(sample[color_by].fillna('Unknown').astype(str).unique())
    palette = plt.get_cmap('tab20' if len(categories) > 10 else 'tab10')
    colors = {name: palette(i % palette.N) for i, name in enumerate(categories)}
    print(f'Sampled {len(sample)} of {source_info["nonempty_rows"]} nonempty emails; '
          f'coloring by {color_by}. Only sampled text is vectorized.', flush=True)

    with threadpool_limits(limits=args.threads):
        original, reduced, feature_info = text_features(sample, args)
        k = min(5, (len(sample) - 1) // 2)
        reference = neighbors(pairwise_distances(original, metric='cosine'), k)
        runs = {}

        def get_run(perplexity, seed):
            key = (float(perplexity), seed)
            if key not in runs:
                runs[key] = project(original, reduced, args, perplexity, seed, reference)
            return runs[key]

        xy, primary_quality = get_run(args.perplexity, args.tsne_seed)
        coordinates = sample.copy()
        coordinates['tsne_1'], coordinates['tsne_2'] = xy.T
        coordinates.to_csv(out / 'coordinates.csv', index=False)
        fig, ax = plt.subplots(figsize=(9, 7), layout='constrained')
        draw(ax, xy, sample, color_by, colors)
        ax.set_title(f'{args.dataset}: {len(sample)} emails | TF-IDF + SVD + t-SNE\n'
                     f'perplexity={args.perplexity:g}, seed={args.tsne_seed}; colored by {color_by}')
        ax.legend(title=color_by, fontsize=8)
        save_figure(fig, out / 'tsne')

        if args.sweep:
            fig, axes = plt.subplots(len(args.tsne_seeds), len(args.perplexities),
                figsize=(5 * len(args.perplexities), 4 * len(args.tsne_seeds)), squeeze=False)
            sweep_coordinates = []
            for row, seed in enumerate(args.tsne_seeds):
                for column, perplexity in enumerate(args.perplexities):
                    current, quality = get_run(perplexity, seed)
                    ax = axes[row, column]
                    draw(ax, current, sample, color_by, colors)
                    ax.set_title(f'p={perplexity:g}, seed={seed}\n'
                                 f'T={quality["trustworthiness_original_tfidf"]:.3f}, '
                                 f'overlap={quality["neighbor_overlap_original_tfidf"]:.3f}')
                    sweep_coordinates.append(pd.DataFrame({'row_id': sample.row_id,
                        'perplexity': perplexity, 'tsne_seed': seed,
                        'tsne_1': current[:, 0], 'tsne_2': current[:, 1]}))
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.97),
                       ncol=min(len(labels), 3), fontsize=8, frameon=False)
            fig.suptitle(f'{args.dataset}: SAME {len(sample)} emails and word features in every panel', y=0.995)
            fig.tight_layout(rect=(0, 0, 1, 0.92))
            save_figure(fig, out / 'parameter_comparison')
            pd.concat(sweep_coordinates, ignore_index=True).to_csv(out / 'sweep_coordinates.csv', index=False)

    pd.DataFrame([quality for _, quality in runs.values()]).to_csv(out / 'quality_metrics.csv', index=False)
    metadata = {'parameters': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'source': source_info, 'features': feature_info, 'color_by': color_by,
        'sample_label_counts': sample.label.value_counts().to_dict() if 'label' in sample else {},
        'primary_quality': primary_quality,
        'tsne_fixed_parameters': {'n_components': 2, 'metric': 'euclidean',
                                  'method': 'barnes_hut', 'angle': 0.5},
        'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__,
                     'numpy': np.__version__, 'pandas': pd.__version__},
        'interpretation': 'Lexical similarity only. Colors do not train the projection. '
            'Cluster sizes and distances between apparent groups are not reliable global distances. '
            'Trustworthiness evaluates local neighborhoods, not domain-label correctness. '
            'Do not compare KL divergence across different perplexities to select a winner.'}
    (out / 'run_info.json').write_text(json.dumps(metadata, indent=2))
    print(f'Results saved to {out.resolve()}')


if __name__ == '__main__':
    main()
