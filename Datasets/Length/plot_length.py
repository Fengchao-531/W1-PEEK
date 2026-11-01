"""Plot email-length histograms with overlaid density curves without model calls.

Run from Datasets:
    ../.venv-lloom/bin/python Length/plot_length.py
    ../.venv-lloom/bin/python Length/plot_length.py --unit characters
    ../.venv-lloom/bin/python Length/plot_length.py --dataset phishing --scale log

Words are whitespace-separated items counted from the original text. Characters
are Unicode code points. The optional existing unit uses the supplied Length
column without assuming it represents words, characters, or model tokens.
All rows are included; missing text counts as zero for recomputed lengths.
The default scale is linear. Optional log scale uses log(1 + length), including
zero-length emails; KDE then estimates that transformed variable. The three CSVs
are separate views of the data; full_set is not added to the two splits.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from scipy.stats import gaussian_kde
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
DATASETS = HERE.parent
NAMES = {'full_set': 'Full set', 'benign': 'Benign', 'phishing': 'Phishing'}
COLORS = {'full_set': '#4477AA', 'benign': '#228833', 'phishing': '#CC6677'}
UNITS = {'words': 'Words', 'characters': 'Characters', 'existing': 'Original Length value'}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', choices=('all', *NAMES), default='all')
    parser.add_argument('--unit', choices=tuple(UNITS), default='words')
    parser.add_argument('--scale', choices=('log', 'linear'), default='linear')
    parser.add_argument('--bins', type=int, default=50)
    parser.add_argument('--bandwidth', type=float,
                        help='Optional scipy KDE bandwidth factor; default is Scott\'s rule.')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    if args.bins < 2:
        parser.error('--bins must be at least 2.')
    if args.bandwidth is not None and (not np.isfinite(args.bandwidth) or args.bandwidth <= 0):
        parser.error('--bandwidth must be a positive finite value.')
    return args


def load_lengths(name, unit):
    source = DATASETS / f'{name}.csv'
    df = pd.read_csv(source, dtype={'row_id': str})
    if 'text' not in df:
        raise ValueError(f'{source.name}: missing text column.')
    text = df.text.fillna('').astype(str)
    if unit == 'words':
        lengths = text.map(lambda value: len(value.split())).to_numpy(dtype=float)
    elif unit == 'characters':
        lengths = text.str.len().to_numpy(dtype=float)
    else:
        if 'Length' not in df:
            raise ValueError(f'{source.name}: missing Length column.')
        lengths = pd.to_numeric(df.Length, errors='coerce').to_numpy(dtype=float)
    if not len(lengths) or not np.isfinite(lengths).all() or (lengths < 0).any():
        raise ValueError(f'{source.name}: empty dataset or invalid/negative lengths.')
    records = pd.DataFrame({'dataset': name,
        'row_id': df.row_id if 'row_id' in df else np.arange(len(df)).astype(str),
        'length': lengths})
    for column in ('label', 'Dataset'):
        if column in df:
            records[column] = df[column]
    quantiles = np.percentile(lengths, [5, 25, 50, 75, 95, 99])
    summary = {'dataset': name, 'unit': unit, 'count': len(lengths),
               'empty_texts': int(text.str.strip().eq('').sum()),
               'zero_lengths': int((lengths == 0).sum()),
               'mean': float(lengths.mean()), 'median': float(np.median(lengths)),
               'std': float(lengths.std(ddof=1)) if len(lengths) > 1 else 0.0,
               'min': float(lengths.min()), 'max': float(lengths.max())}
    summary.update({f'p{p}': float(value) for p, value in zip((5, 25, 50, 75, 95, 99), quantiles)})
    return lengths, records, summary


def transform(values, scale):
    return np.log1p(values) if scale == 'log' else values


def format_axis(ax, args, max_length):
    ax.set_xlim(0, max(transform(np.array([max_length]), args.scale)[0], 1e-6))
    if args.scale == 'log':
        ticks = [0.0] + [float(10 ** power) for power in range(12) if 10 ** power <= max_length]
        ax.set_xticks(np.log1p(ticks))
        ax.set_xticklabels([f'{int(value):,}' for value in ticks])
        ax.set_xlabel(f'{UNITS[args.unit]} (log(1 + length) scale)')
    else:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f'{value:,.0f}'))
        ax.set_xlabel(UNITS[args.unit])
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.18)
    ax.set_axisbelow(True)


def save_figure(fig, path):
    fig.savefig(path.with_suffix('.png'), dpi=180, bbox_inches='tight')
    fig.savefig(path.with_suffix('.svg'), bbox_inches='tight')
    plt.close(fig)


def plot_distributions(groups, args, out):
    max_length = max(float(values.max()) for values in groups.values())
    max_value = max(float(transform(np.array([max_length]), args.scale)[0]), 1e-6)
    edges = np.linspace(0, max_value, args.bins + 1)
    grid = np.linspace(0, max_value, 600)
    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4.6),
                             squeeze=False, layout='constrained')
    histograms, densities = [], []
    bin_width = float(edges[1] - edges[0])
    for ax, (name, values) in zip(axes.flat, groups.items()):
        converted = transform(values, args.scale)
        counts, _ = np.histogram(converted, bins=edges)
        histograms.extend({'dataset': name, 'bin_left_transformed': float(left),
                           'bin_right_transformed': float(right),
                           'bin_left_length': float(np.expm1(left) if args.scale == 'log' else left),
                           'bin_right_length': float(np.expm1(right) if args.scale == 'log' else right),
                           'count': int(count)}
                          for left, right, count in zip(edges[:-1], edges[1:], counts))
        ax.hist(converted, bins=edges, color=COLORS[name], alpha=0.4,
                edgecolor='white', linewidth=0.35, label='Histogram')
        ax.set_title(f'{NAMES[name]} (n={len(values):,})')
        ax.set_ylabel('Number of emails')
        if len(values) < 2 or np.ptp(converted) < 1e-12:
            ax.axvline(converted[0], color=COLORS[name], label='Constant length')
        else:
            estimator = gaussian_kde(converted, bw_method=args.bandwidth)
            # Reflection at zero keeps the density on the nonnegative domain.
            density = estimator(grid) + estimator(-grid)
            # Convert density to expected counts per bin to share the histogram axis.
            curve_counts = density * len(values) * bin_width
            ax.plot(grid, curve_counts, color=COLORS[name], linewidth=2,
                    label='KDE (scaled to counts)')
            densities.extend({'dataset': name,
                              'length': float(np.expm1(x) if args.scale == 'log' else x),
                              'x_transformed': float(x), 'density': float(y),
                              'expected_count_per_bin': float(count)}
                             for x, y, count in zip(grid, density, curve_counts))
        format_axis(ax, args, max_length)
        ax.legend(fontsize=8)
    fig.suptitle('Email length distributions: histogram and KDE')
    save_figure(fig, out / 'length_hist')
    pd.DataFrame(histograms).to_csv(out / 'histogram.csv', index=False)
    pd.DataFrame(densities, columns=['dataset', 'length', 'x_transformed', 'density',
                                   'expected_count_per_bin']).to_csv(
        out / 'density.csv', index=False)


def main():
    args = parse_args()
    names = tuple(NAMES) if args.dataset == 'all' else (args.dataset,)
    out = args.output_dir or HERE / 'results' / f'{args.dataset}_{args.unit}_{args.scale}'
    out.mkdir(parents=True, exist_ok=True)
    groups, records, summaries = {}, [], []
    with threadpool_limits(limits=1):
        for name in names:
            lengths, rows, summary = load_lengths(name, args.unit)
            groups[name] = lengths
            records.append(rows)
            summaries.append(summary)
            print(f'{name}: n={len(lengths):,}, mean={summary["mean"]:.1f}, '
                  f'median={summary["median"]:.1f}, max={summary["max"]:,.0f}', flush=True)
        summary = pd.DataFrame(summaries)
        summary.to_csv(out / 'summary.csv', index=False)
        pd.concat(records, ignore_index=True).to_csv(out / 'lengths.csv', index=False)
        plot_distributions(groups, args, out)
    metadata = {'dataset': args.dataset, 'unit': args.unit, 'scale': args.scale,
                'histogram_bins': args.bins,
                'kde_bandwidth': args.bandwidth if args.bandwidth is not None else 'scott',
                'kde_variable': 'log(1 + length)' if args.scale == 'log' else 'length',
                'kde_boundary': 'reflected at zero',
                'kde_plot_scaling': 'density * dataset_count * transformed_bin_width',
                'tail_clipping': False,
                'length_definition': {'words': 'len(text.split())',
                    'characters': 'len(text), Unicode code points',
                    'existing': 'Original Length column; original unit is not inferred'}[args.unit],
                'missing_text': 'Included as zero when recomputing words or characters',
                'group_relationship': 'full_set overlaps its benign/phishing splits; views are not summed'}
    (out / 'run_info.json').write_text(json.dumps(metadata, indent=2))
    print(f'Plots and statistics saved to {out.resolve()}')


if __name__ == '__main__':
    main()
