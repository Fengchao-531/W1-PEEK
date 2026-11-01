"""Draw every exact LLooM label set, including single topics and Unknown."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from visualize_lloom_tsne import ROOT, load_labels


def draw_combined(items, topics, codes, total, out):
    """Render all columns on one shared axis, retaining vector text in PDF/SVG."""
    width = max(20, .43 * len(items) + 5)
    fig, (bars, dots) = plt.subplots(2, 1, figsize=(width, 12), sharex=True,
        gridspec_kw={'height_ratios': [2, 3.6], 'hspace': .08})
    x = np.arange(len(items))
    heights = [n for _, n in items]
    colors = ['#999999' if not ds else '#387eae' if len(ds) == 1 else '#db8a32' for ds, _ in items]
    bars.bar(x, heights, color=colors, width=.72)
    bars.set_yscale('log')
    bars.set_ylim(.72, max(heights) * 2.5)
    for j, (_, n) in enumerate(items):
        bars.text(j, n * 1.10, f'{n:,}\n{n / total:.2%}', ha='center', va='bottom', fontsize=7.5)
    bars.set_ylabel('Emails with this EXACT label set\n(logarithmic count axis)', fontsize=11)
    bars.grid(axis='y', alpha=.18)
    bars.set_axisbelow(True)
    bars.set_title(f'All {len(items)} exact label combinations | {total:,} emails | one continuous panel\n'
                   'Blue = one topic; orange = multiple topics; gray = Unknown', fontsize=18, pad=18)
    for j, (ds, _) in enumerate(items):
        dots.scatter([j] * (len(topics) + 1), range(len(topics) + 1), c='#dddddd', s=18)
        ys = [i for i, name in enumerate(topics) if name in ds] if ds else [len(topics)]
        dots.plot([j] * len(ys), ys, c='#222222', linewidth=1)
        dots.scatter([j] * len(ys), ys, c='#222222', s=29, zorder=3)
    dots.set_yticks(range(len(topics) + 1),
                   labels=[f'{codes[t]}  {t}' for t in topics] + ['Unknown (no topic matched)'])
    dots.tick_params(axis='y', labelsize=10)
    dots.set_ylim(len(topics) + .5, -.5)
    dots.set_xticks(x, labels=[str(j + 1) for j in x])
    dots.tick_params(axis='x', labelsize=7)
    dots.set_xlabel('Exact-combination rank (highest count first)', fontsize=11)
    dots.set_xlim(-.7, len(items) - .3)
    dots.grid(axis='y', alpha=.12)
    fig.subplots_adjust(left=5.3 / width, right=.992, top=.91, bottom=.105)
    fig.text(5.3 / width, .018, 'Each email appears in exactly one column. Black dots specify the COMPLETE label set; gray dots are absent labels.\n'
             'This is an all-email overview (0 to 5 topics); percentages use all emails as denominator. '
             'The count axis is logarithmic.', fontsize=10)
    for suffix in ['png', 'pdf']:
        fig.savefig(out / f'all_exact_combinations_combined.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'lloom_analysis/domain_two_topics_only')
    parser.add_argument('--annotation-root', type=Path, default=ROOT / 'Annotation')
    args = parser.parse_args()
    frame, domains, plan = load_labels(args.annotation_root)
    topics = [d['name'] for d in plan['definitions']]
    codes = {name: f'T{i+1:02}' for i,name in enumerate(topics)}
    counts = Counter(tuple(sorted(ds,key=topics.index)) for ds in domains)
    items = sorted(counts.items(),key=lambda item:(-item[1],len(item[0]),item[0]))
    assert sum(n for _,n in items)==len(frame)
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    rows=[{'rank':i,'topic_codes':' + '.join(codes[name] for name in ds) or 'Unknown',
           'topics_json':json.dumps(ds,ensure_ascii=False),'topic_count':len(ds),
           'emails':n,'percentage_of_all_emails':round(100*n/len(frame),6)}
          for i,(ds,n) in enumerate(items,1)]
    table=pd.DataFrame(rows)
    table.to_csv(out/'all_exact_combinations.csv',index=False,encoding='utf-8-sig')
    table.sort_values(['topic_count','emails','topic_codes'],ascending=[True,False,True]).to_csv(
        out/'exact_combinations_by_topic_count.csv',index=False,encoding='utf-8-sig')
    draw_combined(items,topics,codes,len(frame),out)
    print(f'Updated full-label overview: {len(items)} exact sets, {len(frame)} emails, in {out.resolve()}')


if __name__ == '__main__':
    main()
