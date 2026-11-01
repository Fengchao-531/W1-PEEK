"""One consolidated Domain heatmap report for EXACTLY two topic labels.

All four panels share the same exact-pair numerator. Row and global denominators
stay comparable to the older inclusive charts: all emails carrying A, and all
source emails, respectively. Jaccard uses topic sets restricted to the exactly-two
subset, so its union and intersection refer to the same population. Diagonals
are left blank.
"""
import argparse
from collections import Counter
from pathlib import Path
import sys
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from visualize_lloom_tsne import ROOT, load_labels

ZH = ['金融', '购物零售', '物流配送', '邮件云服务', '工作组织', '就业招聘', '商业服务',
      '政府法律', '医疗健康', '保险', '教育培训', '旅行交通', '社交媒体', '科技IT', '个人家庭', '娱乐休闲']


def format_percent(value, decimals):
    if value == 0:
        return '0'
    if value < 10 ** -decimals:
        return f'<{10 ** -decimals:g}'
    return f'{value:.{decimals}f}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'lloom_analysis/domain_two_topics_only')
    parser.add_argument('--annotation-root', type=Path, default=ROOT / 'Annotation')
    args = parser.parse_args()
    frame, domains, plan = load_labels(args.annotation_root)
    names = [x['name'] for x in plan['definitions']]
    codes = [f'T{i+1:02}' for i in range(len(names))]
    positions = {name: i for i, name in enumerate(names)}
    all_membership = np.array([[name in ds for name in names] for ds in domains], dtype=np.int64)
    all_topic_n = all_membership.sum(axis=0)
    two_mask = domains.map(len).eq(2).to_numpy()
    exact_counts = Counter(tuple(sorted(ds, key=positions.get)) for ds in domains[two_mask])
    matrix = np.zeros((len(names), len(names)), dtype=np.int64)
    for (a, b), count in exact_counts.items():
        i, j = positions[a], positions[b]
        matrix[i, j] = matrix[j, i] = count
    n_all, n_two = len(frame), int(two_mask.sum())
    n_two_by_topic = all_membership[two_mask].sum(axis=0)
    row_percent = 100 * matrix / all_topic_n[:, None]
    global_percent = 100 * matrix / n_all
    two_unions = n_two_by_topic[:, None] + n_two_by_topic[None, :] - matrix
    jaccard = np.divide(matrix, two_unions, out=np.zeros_like(matrix, dtype=float), where=two_unions>0)
    assert sum(exact_counts.values()) == n_two
    assert np.tril(matrix, -1).sum() == n_two
    assert np.array_equal(matrix, matrix.T)
    assert not np.diag(matrix).any()
    assert np.array_equal(matrix.sum(axis=1), n_two_by_topic)
    np.testing.assert_allclose(row_percent.sum(axis=1), 100*n_two_by_topic/all_topic_n)
    for i in range(len(names)):
        for j in range(i+1,len(names)):
            direct_union = np.count_nonzero((all_membership[:,i] | all_membership[:,j]) & two_mask)
            assert direct_union == two_unions[i,j]
    def pair_record(i, j):
        count = int(matrix[i,j])
        return {'行Topic': codes[i], '行Topic名称': ZH[i],
                '列Topic': codes[j], '列Topic名称': ZH[j],
                '完整主题组合': ' + '.join(sorted([codes[i],codes[j]])),
                '恰好两主题邮件数': count,
                '行Topic全部邮件数': int(all_topic_n[i]), '列Topic全部邮件数': int(all_topic_n[j]),
                '行内百分比': round(float(row_percent[i,j]),4),
                '反方向行内百分比': round(float(row_percent[j,i]),4),
                '占全体邮件百分比': round(float(global_percent[i,j]),4),
                '全体邮件分母': n_all,
                '行Topic在双主题子集中的邮件数': int(n_two_by_topic[i]),
                '列Topic在双主题子集中的邮件数': int(n_two_by_topic[j]),
                'Jaccard并集邮件数_限双主题子集': int(two_unions[i,j]),
                'Jaccard_限双主题子集': round(float(jaccard[i,j]),6)}

    rows = []
    for (a, b), count in sorted(exact_counts.items(), key=lambda x: (-x[1], x[0])):
        i, j = positions[a], positions[b]
        rows.append({'排名': len(rows) + 1, '完整主题组合': f'{codes[i]} + {codes[j]}',
                     '组合中文': f'{ZH[i]}＋{ZH[j]}', '恰好两主题邮件数': count,
                     'A主题全部邮件数': int(all_topic_n[i]), 'B主题全部邮件数': int(all_topic_n[j]),
                     '占A主题全部邮件百分比': round(float(row_percent[i,j]),4),
                     '占B主题全部邮件百分比': round(float(row_percent[j,i]),4),
                     '占全体邮件百分比': round(float(global_percent[i,j]),4),
                     '全体邮件分母': n_all,
                     'A主题在恰好双主题子集中的邮件数': int(n_two_by_topic[i]),
                     'B主题在恰好双主题子集中的邮件数': int(n_two_by_topic[j]),
                     'Jaccard并集邮件数_限双主题子集': int(two_unions[i,j]),
                     'Jaccard_限双主题子集': round(float(jaccard[i,j]),6)})
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'exact_two_topics_pairs.csv', index=False, encoding='utf-8-sig')
    all_pairs = [pair_record(i,j) for i in range(len(names)) for j in range(i+1,len(names))]
    by_row = [pair_record(i,j) for i in range(len(names)) for j in range(len(names)) if i!=j]
    assert len(all_pairs)==120 and len(by_row)==240
    assert sum(row['恰好两主题邮件数'] for row in all_pairs)==n_two
    pd.DataFrame(all_pairs).to_csv(out/'exact_two_topics_all_pairs.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(by_row).to_csv(out/'exact_two_topics_by_row.csv',index=False,encoding='utf-8-sig')
    single_rows=[]
    for i,name in enumerate(names):
        sizes = Counter(len(ds) for ds in domains if name in ds)
        single_rows.append({'Topic':codes[i],'主题':ZH[i], '含该主题总邮件数':int(all_topic_n[i]),
            '恰好单主题邮件数':sizes[1], '恰好双主题邮件数':sizes[2],
            '三个及以上主题邮件数':sum(n for k,n in sizes.items() if k>=3),
            '单主题占该主题百分比':round(100*sizes[1]/all_topic_n[i],4)})
    assert sum(r['恰好单主题邮件数'] for r in single_rows)==sum(len(ds)==1 for ds in domains)
    pd.DataFrame(single_rows).to_csv(out/'single_topic_counts.csv',index=False,encoding='utf-8-sig')

    fig, axes = plt.subplots(2, 2, figsize=(24, 24))
    diagonal = np.eye(len(names), dtype=bool)
    upper = np.triu(np.ones_like(matrix, dtype=bool))
    for index, ax in enumerate(axes.flat):
        if index == 0:
            values, mask = matrix, upper
            cmap = plt.get_cmap('Blues').copy(); cmap.set_bad('#f4f4f4')
            image = ax.imshow(np.ma.masked_where(mask | (values == 0), values),
                              cmap=cmap, norm=LogNorm(vmin=1, vmax=max(2,int(matrix.max()))))
            title = '1. EXACTLY two topics: email counts\nNo third, fourth or fifth topic is allowed'
            cbar_label = 'Exact-pair email count (log color scale)'
        elif index == 1:
            values, mask = jaccard, upper
            cmap = plt.get_cmap('YlOrRd').copy(); cmap.set_bad('#eeeeee')
            vmax = max(.01,float(jaccard[~diagonal].max()))
            image = ax.imshow(np.ma.masked_where(mask, values),cmap=cmap,vmin=0,vmax=vmax)
            title = '2. Jaccard overlap — EXACTLY-TWO subset only\nExact A+B / (n2(A) + n2(B) - Exact A+B)'
            cbar_label = 'Jaccard (0 to 1)'
        else:
            values, mask = (row_percent if index == 2 else global_percent), diagonal
            cmap = plt.get_cmap('YlOrRd').copy(); cmap.set_bad('#eeeeee')
            vmax = 100 if index == 2 else max(.01,float(global_percent.max()))
            image = ax.imshow(np.ma.masked_where(mask, values), cmap=cmap, vmin=0, vmax=vmax)
            title = ('3. Row-normalized exact-pair share (%)\nExact A+B / ALL emails carrying A'
                     if index == 2 else f'4. Global exact-pair share (%)\nExact A+B / ALL {n_all:,} emails')
            cbar_label = 'Percentage (%)'
        for i in range(len(names)):
            for j in range(len(names)):
                if mask[i,j]:
                    if i == j:
                        ax.text(j,i,'—',ha='center',va='center',fontsize=8,color='#777777')
                    continue
                value = values[i,j]
                label = (str(int(value)) if index == 0 else f'{value:.3f}' if index == 1
                         else format_percent(float(value),1 if index == 2 else 2))
                if index == 0:
                    dark = value > matrix.max() ** .65
                else:
                    dark = value > .6*vmax
                ax.text(j,i,label,ha='center',va='center',fontsize=8,color='white' if dark else '#222222')
        ax.set(xticks=range(len(names)), xticklabels=codes,
               yticks=range(len(names)), yticklabels=(
                   [f'{code} (n={num:,})' for code,num in zip(codes,all_topic_n)] if index==2 else
                   [f'{code} (n2={num:,})' for code,num in zip(codes,n_two_by_topic)] if index==1 else codes),
               xlabel='Column topic B', ylabel='Row topic A', title=title)
        ax.tick_params(axis='x',labelrotation=90)
        ax.set_xticks(np.arange(-.5,len(names)),minor=True)
        ax.set_yticks(np.arange(-.5,len(names)),minor=True)
        ax.grid(which='minor',color='white',linewidth=.35)
        ax.tick_params(which='minor',bottom=False,left=False)
        fig.colorbar(image,ax=ax,shrink=.65,pad=.025,label=cbar_label)
    fig.suptitle(f'DOMAIN — EXACTLY TWO TOPICS ONLY\n{n_two:,} emails in {len(exact_counts)} disjoint pairs | all-data denominator N={n_all:,}',fontsize=20,y=.98)
    legend='    '.join(f'{code}: {name}' for code,name in zip(codes,names))
    fig.text(.035,.015,'\n'.join(textwrap.wrap(legend,width=175)),fontsize=10,va='bottom')
    fig.text(.035,.075,'All numerators are EXACT A+B counts. Jaccard uses only the exactly-two-topic subset for BOTH intersection and union (n2).\n'
             'Row percentage uses all emails carrying A (n); global percentage uses all 13,799 emails, including Unknown.\n'
             'Diagonal is blank. Count/Jaccard panels show each pair once; percentage panels show both directions. Independent color scales.',fontsize=11)
    fig.subplots_adjust(left=.065,right=.98,bottom=.15,top=.92,wspace=.33,hspace=.33)
    for suffix in ['png','pdf']:
        fig.savefig(out/f'exact_two_topics_summary.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    description = f'''本文件夹统一采用“恰好只有两个 Domain 标签”的口径。

本目录是 Domain 图表及统计的统一入口。
1. exact_two_topics_summary.pdf：最新版恰好双主题四格热力图；PNG是同图图片版。
2. all_exact_combinations_combined.pdf：全体邮件的107个完整标签组总览；PNG是同图图片版。
   这张总览保留0、1、2、3、4、5主题，标题和数据范围与双主题热力图区分。

exact_two_topics_pairs.csv 包含全部 {len(exact_counts)} 个实际存在的完整双主题组合，按邮件数降序排列，不重复计算 A+B 与 B+A。
exact_two_topics_all_pairs.csv 列出全部120对，包括零计数，采用同一双主题口径。
exact_two_topics_by_row.csv 按T01到T16逐行列全240项，替代旧overlap表；A到B与B到A的分子相同，行内分母不同。
single_topic_counts.csv 包含16个主题的恰好单主题数量，并列出双主题、三主题以上及总数。
all_exact_combinations.csv 与完整标签总览的序号一一对应。
exact_combinations_by_topic_count.csv 是相同完整标签数据按主题个数分组的视图。

数据范围：主标注 {n_all:,} 封邮件。恰好两个主题 {n_two:,} 封。排除单主题、三个及以上主题、Unknown 的邮件作为双主题计数的分子。

总图为2×2布局，按先上后下、从左到右阅读：
1. 数量：完整标签恰好是 {{A,B}} 的邮件数。
2. Jaccard：先只保留恰好两个标签的{n_two:,}封邮件，再取含A、含B的集合，计算交集/并集。
   n2(A)表示双主题子集中包含A的邮件数，n2(B)同理。
   J = 恰好{{A,B}}的邮件数 / [n2(A)+n2(B)-恰好{{A,B}}的邮件数]。
   例如金融＋Government：191/(870+215-191)=0.213647。
   不能把双主题交集直接与旧图全数据集合的并集混用，那样不是同一集合口径的Jaccard。
3. 行内归一化：恰好 {{A,B}} 的邮件数 / 全部包含 A 的邮件数 ×100%。
   行旁 n 仍是全数据中全部包含 A 的邮件数，保留之前图的分母。
   例如 Government 的 n=498，金融＋Government 恰好双标签191封，行内比例191/498=38.35%。
   行内各格相加表示该主题中“恰好双主题邮件”的比例，因此不要求加起来100%。
4. 全局归一化：恰好 {{A,B}} 的邮件数 / 全部 {n_all:,} 封邮件 ×100%。
   例如191/13799=1.3842%。分母包含Unknown，沿用之前全局图口径。

没有把归一化分母悄悄改为双主题子集；本图不表示191/215或191/1222。
对角线为空，不是100%，也不表示仅单主题数量。
单主题与三主题以上的邮件不计入双主题分子；行内及全局分母仍按上述定义保留。
旧图包含三个及以上标签的邮件，数值不同。后续要和“恰好2个主题”表核对，请使用本文件夹。

原始数据：{args.annotation_root.resolve()}
源文件 SHA256：{plan['source_sha256']}
绘图代码：本目录的code/。
重新生成本目录两张总图和全部CSV：bash {(Path(__file__).resolve().parent/'run.sh')}
Domain stability文件保留在上一级lloom_analysis目录。
'''
    (out/'说明.txt').write_text(description,encoding='utf-8')
    print(f'Validated: {n_all} total emails; {n_two} exactly-two-topic emails; {len(exact_counts)} unique pairs.')
    print('金融＋商业:',matrix[0,6],'金融＋Government:',matrix[0,7])
    print('Government row:',row_percent[7,0],'global:',global_percent[7,0])
    print('Jaccard financial/business:',jaccard[0,6],'financial/government:',jaccard[0,7])
    print(out.resolve())


if __name__ == '__main__':
    main()
