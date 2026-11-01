"""Estimate story/template diversity in the 17 candidate subsets, locally.

Counts are threshold-dependent semantic/lexical clusters, not verified taxonomy.
No topic labels are used as clustering features. Original annotations are unchanged.
"""
import argparse
import ast
import collections
import csv
import html
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from threadpoolctl import threadpool_limits

from visualize_lloom_tsne import ROOT, load_labels

MODEL = 'sentence-transformers/all-MiniLM-L6-v2'


def normalize(text):
    text = html.unescape(unicodedata.normalize('NFKC', text))
    text = re.sub(r'<[^>\n]{1,300}>', ' ', text)
    # Some corpus messages have a whole body on a line starting "From:".
    # Only remove short, standalone envelope lines.
    text = re.sub(r'(?im)^(?:from|to|cc|bcc|reply-to|date|sent|message-id):[^\n]{0,160}$', ' ', text)
    text = re.sub(r'(?i)\b(?:https?://|www\.)\S+', ' URL ', text)
    text = re.sub(r'(?i)\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b', ' EMAIL ', text)
    # Mask titled names only; institutional/service names remain available.
    text = re.sub(r'\b(?:Mr|Mrs|Ms|Miss|Dr|Prof|Barrister)\.?\s+[A-Z][A-Za-z\x27-]*(?:\s+[A-Z][a-z][A-Za-z\x27-]*){0,2}', ' PERSON ', text)
    text = re.sub(r'(?i)(?:[$€£¥]\s*|\b(?:usd|gbp|eur)\s*)\d[\d, .]*(?:million|billion|thousand)?', ' AMOUNT ', text)
    text = re.sub(r'\d+(?:[.,:/-]\d+)*', ' NUMBER ', text)
    return re.sub(r'\s+', ' ', text.casefold()).strip()


def partition(similarity, threshold):
    distances = np.clip(1.0 - similarity, 0, 2)
    distances = (distances + distances.T) / 2
    np.fill_diagonal(distances, 0)
    tree = linkage(squareform(distances, checks=False), method='complete')
    return fcluster(tree, t=1.0 - threshold, criterion='distance')


def partition_stats(labels):
    sizes = collections.Counter(labels)
    return {'clusters': len(sizes), 'singleton_clusters': sum(n == 1 for n in sizes.values()),
            'multi_email_clusters': sum(n > 1 for n in sizes.values()),
            'emails_in_multi_email_clusters': sum(n for n in sizes.values() if n > 1),
            'largest_cluster': max(sizes.values())}


def write_unique_tables(selected, candidates, definitions, threshold, out):
    domains = selected.domains.map(ast.literal_eval)
    rows = []
    for i, definition in enumerate(definitions, 1):
        group = selected.loc[domains.map(lambda ds: definition['name'] in ds)]
        rows.append({'topic_code': f'T{i:02}', 'topic': definition['name'],
                     'selected_emails': len(group), 'unique_story_clusters': group.story_cluster_id.nunique(),
                     'semantic_threshold': threshold, 'scope': '1279_emails_in_17_candidate_subsets'})
    pd.DataFrame(rows).to_csv(out / 'unique_stories_by_topic.csv', index=False, encoding='utf-8-sig')
    subsets = selected.groupby('subset_id').agg(selected_emails=('row_id', 'size'),
        unique_story_clusters=('story_cluster_id', 'nunique')).reset_index()
    subsets = candidates[['subset_id', 'topic_codes', 'topics_json']].merge(
        subsets, on='subset_id', validate='one_to_one')
    subsets['semantic_threshold'] = threshold
    subsets.to_csv(out / 'unique_stories_by_subset.csv', index=False, encoding='utf-8-sig')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'lloom_story_diversity')
    parser.add_argument('--threshold', type=float, default=.80)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if not 0 < args.threshold < 1 or args.threads < 1:
        parser.error('Threshold must be between 0 and 1; threads must be positive.')
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    frame, domains, plan = load_labels(ROOT / 'Annotation')
    candidates = pd.read_csv(ROOT / 'lloom_subset_candidates.csv', encoding='utf-8-sig')
    mapping = {tuple(sorted(json.loads(r.topics_json))): r.subset_id for r in candidates.itertuples()}
    keys = domains.map(lambda ds: tuple(sorted(ds)))
    selected = frame.loc[keys.map(lambda ds: ds in mapping)].copy().reset_index(drop=True)
    selected['subset_id'] = [mapping[tuple(sorted(ast.literal_eval(ds)))] for ds in selected.domains]
    # Keep source order fixed so linkage tie-breaking and cluster IDs are reproducible.
    selected = selected.sort_values('row_id', key=lambda s: s.astype(int)).reset_index(drop=True)
    normalized = selected.text.map(normalize)
    if normalized.str.strip().eq('').any():
        raise ValueError('Normalization erased a complete email; inspect the cleaning rules.')
    selected['normalized_text'] = normalized
    print(f'Loaded {len(selected)} selected emails; computing normalized lexical similarity.', flush=True)
    with threadpool_limits(limits=args.threads):
        features = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words='english',
                                  strip_accents='unicode', max_features=50000).fit_transform(normalized)
        lexical = cosine_similarity(features).astype(np.float32)
    np.save(out / 'lexical_similarity.npy', lexical)

    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(args.threads)
    print('Loading local semantic encoder (downloads public model weights on first run).', flush=True)
    model = SentenceTransformer(MODEL, device='cpu')
    # Process complete emails in token windows; do not truncate at the model limit.
    chunk_tokens = model.max_seq_length - 2
    overlap = 32
    owners, chunks, weights = [], [], []
    for i, text in enumerate(normalized):
        ids = model.tokenizer.encode(text, add_special_tokens=False, truncation=False)
        for start in range(0, len(ids), chunk_tokens - overlap):
            part = ids[start:start + chunk_tokens]
            owners.append(i)
            chunks.append(model.tokenizer.decode(part, skip_special_tokens=True))
            weights.append(len(part) if start == 0 else max(1, len(part) - overlap))
            if start + chunk_tokens >= len(ids):
                break
    cache = out / 'semantic_embeddings.npz'
    if cache.exists():
        saved = np.load(cache)
        if np.array_equal(saved['row_ids'].astype(str), selected.row_id.to_numpy(dtype=str)):
            embeddings = saved['embeddings'].copy()
            changed = np.flatnonzero(saved['normalized_text'].astype(str) != normalized.to_numpy(dtype=str))
            print(f'Reusing cached embeddings; updating {len(changed)} changed emails.', flush=True)
            if len(changed):
                positions = [j for j, owner in enumerate(owners) if owner in set(changed)]
                vectors = model.encode([chunks[j] for j in positions], batch_size=32,
                                       show_progress_bar=True, normalize_embeddings=True)
                embeddings[changed] = 0
                for j, vector in zip(positions, vectors):
                    embeddings[owners[j]] += vector * weights[j]
                embeddings[changed] /= np.linalg.norm(embeddings[changed], axis=1, keepdims=True).clip(min=1e-12)
                np.savez_compressed(cache, embeddings=embeddings, row_ids=selected.row_id.to_numpy(dtype=str),
                                    normalized_text=normalized.to_numpy(dtype=str))
        else:
            raise ValueError('Embedding cache has different row IDs; use a fresh output directory.')
    else:
        print(f'Encoding {len(chunks)} chunks from {len(selected)} complete emails.', flush=True)
        vectors = model.encode(chunks, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
        embeddings = np.zeros((len(selected), vectors.shape[1]), dtype=np.float32)
        for i, vector, weight in zip(owners, vectors, weights):
            embeddings[i] += vector * weight
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-12)
        np.savez_compressed(cache, embeddings=embeddings, row_ids=selected.row_id.to_numpy(dtype=str),
                            normalized_text=normalized.to_numpy(dtype=str))
    semantic = np.clip(embeddings @ embeddings.T, -1, 1)
    np.save(out / 'semantic_similarity.npy', semantic)
    rows = []
    for method, similarity, thresholds in [('semantic', semantic, [.70, .75, .80, .85, .90]),
                                           ('lexical', lexical, [.70, .80, .90])]:
        for threshold in thresholds:
            labels = partition(similarity, threshold)
            stats = {'method': method, 'minimum_within_cluster_cosine': threshold, **partition_stats(labels)}
            rows.append(stats)
            print(stats, flush=True)
    pd.DataFrame(rows).to_csv(out / 'threshold_sensitivity.csv', index=False)
    labels = partition(semantic, args.threshold)
    groups = sorted((np.flatnonzero(labels == lab) for lab in np.unique(labels)),
                    key=lambda ids: (-len(ids), int(selected.iloc[ids].row_id.astype(int).min())))
    audit = []
    summaries = []
    selected['story_cluster_id'] = ''
    selected['story_cluster_size'] = 0
    for number, ids in enumerate(groups, 1):
        cid = f'ST{number:04}'
        sub = semantic[np.ix_(ids, ids)]
        representative = ids[np.argmax(sub.mean(axis=1))]
        selected.loc[ids, 'story_cluster_id'] = cid
        selected.loc[ids, 'story_cluster_size'] = len(ids)
        union_topics = sorted(set().union(*(set(ast.literal_eval(v)) for v in selected.iloc[ids].domains)))
        summaries.append({'story_cluster_id': cid, 'emails': len(ids),
                          'representative_row_id': selected.iloc[representative].row_id,
                          'min_pairwise_semantic_cosine': float(sub.min()),
                          'subset_ids': ' | '.join(sorted(selected.iloc[ids].subset_id.unique())),
                          'topics_union_json': json.dumps(union_topics),
                          'representative_text': selected.iloc[representative].text})
        if len(ids) > 1:
            a, b = np.unravel_index(np.argmin(np.where(np.eye(len(ids), dtype=bool), np.inf, sub)), sub.shape)
            a, b = ids[a], ids[b]
            audit.append({'story_cluster_id': cid, 'cluster_size': len(ids), 'row_id_a': selected.iloc[a].row_id,
                          'row_id_b': selected.iloc[b].row_id, 'semantic_cosine': float(semantic[a, b]),
                          'lexical_cosine': float(lexical[a, b]), 'text_a': selected.iloc[a].text,
                          'text_b': selected.iloc[b].text})
    selected.to_csv(out / 'email_story_assignments.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(summaries).to_csv(out / 'story_cluster_summary.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(audit).to_csv(out / 'cluster_audit_pairs.csv', index=False, encoding='utf-8-sig')
    selected.groupby('subset_id').agg(emails=('row_id', 'size'),
        story_clusters_present=('story_cluster_id', 'nunique')).reset_index().to_csv(out / 'subset_diversity.csv', index=False)
    write_unique_tables(selected, candidates, plan['definitions'], args.threshold, out)
    metadata = {'emails': len(selected), 'raw_unique_texts': selected.text.nunique(),
                'normalized_unique_texts': normalized.nunique(), 'model': MODEL,
                'model_revision': getattr(model[0].auto_model.config, '_commit_hash', None),
                'chunk_tokens': chunk_tokens, 'overlap_tokens': overlap, 'chunks': len(chunks),
                'clustering': 'complete linkage on cosine distance; all within-cluster similarities meet threshold',
                'selected_threshold': args.threshold, 'selected_result': partition_stats(labels),
                'interpretation': 'Exploratory semantic template-family estimate, not manually verified unique plots.',
                'normalization': 'NFKC, HTML unescape, remove markup and selected mail headers, mask URLs/emails/titled person names/numbers/amounts, casefold, collapse whitespace'}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print('DONE', json.dumps(metadata['selected_result']), flush=True)


if __name__ == '__main__':
    main()
