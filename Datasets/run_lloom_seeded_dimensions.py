"""Score the current user-defined domains and interaction types for ten emails."""
import argparse
import asyncio
import hashlib
import inspect
import json
import os
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'lloom/text_lloom/src'))
from text_lloom import workbench as wb
from lloom_domain_definitions import DOMAIN_CONTEXT, add_domain_concepts
from lloom_interaction_type_definitions import (
    INTERACTION_TYPE_CONTEXT, add_interaction_type_concepts,
)

OUT = ROOT / 'lloom_defined_dimensions_10'
TASKS = {
    'domain': (DOMAIN_CONTEXT, add_domain_concepts, 16),
    'interaction_type': (INTERACTION_TYPE_CONTEXT, add_interaction_type_concepts, 10),
}


def export_page(session, out):
    widget = session.vis(cols_to_show=['label','Dataset'], slice_col='Dataset')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matrix = pd.read_csv(out/'scores.csv').pivot_table(index='doc_id', columns='concept_name', values='score', aggfunc='max')
    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(matrix.values, vmin=0, vmax=1, cmap='Blues', aspect='auto')
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=40, ha='right')
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_ylabel('Original row ID')
    ax.set_title('LLooM concept scores: 10 phishing-labelled emails')
    fig.colorbar(im, ax=ax, label='Concept match score')
    fig.tight_layout()
    fig.savefig(out/'concept_scores.png', dpi=160)
    plt.close(fig)
    fields = ['data','data_items','data_items_wide','metadata','slice_col','norm_by']
    state = {k:getattr(widget,k) for k in fields}
    (out/'widget_state.json').write_text(json.dumps(state,ensure_ascii=False))
    assets = ROOT.parent/'lloom/text_lloom/src/text_lloom/static'
    # Use the official LLooM Svelte renderer with its read-only model interface.
    (out/'workbench.js').write_text((assets/'index.js').read_text())
    (out/'workbench.css').write_text((assets/'index.css').read_text())
    payload = json.dumps(state,ensure_ascii=False).replace('<','\\u003c')
    html = '''<!doctype html><html><head><meta charset="utf-8"><title>LLooM — 10 phishing emails</title>
<link rel="stylesheet" href="workbench.css"></head><body>
<h2>LLooM Workbench — 10 phishing-labelled emails</h2>
<p>Click a concept or matrix cell to inspect examples, criteria, scores and evidence. Labels were renamed from spam; they are source labels, not verified attack outcomes.</p>
<div id="error" style="color:red"></div><div id="workbench"></div>
<script type="module">
const state = PAYLOAD;
try {
 const {render} = await import('./workbench.js');
 const model = {get:key=>state[key],set:(key,value)=>{state[key]=value},on:()=>{},off:()=>{},save_changes:()=>{}};
 render({model,el:document.getElementById('workbench')});
} catch(e) {document.getElementById('error').textContent='Could not load Workbench: '+e+' . Serve this directory over HTTP and allow access to esm.sh.';}
</script></body></html>'''.replace('PAYLOAD',payload)
    (out/'index.html').write_text(html)

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dimension', choices=('domain', 'interaction_type', 'both'),
                        default='domain', help='Run one task at a time (default: domain).')
    parser.add_argument('--rerun', action='store_true',
                        help='Rescore even if the input and definitions have not changed.')
    return parser.parse_args(argv)


def input_signature(dim, df):
    seed, add_concepts, _ = TASKS[dim]
    payload = {'context': seed, 'definitions': inspect.getsource(add_concepts),
               'emails': df[['row_id', 'text']].to_dict(orient='records')}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


async def run_dimension(dim, df, rerun=False):
    target = OUT / dim
    target.mkdir(parents=True, exist_ok=True)
    seed, add_concepts, expected_concepts = TASKS[dim]
    signature = input_signature(dim, df)
    config_path = target / 'generation_config.json'
    cached = all((target / name).exists() for name in (
        'generation_config.json', 'session.pkl', 'scores.csv'))
    if cached and not rerun:
        config = json.loads(config_path.read_text())
        if config.get('input_signature') == signature:
            with (target / 'session.pkl').open('rb') as f:
                session = pickle.load(f)
            export_page(session, target)
            print(f'{dim}: reusing scores for the same emails and definitions', flush=True)
            return pd.read_csv(target / 'scores.csv'), session
        print(f'{dim}: definitions or input changed; rescoring', flush=True)

    session = wb.lloom(df=df, text_col='text', id_col='row_id')
    print(f'{dim}: scoring {expected_concepts} defined concepts against {len(df)} complete emails',
          flush=True)
    # add() immediately scores each named concept. No generation or sentence splitting.
    await add_concepts(session, debug=False)
    for cid, concept in session.concepts.items():
        concept.seed = seed
        session.results[cid]['concept_seed'] = seed
    scores = pd.concat(list(session.results.values()), ignore_index=True)
    expected_pairs = {(str(doc_id), cid) for doc_id in df.row_id
                      for cid in session.concepts}
    actual_pairs = set(zip(scores.doc_id.astype(str), scores.concept_id))
    if (len(session.concepts) != expected_concepts or
            len(scores) != len(df) * expected_concepts or actual_pairs != expected_pairs):
        raise RuntimeError(f'{dim}: incomplete document-by-concept scores')

    (target / 'seed.txt').write_text(seed)
    session.df_to_score.to_csv(target / 'input_emails.csv', index=False)
    (target / 'concepts.json').write_text(json.dumps(
        {k: v.to_dict() for k, v in session.concepts.items()}, indent=2))
    scores.to_csv(target / 'scores.csv', index=False)
    scores.pivot(index='doc_id', columns='concept_name', values='score').to_csv(
        target / 'score_matrix.csv')
    session.save(str(target), file_name='session')
    export_page(session, target)
    # Mark the cache reusable only after the scoring and exports finish successfully.
    config = {'unit': 'complete email', 'concept_mode': 'user_defined',
              'registration': 'l.add(name=..., prompt=...)',
              'concepts': len(session.concepts), 'auto_generation': False,
              'definition_file': Path(inspect.getfile(add_concepts)).name,
              'input_signature': signature}
    config_path.write_text(json.dumps(config, indent=2))
    print(dim, 'concepts', len(session.concepts), 'scores', len(scores), flush=True)
    return scores, session


async def main():
    args = parse_args()
    if not os.environ.get('OPENAI_API_KEY'):
        raise SystemExit('Set OPENAI_API_KEY in the environment.')
    df = pd.read_csv(OUT / 'sample.csv', dtype={'row_id': str})
    if len(df) != 10 or not df.label.eq('phishing').all():
        raise RuntimeError('Expected the original ten phishing-labelled sample emails.')
    summaries = []
    dimensions = tuple(TASKS) if args.dimension == 'both' else (args.dimension,)
    for dim in dimensions:
        scores, session = await run_dimension(dim, df, rerun=args.rerun)
        # This summary records full matches; scores.csv retains all graded scores.
        full = scores.loc[scores.score.eq(1) & scores.concept_id.isin(session.concepts)]
        matches = {str(k): sorted(v.concept_name.unique().tolist())
                   for k, v in full.groupby('doc_id')}
        for row_id in df.row_id:
            names = matches.get(str(row_id), [])
            summaries.append({'row_id': row_id, 'dimension': dim,
                              'assigned_concepts': ' | '.join(names) if names else 'Undetermined',
                              'n_full_matches': len(names)})
        (OUT / dim / 'status.json').write_text(json.dumps({
            'status': 'complete', 'dimension': dim, 'concepts': len(session.concepts),
            'score_rows': len(scores), 'score_threshold_for_assignment': 1.0,
            'unknown_definition': 'No concept scored 1.0; not a claim of absolute absence',
        }, indent=2))
    current = pd.DataFrame(summaries)
    previous_path = OUT / 'assignments.csv'
    if previous_path.exists():
        previous = pd.read_csv(previous_path, dtype={'row_id': str})
        current = pd.concat([previous.loc[~previous.dimension.isin(dimensions)], current],
                            ignore_index=True)
    current.to_csv(previous_path, index=False)
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == '__main__':
    asyncio.run(main())
