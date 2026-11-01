"""Domain or interaction-type annotation covering all nonempty phishing emails.

Run:     bash run.sh --model gpt-6-luna
One batch per invocation: bash run.sh --max-batches 1
Prepare: bash run.sh --stage prepare
Review:  bash run.sh --stage manual --batch b001
Report:  bash run.sh --stage report
Interaction types: bash ../run_lloom_interaction_type.sh

A single master seed fixes the complete dataset order and all subset memberships.
The saved plan is authoritative on resume; no batch or subset is redrawn.
Each batch contains up to 300 emails and repeats annotation of its SAME fixed
subset of up to 50 emails five times. Manual review is a separate export of up to
100 emails and never blocks the next batch. The final partial batch is retained.
"""
import argparse
import ast
import asyncio
import fcntl
import getpass
import hashlib
import html
import itertools
import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATASETS = HERE.parent
sys.path.insert(0, str(DATASETS))
sys.path.insert(0, str(DATASETS.parent / 'lloom/text_lloom/src'))
from lloom_domain_definitions import DOMAIN_CONTEXT, add_domain_concepts
from text_lloom import workbench as wb
from text_lloom.concept import Concept
from text_lloom.llm import OpenAIModel
from text_lloom.prompts import score_highlight_prompt, summarize_concept_prompt

LOG = logging.getLogger('annotation')
SIZES = {'main300': 300, 'stable50': 50, 'manual100': 100}
REPEATS = 5
THRESHOLD = 1.0
SCORES = {'A': 1.0, 'B': 0.75, 'C': 0.5, 'D': 0.25, 'E': 0.0}
SYSTEM_PROMPT = 'You identify patterns in complete email texts. Respond with valid JSON.'
# Standard USD rates per million tokens, checked against official docs 2026-10-24.
# These rates are estimates only; the response usage and actual model are recorded.
MODELS = {
    'gpt-6-luna': (1050000, 0.10, 0.50, 'none'),
    'gpt-6.1-sol': (1050000, 2.00, 10.00, 'low'),
    'gpt-6-astra': (1050000, 10.00, 50.00, 'low'),
    'gpt-4o-mini': (128000, 0.15, 0.60, None),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def write_csv(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def read_csv(path):
    frame = pd.read_csv(path, dtype={'row_id': str, 'doc_id': str}, keep_default_na=False)
    if 'score' in frame:
        frame['score'] = pd.to_numeric(frame['score'].replace('', np.nan), errors='raise')
    return frame


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', choices=('annotate', 'prepare', 'manual', 'report', 'serve'),
                        default='annotate')
    parser.add_argument('--batch', '--round', dest='batch', default='all',
                        help='Batch ID from the saved plan, e.g. b001, or all.')
    parser.add_argument('--max-batches', type=int,
                        help='Finish at most this many unfinished batches in this invocation.')
    parser.add_argument('--source', type=Path, default=DATASETS / 'phishing.csv')
    parser.add_argument('--dimension', choices=('domain', 'interaction_type'), default='domain')
    parser.add_argument('--output-root', type=Path,
                        help='Defaults to Annotation for domain, InteractionType for interaction types.')
    parser.add_argument('--sampling-plan', type=Path,
                        help='Reuse exact batch and subset IDs from an existing plan.')
    parser.add_argument('--seed', type=int, default=None,
                        help='First run defaults to 42; resume restores the seed from plan.json.')
    parser.add_argument('--model', default='gpt-6-luna')
    parser.add_argument('--reasoning-effort', choices=('none', 'low', 'medium', 'high', 'xhigh', 'max'))
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--max-output-tokens', type=int, default=4096)
    parser.add_argument('--attempts', type=int, default=4)
    parser.add_argument('--context-window', type=int)
    parser.add_argument('--input-price', type=float, help='USD per million uncached input tokens.')
    parser.add_argument('--output-price', type=float, help='USD per million output tokens.')
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args(argv)
    if (args.seed is not None and args.seed < 0) or args.concurrency < 1 or args.attempts < 1:
        parser.error('Use a nonnegative seed and positive concurrency/attempt counts.')
    if args.max_batches is not None and args.max_batches < 1:
        parser.error('--max-batches must be positive.')
    if not 0 <= args.temperature <= 2 or args.max_output_tokens < 256:
        parser.error('Temperature must be 0..2; max-output-tokens must be at least 256.')
    return args


def model_config(args):
    preset = MODELS.get(args.model)
    if preset is None and any(v is None for v in
                              (args.context_window, args.input_price, args.output_price)):
        raise ValueError('Custom models require --context-window, --input-price and --output-price.')
    context = args.context_window if args.context_window is not None else preset[0]
    input_price = args.input_price if args.input_price is not None else preset[1]
    output_price = args.output_price if args.output_price is not None else preset[2]
    effort = args.reasoning_effort if args.reasoning_effort is not None else (preset[3] if preset else None)
    if args.model in ('gpt-6-astra', 'gpt-6.1-sol') and effort == 'none':
        raise ValueError('This model requires reasoning effort low or higher.')
    if context <= args.max_output_tokens or min(input_price, output_price) < 0:
        raise ValueError('Invalid context window or token prices.')
    return {'model': args.model, 'context_window': context,
            'input_price_per_million': input_price, 'output_price_per_million': output_price,
            'reasoning_effort': effort,
            'temperature': args.temperature if effort in (None, 'none') else None,
            'max_output_tokens': args.max_output_tokens,
            'endpoint': 'chat.completions', 'response_format': 'json_object'}


async def collect_definitions(dimension='domain'):
    if dimension == 'interaction_type':
        from lloom_interaction_type_definitions import (
            INTERACTION_TYPE_CONTEXT, add_interaction_type_concepts)
        context, register, expected, prefix = INTERACTION_TYPE_CONTEXT, add_interaction_type_concepts, 10, 'i'
    else:
        context, register, expected, prefix = DOMAIN_CONTEXT, add_domain_concepts, 16, 'd'
    class Collector:
        in_df = pd.DataFrame(columns=['row_id', 'text'])
        doc_id_col = 'row_id'
        doc_col = 'text'

        def __init__(self):
            self.definitions = []

        async def add(self, name, prompt, **kwargs):
            self.definitions.append({'id': f'{prefix}{len(self.definitions) + 1:02}',
                                     'name': name, 'prompt': prompt, 'context': context})
    collector = Collector()
    await register(collector, debug=False)
    if len(collector.definitions) != expected or len({d['name'] for d in collector.definitions}) != expected:
        raise ValueError(f'Expected exactly {expected} distinct {dimension} definitions.')
    return collector.definitions


def validate_sample_ids(samples):
    """Check dynamic batches and their fixed subsets before annotation resumes."""
    if not samples:
        raise ValueError('The saved plan contains no batches.')
    seen = set()
    for batch_id, memberships in samples.items():
        if set(memberships) != set(SIZES):
            raise ValueError(f'Incomplete fixed sample memberships: {batch_id}')
        main_count = len(memberships['main300'])
        if not 1 <= main_count <= SIZES['main300']:
            raise ValueError(f'Invalid batch size: {batch_id}')
        expected_counts = {'main300': main_count,
                           'stable50': min(SIZES['stable50'], main_count),
                           'manual100': min(SIZES['manual100'], main_count)}
        for name, count in expected_counts.items():
            ids = memberships[name]
            if (not isinstance(ids, list) or len(ids) != count or
                    any(not isinstance(row_id, str) for row_id in ids) or len(set(ids)) != count):
                raise ValueError(f'Invalid or duplicate fixed email IDs: {batch_id}/{name}')
        main = set(memberships['main300'])
        if not set(memberships['stable50']).issubset(main):
            raise ValueError(f'{batch_id}: stability sample is outside its main batch.')
        if not set(memberships['manual100']).issubset(main):
            raise ValueError(f'{batch_id}: manual sample is outside its main batch.')
        if seen & main:
            raise ValueError(f'Saved main batches overlap: {batch_id}')
        seen.update(main)
    return seen


def append_batches(samples, available, seed):
    """Allocate remaining IDs with ONE generator, never reseeding per batch.

    Existing memberships are left intact, including checkpoints from the former
    two-round version. Its completed batches keep their original folder names.
    """
    used = {row_id for group in samples.values() for row_id in group['main300']}
    remaining = available.loc[~available.row_id.isin(used)].reset_index(drop=True)
    if remaining.empty:
        return
    rng = np.random.default_rng(seed)
    ordered = remaining.iloc[rng.permutation(len(remaining))].reset_index(drop=True)
    legacy_names = bool(samples) and next(iter(samples)).startswith('r')
    start = len(samples)
    for number, offset in enumerate(range(0, len(ordered), SIZES['main300']), start + 1):
        batch_id = f'r{number:02}' if legacy_names else f'b{number:03}'
        if batch_id in samples:
            raise ValueError(f'Unexpected duplicate batch name: {batch_id}')
        main = ordered.iloc[offset:offset + SIZES['main300']]
        stable_positions = rng.choice(len(main), size=min(50, len(main)), replace=False)
        manual_positions = rng.choice(len(main), size=min(100, len(main)), replace=False)
        samples[batch_id] = {'main300': main.row_id.tolist(),
                            'stable50': main.iloc[stable_positions].row_id.tolist(),
                            'manual100': main.iloc[manual_positions].row_id.tolist()}


async def prepare(args):
    """Freeze all dataset batches and subsets before any API requests."""
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / 'plan.json'
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else None
    reference = None
    if args.sampling_plan is not None:
        reference = json.loads(args.sampling_plan.read_text())
        validate_sample_ids(reference['samples'])
        if reference.get('sample_ids_sha256') != digest(reference['samples']):
            raise ValueError('Reference plan sample IDs fail their saved hash.')
        if reference['sizes'] != SIZES or reference['repeats'] != REPEATS:
            raise ValueError('Reference plan uses different sampling sizes or repeat count.')
        if plan is not None and plan['samples'] != reference['samples']:
            raise ValueError('Output plan and reference plan contain different fixed sample IDs.')
    existing_batches = [path for path in root.iterdir() if path.is_dir() and
                        path.name[:1] in ('b', 'r') and path.name[1:].isdigit()]
    if plan is None and existing_batches:
        raise ValueError('Batch folders exist but plan.json is missing. Restore the original '
                         'plan.json; refusing to redraw samples over existing checkpoints.')
    if plan is not None:
        if args.seed is not None and args.seed != plan['seed']:
            raise ValueError(f'This experiment uses seed {plan["seed"]}; '
                             'omit --seed to resume with the saved plan.')
        args.seed = plan['seed']
        LOG.info('Resume: using saved master seed %s and saved email IDs.', args.seed)
    elif args.seed is None:
        args.seed = reference['seed'] if reference is not None else 42
    if reference is not None and args.seed != reference['seed']:
        raise ValueError('Seed differs from the reference sampling plan.')
    definitions = await collect_definitions(args.dimension)
    config = model_config(args)
    source = args.source.resolve()
    wanted = {'source': str(source), 'source_sha256': file_digest(source),
              'seed': args.seed, 'model_config': config, 'definitions': definitions,
              'threshold': THRESHOLD, 'sizes': SIZES, 'repeats': REPEATS,
              'protocol': {'system_prompt': SYSTEM_PROMPT, 'score_prompt': score_highlight_prompt,
                           'summary_prompt': summarize_concept_prompt, 'score_batch_size': 1}}
    if args.dimension != 'domain':
        wanted['dimension'] = args.dimension
    if reference is not None:
        if reference['source_sha256'] != wanted['source_sha256']:
            raise ValueError('Source CSV differs from the reference sampling plan.')
        wanted['reference_sample_ids_sha256'] = digest(reference['samples'])
    if plan is not None and plan['fingerprint'] != digest(wanted):
        raise ValueError('Source, prompts or model settings differ from the frozen experiment. '
                         'Use the original settings to resume, or a new --output-root.')
    df = pd.read_csv(source, dtype={'row_id': str})
    if not {'text', 'label'}.issubset(df.columns):
        raise ValueError('Source CSV must contain text and label columns.')
    if 'row_id' not in df:
        df.insert(0, 'row_id', [str(i) for i in range(len(df))])
    if df.row_id.isna().any() or df.row_id.duplicated().any():
        raise ValueError('Source row IDs must be nonempty and unique.')
    df['row_id'] = df.row_id.astype(str)
    if not df.label.eq('phishing').all():
        raise ValueError('Every source row must have the phishing label.')
    available = df.loc[df.text.notna() & df.text.fillna('').str.strip().ne('')].copy()
    if available.empty:
        raise ValueError('No nonempty phishing emails are available.')
    if plan is not None:
        validate_sample_ids(plan['samples'])
        saved_hash = plan.get('sample_ids_sha256', digest(plan['samples']))
        if saved_hash != digest(plan['samples']):
            raise ValueError('Saved sample IDs were modified; refusing to reuse checkpoints.')
        saved_ids = {row_id for group in plan['samples'].values() for row_id in group['main300']}
        if not saved_ids.issubset(set(available.row_id)):
            raise ValueError('Saved emails are missing from the source.')
        if plan.get('sampling_version') != 2:
            # Extend the former two-round plan without changing any already
            # selected IDs, annotation directories, or run fingerprints.
            old_count = len(plan['samples'])
            append_batches(plan['samples'], available, args.seed)
            if 'sampling_seeds' in plan:
                plan['legacy_sampling_seeds'] = plan.pop('sampling_seeds')
            LOG.info('Preserved %s existing batches and their subsets; added %s remaining batches.',
                     old_count, len(plan['samples']) - old_count)
        else:
            LOG.info('Reusing all %s saved batches; no sampling is performed.', len(plan['samples']))
    else:
        if reference is None:
            samples = {}
            append_batches(samples, available, args.seed)
        else:
            samples = json.loads(json.dumps(reference['samples']))
            LOG.info('Reusing EXACT batch/stability/manual memberships from %s.', args.sampling_plan)
        plan = {**wanted, 'fingerprint': digest(wanted), 'created_at_utc': now(),
                'samples': samples, 'jaccard_empty_union': 1.0}
    all_main_ids = validate_sample_ids(plan['samples'])
    if all_main_ids != set(available.row_id):
        raise ValueError('Saved batches do not cover the complete nonempty source dataset.')
    plan.update({'sampling_version': 2, 'sampling_seed': args.seed,
                 'sampling_rule': 'One seeded generator fixes the entire dataset order and all subsets; '
                                  'batch memberships are authoritative on resume.',
                 'source_rows': len(df), 'eligible_emails': len(available),
                 'empty_rows_excluded': len(df) - len(available),
                 'duplicate_text_rows_retained': int(available.text.duplicated().sum()),
                 'batch_count': len(plan['samples']),
                 'sample_ids_sha256': digest(plan['samples']),
                 'review_overlap': 'Independent subset of its batch; may overlap stable50.'})
    # Save the complete ID allocation first, so interruption while writing the
    # sample CSVs can recover every membership without invoking the RNG again.
    write_json(plan_path, plan)
    indexed = df.set_index('row_id', drop=False)
    for batch_id, samples in plan['samples'].items():
        for name, ids in samples.items():
            frame = indexed.loc[ids].reset_index(drop=True)
            target = root / batch_id / '01_samples' / f'{name}.csv'
            if target.exists():
                existing = read_csv(target)
                if existing.row_id.tolist() != ids or existing.text.tolist() != frame.text.tolist():
                    raise ValueError(f'Fixed sample was edited: {target}')
            else:
                write_csv(target, frame)
        LOG.info('%s fixed sizes: main %s, stability %s, manual %s; master seed %s.',
                 batch_id, len(samples['main300']), len(samples['stable50']),
                 len(samples['manual100']), args.seed)
    LOG.info('Dataset plan: %s nonempty emails -> %s batches of up to 300; '
             'final batch %s emails. No per-batch seed changes.',
             len(available), len(plan['samples']),
             len(next(reversed(plan['samples'].values()))['main300']))
    write_index(root)
    return plan


def selected_batches(root, selection):
    plan = json.loads((root / 'plan.json').read_text())
    batches = list(plan['samples'])
    if selection == 'all':
        return batches
    if selection not in plan['samples']:
        raise ValueError(f'Unknown batch {selection}; available IDs: {", ".join(batches)}')
    return [selection]


def batch_complete(path):
    locations = ['02_labels'] + [f'03_stability/run{i:02}' for i in range(1, REPEATS + 1)]
    for location in locations:
        status = path / location / 'status.json'
        if not status.exists() or json.loads(status.read_text()).get('state') != 'complete':
            return False
    return (path / '03_stability/summary.csv').exists()


def reject_json_constant(value):
    raise ValueError(f'Non-JSON numeric constant: {value}')


def score_request_id(prompt):
    """Read the actual requested ID from the prompt's first JSON document."""
    prefix = score_highlight_prompt.split('{examples_json}', 1)[0]
    if not prompt.startswith(prefix):
        raise ValueError('Unrecognized scoring prompt; cannot verify requested email ID.')
    body = prompt[len(prefix):].lstrip()
    if body.startswith('{{'):
        # LLooM's dict_to_json doubles ALL braces for template formatting,
        # including braces inside email text. Reverse that transport escaping
        # only for that representation; preserve ordinary JSON strings intact.
        marker = score_highlight_prompt.split('{examples_json}', 1)[1].split('{concept_name}', 1)[0]
        block, separator, _ = body.partition(marker)
        if not separator:
            raise ValueError('Cannot locate the end of LLooM scoring examples.')
        examples = json.loads(block.replace('{{', '{').replace('}}', '}'))
    else:
        examples, _ = json.JSONDecoder().raw_decode(body)
    rows = examples.get('cur_examples') if isinstance(examples, dict) else None
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValueError('Expected exactly one email in the scoring request.')
    row_id = rows[0].get('example_id')
    if isinstance(row_id, bool) or not isinstance(row_id, (str, int)):
        raise ValueError('Scoring request has an invalid email ID.')
    return str(row_id)


def validate_content(content, kind, frame, expected_id=None):
    value = json.loads(content, parse_constant=reject_json_constant)
    if not isinstance(value, dict):
        raise ValueError('Expected a JSON object, not an array or scalar.')
    if kind == 'summary':
        if not isinstance(value.get('summary'), str) or not value['summary'].strip():
            raise ValueError('Missing concept summary.')
        return value
    rows = value.get('pattern_results')
    if not isinstance(rows, list) or len(rows) != 1:
        raise ValueError('Expected one scored email per response.')
    result = rows[0]
    if not isinstance(result, dict):
        raise ValueError('Expected an object for the scored email.')
    row_id = result.get('example_id')
    if isinstance(row_id, bool) or not isinstance(row_id, (str, int)):
        raise ValueError('Missing valid email ID.')
    row_id = str(row_id)
    if row_id not in frame.index:
        raise ValueError('Model returned an unknown email ID.')
    if expected_id is not None and row_id != expected_id:
        raise ValueError('Model returned a different email ID from the requested email.')
    # LLooM compares IDs directly; a JSON number must not silently miss a str ID.
    result['example_id'] = row_id
    if (not isinstance(result.get('answer'), str) or result['answer'] not in SCORES or
            not isinstance(result.get('rationale'), str) or not result['rationale'].strip()):
        raise ValueError('Missing valid A-E answer or rationale.')
    if not isinstance(result.get('quote'), str):
        raise ValueError('Missing evidence field.')
    if result['answer'] == 'A' and not result['quote'].strip():
        raise ValueError('A full match requires an evidence quote.')
    return value


def canonical_content(content, kind, frame, expected_id=None):
    """Preserve JSON meaning and escape Unicode safely for downstream parsers."""
    return json.dumps(validate_content(content, kind, frame, expected_id),
                      ensure_ascii=True, allow_nan=False)


class RecordedAPI:
    """Validated API adapter with response checkpoints scoped to ONE independent run."""
    def __init__(self, path, frame, config, args, prefix):
        self.path, self.config, self.args, self.prefix = path, config, args, prefix
        self.frame = frame.set_index('row_id', drop=False)
        self.semaphore = asyncio.Semaphore(args.concurrency)
        self.domain = None
        self.completed = 0
        self.failed = {}
        self.fatal = None
        self.blocked_path = path / 'blocked_requests.json'
        self.blocked = json.loads(self.blocked_path.read_text()) if self.blocked_path.exists() else {}
        (path / 'api').mkdir(parents=True, exist_ok=True)

    async def __call__(self, model, prompt):
        kind = 'score' if prompt.lstrip().startswith('CONTEXT:') else 'summary'
        expected_id = score_request_id(prompt) if kind == 'score' else None
        request = {'model': self.config['model'],
                   'messages': [{'role': 'system', 'content': SYSTEM_PROMPT},
                       {'role': 'user', 'content': prompt}],
                   'response_format': {'type': 'json_object'}}
        extra = {'max_completion_tokens': self.config['max_output_tokens']}
        if self.config['reasoning_effort'] is not None:
            extra['reasoning_effort'] = self.config['reasoning_effort']
        if self.config['temperature'] is not None:
            request['temperature'] = self.config['temperature']
        request['extra_body'] = extra
        key = digest(request)
        cache = self.path / 'api' / f'{key}.json'
        if key in self.blocked:
            return self.blocked_result(kind, expected_id)
        if cache.exists():
            try:
                record = json.loads(cache.read_text())
                if record['request'] != request:
                    raise ValueError('Cached response does not match this request.')
                content = canonical_content(record['content'], kind, self.frame, expected_id)
                tokens = record['tokens']
                if (not isinstance(tokens, list) or len(tokens) != 2 or
                        any(type(n) is not int or n < 0 for n in tokens)):
                    raise ValueError('Cached response has invalid token usage.')
            except (ValueError, TypeError, KeyError) as exc:
                # Preserve the bad file for inspection, then retry this ONE
                # request instead of failing on the same cache on every resume.
                quarantine = self.path / 'invalid_cache'
                quarantine.mkdir(exist_ok=True)
                stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                cache.replace(quarantine / f'{key}.{stamp}.json')
                failure = {'at_utc': now(), 'domain': self.domain, 'request_hash': key,
                           'phase': 'cache_validation', 'type': type(exc).__name__,
                           'message': str(exc)[:1500]}
                with (self.path / 'errors.jsonl').open('a') as stream:
                    stream.write(json.dumps(failure) + '\n')
                LOG.warning('%s %s: invalid response cache archived; retrying only this request.',
                            self.prefix, self.domain)
            else:
                self.failed.pop(key, None)
                self.progress(kind, resumed=True)
                return content, tuple(tokens)
        if self.fatal:
            self.failed[key] = self.fatal
            raise RuntimeError(self.fatal)
        # Never truncate a complete email to fit a context window.
        if model.count_tokens_fn(model, prompt) + self.config['max_output_tokens'] > model.context_window:
            self.failed[key] = 'Complete prompt exceeds model context; no truncation performed.'
            raise ValueError(self.failed[key])
        async with self.semaphore:
            for attempt in range(1, max(3, self.args.attempts) + 1):
                try:
                    if self.fatal:
                        raise RuntimeError(self.fatal)
                    response = await model.client.chat.completions.create(**request)
                    # Preserve EVERY billable response, including malformed answers
                    # that are rejected and retried. Never store API credentials.
                    received = {'at_utc': now(), 'domain': self.domain, 'kind': kind,
                                'request_hash': key, 'request': request,
                                'response': response.model_dump(mode='json')}
                    with (self.path / 'attempts.jsonl').open('a') as stream:
                        stream.write(json.dumps(received, ensure_ascii=False) + '\n')
                    content = response.choices[0].message.content or ''
                    if response.choices[0].finish_reason != 'stop':
                        raise ValueError('Incomplete or refused model response.')
                    content = canonical_content(content, kind, self.frame, expected_id)
                    tokens = [response.usage.prompt_tokens, response.usage.completion_tokens]
                    record = {'at_utc': now(), 'domain': self.domain, 'kind': kind,
                              'request': request, 'response': response.model_dump(mode='json'),
                              'content': content, 'tokens': tokens}
                    write_json(cache, record)
                    self.failed.pop(key, None)
                    self.progress(kind)
                    return content, tuple(tokens)
                except Exception as exc:
                    status = getattr(exc, 'status_code', None)
                    failure = {'at_utc': now(), 'domain': self.domain, 'request_hash': key,
                               'attempt': attempt, 'type': type(exc).__name__,
                               'status': status, 'message': str(exc)[:1500]}
                    with (self.path / 'errors.jsonl').open('a') as stream:
                        stream.write(json.dumps(failure) + '\n')
                    self.failed[key] = failure['message']
                    rejected = (status == 400 and 'invalid_prompt' in str(exc)) or (
                        isinstance(exc, ValueError) and str(exc) == 'Incomplete or refused model response.'
                        and 'response' in locals() and response.choices[0].finish_reason == 'content_filter')
                    if rejected:
                        if attempt < 3:
                            LOG.warning('%s %s: content rejected, attempt %s/3.', self.prefix, self.domain, attempt)
                            await asyncio.sleep(2 ** attempt)
                            continue
                        self.blocked[key] = {**failure, 'row_id': expected_id, 'kind': kind,
                                             'review_status': '待人工复核'}
                        write_json(self.blocked_path, self.blocked)
                        self.failed.pop(key, None)
                        LOG.warning('%s %s: blocked after 3 attempts; row_id=%s; continuing.',
                                    self.prefix, self.domain, expected_id)
                        return self.blocked_result(kind, expected_id)
                    if status is not None and 400 <= status < 500 and status not in (408, 409, 429):
                        self.fatal = failure['message']
                    LOG.warning('%s %s request failed (%s), attempt %s/%s.',
                                self.prefix, self.domain, failure['type'], attempt, self.args.attempts)
                    if self.fatal or attempt == self.args.attempts:
                        raise
                    await asyncio.sleep(min(30, 2 ** attempt))

    @staticmethod
    def blocked_result(kind, row_id):
        # Transport-only placeholder for LLooM; converted to a missing score
        # before validation/export. It is never saved as a successful API answer.
        if kind == 'summary':
            return json.dumps({'summary': 'blocked / 待人工复核'}), (0, 0)
        return json.dumps({'pattern_results': [{'example_id': row_id,
            'rationale': 'blocked / 待人工复核', 'quote': ''}]}), (0, 0)

    def progress(self, kind, resumed=False):
        if kind == 'score':
            self.completed += 1
            if self.completed % 5 == 0 or self.completed == len(self.frame):
                LOG.info('%s %s: emails %s/%s%s', self.prefix, self.domain,
                         self.completed, len(self.frame), ' (checkpoint)' if resumed else '')


def make_model(config, api, key):
    import tiktoken
    from openai import AsyncOpenAI
    model = OpenAIModel(name=config['model'], api_key=key, fn=api,
        setup_fn=lambda value: AsyncOpenAI(api_key=value, max_retries=0, timeout=120),
        context_window=config['context_window'],
        cost=(config['input_price_per_million'] / 1e6, config['output_price_per_million'] / 1e6),
        rate_limit=(100000, 0))
    try:
        encoding = tiktoken.encoding_for_model(config['model'])
    except KeyError:
        encoding = tiktoken.get_encoding('o200k_base')
    model.count_tokens_fn = lambda current, text: len(encoding.encode(text, disallowed_special=()))
    return model


def validate_scores(scores, frame, concepts):
    expected = {(str(row_id), cid) for row_id in frame.row_id for cid in concepts}
    actual = set(zip(scores.doc_id.astype(str), scores.concept_id))
    if len(scores) != len(expected) or actual != expected:
        raise RuntimeError('Incomplete or duplicate email-by-concept results; not treated as Unknown.')
    blocked = scores.get('status', pd.Series('', index=scores.index)).eq('blocked')
    if (not (scores.score.isin(SCORES.values()) | (blocked & scores.score.isna())).all() or
            not scores.rationale.map(lambda value: isinstance(value, str) and bool(value.strip())).all()):
        raise RuntimeError('Invalid scores or missing reasons; not treated as Unknown.')


def labels_from_scores(frame, scores, dimension='domain'):
    by_id = {str(key): group for key, group in scores.groupby('doc_id', sort=False)}
    rows = []
    for source in frame.to_dict(orient='records'):
        group = by_id[str(source['row_id'])]
        domains = tuple(sorted(set(group.loc[group.score.ge(THRESHOLD), 'concept_name'])))
        reason_key = 'interaction_type' if dimension == 'interaction_type' else 'domain'
        labels_key = 'interaction_types' if dimension == 'interaction_type' else 'domains'
        count_key = 'interaction_type_count' if dimension == 'interaction_type' else 'domain_count'
        blocked = group.get('status', pd.Series('', index=group.index)).eq('blocked').any()
        reasons = [{reason_key: item.concept_name, 'score': float(item.score) if pd.notna(item.score) else None,
                    'reason': item.rationale, 'evidence': item.highlight}
                   for item in group.itertuples()]
        rows.append({**source, labels_key: repr(domains), count_key: len(domains),
                     'status': 'blocked' if blocked else ('Known' if domains else 'Unknown'),
                     'review_status': '待人工复核' if blocked else '',
                     'reasons_json': json.dumps(reasons, ensure_ascii=False)})
    return pd.DataFrame(rows)


def export_page(session, path, title):
    """Save the native LLooM renderer, data, metadata, and a complete HTML view."""
    widget = session.vis(cols_to_show=['label', 'Dataset'], slice_col='Dataset')
    fields = ['data', 'data_items', 'data_items_wide', 'metadata', 'slice_col', 'norm_by']
    state = {key: getattr(widget, key) for key in fields}
    write_json(path / 'widget_state.json', state)
    assets = DATASETS.parent / 'lloom/text_lloom/src/text_lloom/static'
    for source, target in [('index.js', 'workbench.js'), ('index.css', 'workbench.css')]:
        shutil.copy2(assets / source, path / target)
    payload = json.dumps(state, ensure_ascii=False).replace('<', '\\u003c')
    document = '''<!doctype html><html><head><meta charset="utf-8">
<title>TITLE</title><link rel="stylesheet" href="workbench.css"></head><body>
<h2>TITLE</h2><p>One complete email per document. Outlier in LLooM means Unknown:
no defined concept scored 1.0. Inspect each concept for scores, reasons and evidence.</p>
<p><a href="labels.csv">Labels</a> | <a href="scores.csv">All scores</a> |
<a href="unknown.csv">Unknown emails</a></p>
<div id="error" style="color:red"></div><div id="workbench"></div>
<script type="module">const state=PAYLOAD;
try { const {render}=await import('./workbench.js');
const model={get:k=>state[k],set:(k,v)=>{state[k]=v},on:()=>{},off:()=>{},save_changes:()=>{}};
render({model,el:document.getElementById('workbench')});
} catch(e) {document.getElementById('error').textContent='Serve over HTTP; allow esm.sh. '+e;}
</script></body></html>'''
    (path / 'index.html').write_text(document.replace('TITLE', html.escape(title)).replace('PAYLOAD', payload))


def api_usage(path, config):
    total = {'requests': 0, 'input_tokens': 0, 'output_tokens': 0, 'cached_input_tokens': 0}
    models = set()
    records = [json.loads(line) for line in (path / 'attempts.jsonl').read_text().splitlines() if line.strip()] if (path / 'attempts.jsonl').exists() else []
    for record in records:
        usage = record['response'].get('usage') or {}
        total['requests'] += 1
        total['input_tokens'] += usage.get('prompt_tokens', 0)
        total['output_tokens'] += usage.get('completion_tokens', 0)
        total['cached_input_tokens'] += (usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0)
        models.add(record['response'].get('model', 'unknown'))
    total['models_returned'] = sorted(models)
    total['uncached_list_price_upper_estimate_usd'] = (
        total['input_tokens'] * config['input_price_per_million'] +
        total['output_tokens'] * config['output_price_per_million']) / 1e6
    return total


async def annotate_run(path, frame, plan, args, key, prefix):
    path.mkdir(parents=True, exist_ok=True)
    manifest = {'experiment': plan['fingerprint'], 'sample_ids': frame.row_id.tolist(),
                'sample_text_hash': digest(frame[['row_id', 'text']].to_dict(orient='records')),
                'run': str(path.relative_to(args.output_root))}
    manifest_path = path / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError(f'Run settings differ from its checkpoint: {path}')
    write_json(manifest_path, manifest)
    status_path = path / 'status.json'
    if status_path.exists() and json.loads(status_path.read_text()).get('state') == 'complete':
        LOG.info('%s complete; keeping this independent run, no new requests.', prefix)
        return read_csv(path / 'labels.csv')
    write_json(status_path, {'state': 'running', 'started_at_utc': now()})
    api = RecordedAPI(path, frame, plan['model_config'], args, prefix)
    model = make_model(plan['model_config'], api, key)
    session = wb.lloom(df=frame, text_col='text', id_col='row_id',
                       distill_model=model, synth_model=model, score_model=model,
                       cluster_model=SimpleNamespace(name='unused'))
    session.df_to_score = frame
    session.df_filtered = frame[['row_id', 'text']].copy()
    session.df_bullets = session.df_filtered.copy()
    for definition in plan['definitions']:
        concept = Concept(definition['name'], definition['prompt'], [], True, seed=definition['context'])
        concept.id = definition['id']
        session.concepts[concept.id] = concept
    parts = path / 'parts'
    parts.mkdir(exist_ok=True)
    try:
        for number, (cid, concept) in enumerate(session.concepts.items(), 1):
            api.domain, api.completed = cid, 0
            part = parts / f'{cid}.csv'
            metadata = parts / f'{cid}.json'
            LOG.info('%s concept %s/%s: %s', prefix, number, len(session.concepts), concept.name)
            if part.exists() and metadata.exists():
                scores = read_csv(part)
                validate_scores(scores, frame, {cid: concept})
                session.results[cid] = scores
                concept.summary = json.loads(metadata.read_text())['summary']
                LOG.info('%s %s: restored %s email scores.', prefix, cid, len(scores))
                continue
            # LLooM shuffles evidence with pandas.sample() when producing a
            # concept summary. Keep that local order stable on resume so an
            # already saved summary request can use its own run's checkpoint.
            # This does not seed the remote model or share responses across runs.
            np.random.seed(plan['seed'] % (2 ** 32))
            await session.score(c_ids=[cid], df=frame, get_highlights=True,
                                batch_size=1, debug=False)
            if api.failed:
                raise RuntimeError('Some requests failed. Resume this same command; '
                                   'failed calls are not valid Unknown annotations.')
            scores = session.results[cid].copy()
            blocked_ids = {str(item['row_id']) for item in api.blocked.values()
                           if item['kind'] == 'score' and item['domain'] == cid}
            mask = scores.doc_id.astype(str).isin(blocked_ids)
            scores['status'] = np.where(mask, 'blocked', 'scored')
            scores.loc[mask, 'score'] = np.nan
            session.results[cid] = scores
            validate_scores(scores, frame, {cid: concept})
            write_csv(part, scores)
            write_json(metadata, {'name': concept.name, 'summary': concept.summary})
        scores = pd.concat(list(session.results.values()), ignore_index=True)
        validate_scores(scores, frame, session.concepts)
        scores['answer'] = scores.score.map({value: key for key, value in SCORES.items()})
        scores['evidence_verified'] = [bool(item.highlight) and item.highlight in item.text
                                       for item in scores.itertuples()]
        labels = labels_from_scores(frame, scores, plan.get('dimension', 'domain'))
        write_csv(path / 'scores.csv', scores)
        write_csv(path / 'labels.csv', labels)
        write_csv(path / 'blocked.csv', labels.loc[labels.status.eq('blocked')])
        write_csv(path / 'unknown.csv', labels.loc[labels.status.eq('Unknown')])
        scores.pivot(index='doc_id', columns='concept_name', values='score').to_csv(path / 'matrix.csv')
        write_json(path / 'concepts.json', {cid: item.to_dict() for cid, item in session.concepts.items()})
        session.save(str(path), file_name='session')
        export_page(session, path, f'{prefix}: {len(frame)} complete emails')
        usage = api_usage(path, plan['model_config'])
        write_json(path / 'usage.json', usage)
        write_json(status_path, {'state': 'complete', 'finished_at_utc': now(),
            'emails': len(labels), 'unknown': int(labels.status.eq('Unknown').sum()),
            'blocked': int(labels.status.eq('blocked').sum()), 'score_rows': len(scores)})
        LOG.info('%s DONE: %s emails, %s Unknown, %s score rows; usage in usage.json.',
                 prefix, len(labels), int(labels.status.eq('Unknown').sum()), len(scores))
        return labels
    except BaseException as exc:
        write_json(status_path, {'state': 'incomplete', 'at_utc': now(),
                                'error': type(exc).__name__, 'message': str(exc)[:1000]})
        raise
    finally:
        await model.client.close()


def domain_set(value):
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, tuple) or any(not isinstance(item, str) for item in parsed):
        raise ValueError('Concept labels must be a tuple of strings.')
    return set(parsed)


def label_column(frame):
    columns = [name for name in ('domains', 'interaction_types') if name in frame]
    if len(columns) != 1:
        raise ValueError('Expected exactly one concept-label column.')
    return columns[0]


def compare_labels(left, right, ids, run_a, run_b):
    column = label_column(left)
    if label_column(right) != column:
        raise ValueError('Cannot compare different annotation dimensions.')
    a = left.set_index('row_id')
    b = right.set_index('row_id')
    if not a.index.is_unique or not b.index.is_unique:
        raise ValueError('Duplicate email IDs in label files.')
    rows = []
    for row_id in ids:
        if a.loc[row_id, 'status'] == 'blocked' or b.loc[row_id, 'status'] == 'blocked':
            continue  # A blocked annotation is not an empty/Unknown label set.
        first, second = domain_set(a.loc[row_id, column]), domain_set(b.loc[row_id, column])
        union, intersection = first | second, first & second
        rows.append({'row_id': row_id, 'run_a': run_a, 'run_b': run_b,
                     f'{column}_a': repr(tuple(sorted(first))), f'{column}_b': repr(tuple(sorted(second))),
                     'intersection': len(intersection), 'union': len(union),
                     'jaccard': len(intersection) / len(union) if union else 1.0,
                     'exact_match': first == second, 'both_unknown': not union})
    return pd.DataFrame(rows, columns=['row_id', 'run_a', 'run_b', f'{column}_a', f'{column}_b',
                                      'intersection', 'union', 'jaccard', 'exact_match', 'both_unknown'])


def summarize(frame, kind, name):
    nonempty = frame.loc[~frame.both_unknown.astype(bool)]
    return {'kind': kind, 'comparison': name, 'email_comparisons': len(frame),
            'mean_jaccard': float(frame.jaccard.mean()),
            'exact_match_rate': float(frame.exact_match.mean()),
            'both_unknown_count': int(frame.both_unknown.sum()),
            'mean_jaccard_excluding_both_unknown': float(nonempty.jaccard.mean()) if len(nonempty) else None}


def stability_report(root, batch_id):
    folder = root / batch_id
    ids = read_csv(folder / '01_samples/stable50.csv').row_id.tolist()
    initial = completed_labels(folder / '02_labels')
    runs = {f'run{i:02}': completed_labels(folder / f'03_stability/run{i:02}')
            for i in range(1, REPEATS + 1)}
    pairwise, baseline, summaries = [], [], []
    for run_a, run_b in itertools.combinations(runs, 2):
        comparison = compare_labels(runs[run_a], runs[run_b], ids, run_a, run_b)
        pairwise.append(comparison)
        summaries.append(summarize(comparison, 'pairwise', f'{run_a}-{run_b}'))
    for name, labels in runs.items():
        comparison = compare_labels(initial, labels, ids, 'initial', name)
        baseline.append(comparison)
        summaries.append(summarize(comparison, 'vs_initial', name))
    pairwise = pd.concat(pairwise, ignore_index=True)
    baseline = pd.concat(baseline, ignore_index=True)
    summaries += [summarize(pairwise, 'pairwise', 'overall'),
                  summarize(baseline, 'vs_initial', 'overall')]
    target = folder / '03_stability'
    write_csv(target / 'jaccard.csv', pairwise)
    write_csv(target / 'vs_initial.csv', baseline)
    write_csv(target / 'summary.csv', pd.DataFrame(summaries))
    per_email = pairwise.groupby('row_id', sort=False).agg(
        mean_pairwise_jaccard=('jaccard', 'mean'), exact_match_rate=('exact_match', 'mean'),
        both_unknown_pairs=('both_unknown', 'sum')).reset_index()
    against_initial = baseline.groupby('row_id').jaccard.mean()
    per_email['mean_vs_initial_jaccard'] = per_email.row_id.map(against_initial)
    column = label_column(initial)
    per_email['unknown_runs'] = [sum(labels.set_index('row_id').loc[row_id, 'status'] == 'Unknown'
                                      for labels in runs.values()) for row_id in per_email.row_id]
    write_csv(target / 'by_email.csv', per_email)
    overall = summaries[-2]
    LOG.info('%s STABILITY DONE: mean Jaccard %.4f; exact match %.2f%%; both-Unknown pairs %s/%s.',
             batch_id, overall['mean_jaccard'], overall['exact_match_rate'] * 100,
             overall['both_unknown_count'], len(pairwise))
    return overall


def completed_labels(path):
    status = path / 'status.json'
    if not status.exists() or json.loads(status.read_text()).get('state') != 'complete':
        raise RuntimeError(f'Annotation is not complete: {path}')
    return read_csv(path / 'labels.csv')


def export_manual(root, batch_id):
    folder = root / batch_id
    labels_path = folder / '02_labels/labels.csv'
    status_path = folder / '02_labels/status.json'
    if (not labels_path.exists() or not status_path.exists() or
            json.loads(status_path.read_text())['state'] != 'complete'):
        LOG.info('%s manual export pending: initial annotation is not complete.', batch_id)
        return
    sample = read_csv(folder / '01_samples/manual100.csv')
    labels = read_csv(labels_path).set_index('row_id', drop=False)
    review = labels.loc[sample.row_id].reset_index(drop=True)
    for name in (f'human_{label_column(labels)}', 'review_notes', 'reviewer'):
        review[name] = ''
    path = folder / '04_manual/review.csv'
    if path.exists():
        if read_csv(path).row_id.tolist() != sample.row_id.tolist():
            raise ValueError(f'Manual file membership differs from its fixed sample: {path}')
        LOG.info('%s keeping existing manual review file; human edits are not overwritten.', batch_id)
    else:
        write_csv(path, review)
        LOG.info('%s manual dataset exported: %s emails -> %s', batch_id, len(review), path)


def write_index(root):
    plan = json.loads((root / 'plan.json').read_text())
    title = 'Interaction type annotation' if plan.get('dimension') == 'interaction_type' else 'Domain annotation'
    rows = []
    for batch_id in selected_batches(root, 'all'):
        rows.append(f'<h2>Batch {batch_id}</h2><ul>')
        locations = [('Initial labels', '02_labels')] + [
            (f'Stability {i}/5', f'03_stability/run{i:02}') for i in range(1, 6)]
        for title, location in locations:
            page = f'{batch_id}/{location}/index.html'
            rows.append(f'<li><a href="{page}">{title}</a></li>' if (root / page).exists()
                        else f'<li>{title}: pending</li>')
        for title, location in [('Jaccard summary', '03_stability/summary.csv'),
                                ('Manual dataset', '04_manual/review.csv')]:
            if (root / batch_id / location).exists():
                rows.append(f'<li><a href="{batch_id}/{location}">{title}</a></li>')
        rows.append('</ul>')
    (root / 'index.html').write_text('<!doctype html><meta charset="utf-8">'
        f'<title>{title}</title><h1>{title}</h1>' + ''.join(rows))


async def run_annotation(args, plan):
    selected = selected_batches(args.output_root, args.batch)
    key = None
    processed = 0
    for position, batch_id in enumerate(selected, 1):
        root = args.output_root / batch_id
        if batch_complete(root):
            LOG.info('Batch %s (%s/%s) already complete; skipping all model calls.',
                     batch_id, position, len(selected))
            continue
        if args.max_batches is not None and processed >= args.max_batches:
            LOG.info('Reached --max-batches %s. Resume the same command for the next unfinished batch.',
                     args.max_batches)
            break
        if key is None:
            key = os.environ.get('OPENAI_API_KEY')
            if not key:
                if not sys.stdin.isatty():
                    raise RuntimeError('OPENAI_API_KEY is not set. Run this command in an '
                                       'interactive terminal to enter the key securely and resume.')
                key = getpass.getpass('OpenAI API key: ')
            if not key.strip():
                raise ValueError('An OpenAI API key is required for annotation.')
        main = read_csv(root / '01_samples/main300.csv')
        stable = read_csv(root / '01_samples/stable50.csv')
        LOG.info('=== Batch %s (%s/%s): initial annotation %s emails ===',
                 batch_id, position, len(selected), len(main))
        initial = await annotate_run(root / '02_labels', main, plan, args, key, f'{batch_id}/initial')
        write_index(args.output_root)
        for repeat in range(1, REPEATS + 1):
            LOG.info('=== Batch %s stability %s/%s: SAME fixed %s emails ===',
                     batch_id, repeat, REPEATS, len(stable))
            labels = await annotate_run(root / f'03_stability/run{repeat:02}', stable, plan,
                                        args, key, f'{batch_id}/stability {repeat}/5')
            comparison = compare_labels(initial, labels, stable.row_id.tolist(), 'initial', f'run{repeat:02}')
            result = summarize(comparison, 'vs_initial', f'run{repeat:02}')
            LOG.info('%s stability %s/5 RESULT: Jaccard vs initial %.4f; exact match %.2f%%.',
                     batch_id, repeat, result['mean_jaccard'], result['exact_match_rate'] * 100)
            write_index(args.output_root)
        stability_report(args.output_root, batch_id)
        LOG.info('Batch %s finished. Manual export is separate: --stage manual --batch %s. '
                 'No waiting for human review; continuing workflow.', batch_id, batch_id)
        processed += 1
    write_index(args.output_root)


async def main():
    args = parse_args()
    if args.output_root is None:
        args.output_root = DATASETS / 'InteractionType' if args.dimension == 'interaction_type' else HERE
    args.output_root = args.output_root.resolve()
    if args.dimension == 'interaction_type':
        if args.output_root == HERE.resolve():
            raise ValueError('Interaction type output must be separate from Domain Annotation.')
        if args.sampling_plan is None:
            args.sampling_plan = HERE / 'plan.json'
    args.output_root.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        datefmt='%H:%M:%S', stream=sys.stdout)
    if args.stage == 'serve':
        from functools import partial
        from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
        LOG.info('Forward port %s; open http://localhost:%s/', args.port, args.port)
        with ThreadingHTTPServer(('127.0.0.1', args.port), partial(
                SimpleHTTPRequestHandler, directory=str(args.output_root))) as server:
            server.serve_forever()
        return
    if args.stage in ('manual', 'report'):
        selected = selected_batches(args.output_root, args.batch)
        for batch_id in selected:
            if args.stage == 'manual':
                export_manual(args.output_root, batch_id)
            else:
                stability_report(args.output_root, batch_id)
        write_index(args.output_root)
        return
    # Only preparation and model calls are locked. Manual export can run while
    # a later round is annotating, and it never waits for human input.
    with (args.output_root / '.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another annotation process is already using this output folder.')
        plan = await prepare(args)
        LOG.info('Frozen model: %s; threshold %.2f; %s fixed batches; master seed %s; '
                 '5 extra stability repeats per batch.',
                 plan['model_config']['model'], THRESHOLD, len(plan['samples']), plan['seed'])
        if args.stage == 'annotate':
            await run_annotation(args, plan)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        LOG.info('Interrupted by Ctrl+C. Saved response checkpoints are retained; '
                 'rerun the same command to resume.')
        sys.exit(130)
