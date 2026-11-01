"""Offline regression checks for shared Domain/Interaction Type JSON handling."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pandas as pd

import annotate as annotation
from text_lloom.concept import Concept
from text_lloom.concept_induction import get_ex_batch_args, get_score_df, json_load


def payload(row_id='123', answer='A', quote='Payment of \u0080 712,700,000.00', rationale='Payment is requested.'):
    return json.dumps({'pattern_results': [{'example_id': row_id, 'answer': answer,
        'rationale': rationale, 'quote': quote}]}, ensure_ascii=False)


def prompt(row_id='123'):
    return annotation.score_highlight_prompt.format(
        examples_json=json.dumps({'cur_examples': [{'example_id': row_id, 'example_text': 'mail {not a prompt}'}]}),
        concept_name='Financial Transactions', concept_prompt='Does this email request a payment?')


def response(content, finish='stop'):
    data = {'choices': [{'message': {'content': content}, 'finish_reason': finish}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 20}, 'model': 'offline-test'}
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish)],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20), model_dump=lambda **kwargs: data)


class JSONValidationTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame([{'row_id': '123', 'text': 'first'}, {'row_id': '456', 'text': 'second'}])
        self.indexed = self.frame.set_index('row_id', drop=False)

    def test_original_unicode_failure_and_evidence_round_trip(self):
        # The historical failure was valid JSON with a U+0080 currency byte;
        # YAML rejected it and LLooM generated an empty default score row.
        content = payload(answer='E')
        self.assertEqual(json_load(content, 'pattern_results')[0]['answer'], 'E')
        canonical = annotation.canonical_content(content, 'score', self.indexed, '123')
        self.assertTrue(canonical.isascii())
        self.assertEqual(json.loads(canonical)['pattern_results'][0]['quote'], 'Payment of \u0080 712,700,000.00')
        concept = Concept('Financial Transactions', 'Payment?', [], True, seed='context')
        scores = get_score_df(canonical, self.frame.iloc[:1], concept, 'i03', 'text', 'row_id', True)
        annotation.validate_scores(scores, self.frame.iloc[:1], {'i03': concept})
        self.assertEqual(scores.iloc[0].score, 0.0)
        self.assertTrue(scores.iloc[0].rationale)

    def test_numeric_id_is_normalized_before_lloom_matches_it(self):
        canonical = annotation.canonical_content(payload(row_id=123), 'score', self.indexed, '123')
        self.assertEqual(json.loads(canonical)['pattern_results'][0]['example_id'], '123')
        concept = Concept('Financial Transactions', 'Payment?', [], True, seed='context')
        scores = get_score_df(canonical, self.frame.iloc[:1], concept, 'i03', 'text', 'row_id', True)
        annotation.validate_scores(scores, self.frame.iloc[:1], {'i03': concept})
        self.assertEqual(scores.iloc[0].score, 1.0)

    def test_invalid_response_shapes_and_fields_are_rejected(self):
        bad = ['not JSON', '[]', 'null', '{"pattern_results":[null]}',
               '{"pattern_results":[[]]}', '{"pattern_results":[]}',
               payload(row_id='999'), payload(row_id=True), payload(row_id='456'),
               payload(answer='F'), payload(answer=['A']), payload(rationale=''),
               payload(quote=''), payload(quote=None),
               '{"pattern_results":[],"extra":NaN}']
        for content in bad:
            with self.subTest(content=content), self.assertRaises((ValueError, TypeError)):
                annotation.canonical_content(content, 'score', self.indexed, '123')

    def test_summary_must_be_nonempty_json_object(self):
        self.assertEqual(annotation.validate_content('{"summary":"ok"}', 'summary', self.indexed)['summary'], 'ok')
        for content in ['[]', '{"summary":""}', '{"summary":null}']:
            with self.subTest(content=content), self.assertRaises(ValueError):
                annotation.validate_content(content, 'summary', self.indexed)

    def test_expected_id_comes_from_first_request_json_not_email_body(self):
        self.assertEqual(annotation.score_request_id(prompt()), '123')

    def test_real_lloom_brace_escaping_and_nested_email_text(self):
        frame = pd.DataFrame([{'row_id': '123', 'text': 'mail {nested} {{literal}}; example_id: 456'}])
        args = get_ex_batch_args(frame, 'text', 'row_id', 'Financial Transactions', 'Payment?')
        real_prompt = annotation.score_highlight_prompt.format(**args)
        self.assertIn('{{"cur_examples"', real_prompt)
        self.assertEqual(annotation.score_request_id(real_prompt), '123')

    def test_nan_rationale_is_not_a_valid_unknown(self):
        scores = pd.DataFrame([{'doc_id': '123', 'concept_id': 'i01', 'score': 0.0, 'rationale': float('nan')}])
        with self.assertRaises(RuntimeError):
            annotation.validate_scores(scores, self.frame.iloc[:1], {'i01': None})


class RecordedAPITests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.frame = pd.DataFrame([{'row_id': '123', 'text': 'first'}, {'row_id': '456', 'text': 'second'}])
        self.config = {'model': 'offline-test', 'max_output_tokens': 4096, 'reasoning_effort': None, 'temperature': 0.0}
        self.args = SimpleNamespace(concurrency=2, attempts=3)

    def tearDown(self):
        self.temp.cleanup()

    def model(self, responses):
        create = AsyncMock(side_effect=responses)
        return SimpleNamespace(client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
            count_tokens_fn=lambda model, text: 1, context_window=20000)

    def api(self, folder='run01'):
        p = self.path / folder
        p.mkdir(exist_ok=True)
        api = annotation.RecordedAPI(p, self.frame, self.config, self.args, 'offline-test')
        api.domain = 'i01'
        return api

    async def test_policy_rejection_three_attempts_then_resume_and_continue(self):
        class Rejected(Exception):
            status_code = 400
        api = self.api()
        self.args.attempts = 4
        model = self.model([Rejected('invalid_prompt')] * 3 + [response(payload('456'))])
        with patch.object(annotation.asyncio, 'sleep', new=AsyncMock()):
            content, _ = await api(model, prompt())
        self.assertEqual(model.client.chat.completions.create.await_count, 3)
        self.assertIsNone(api.fatal)
        self.assertFalse(api.failed)
        self.assertEqual(json.loads(content)['pattern_results'][0]['rationale'], 'blocked / 待人工复核')
        await self.api()(model, prompt())
        self.assertEqual(model.client.chat.completions.create.await_count, 3)
        await api(model, prompt('456'))
        self.assertEqual(model.client.chat.completions.create.await_count, 4)
        concept = Concept('Finance', 'Payment?', [], True, seed='context')
        scores = get_score_df(content, self.frame.iloc[:1], concept, 'i01', 'text', 'row_id', True)
        scores['status'] = 'blocked'
        scores['score'] = float('nan')
        annotation.validate_scores(scores, self.frame.iloc[:1], {'i01': concept})
        annotation.write_csv(api.path / 'part.csv', scores)
        scores = annotation.read_csv(api.path / 'part.csv')
        annotation.validate_scores(scores, self.frame.iloc[:1], {'i01': concept})
        labels = annotation.labels_from_scores(self.frame.iloc[:1], scores, 'interaction_type')
        self.assertEqual(labels.iloc[0].status, 'blocked')
        self.assertEqual(labels.iloc[0].review_status, '待人工复核')
        comparison = annotation.compare_labels(labels, labels, ['123'], 'a', 'b')
        self.assertTrue(comparison.empty)
        summary = annotation.summarize(comparison, 'pairwise', 'test')
        self.assertEqual(summary['both_unknown_count'], 0)

    async def test_live_malformed_json_is_recorded_and_retried(self):
        api = self.api()
        model = self.model([response('{broken'), response(payload(row_id=123))])
        with patch.object(annotation.asyncio, 'sleep', new=AsyncMock()):
            content, tokens = await api(model, prompt())
        self.assertEqual(model.client.chat.completions.create.await_count, 2)
        self.assertEqual(tokens, (10, 20))
        self.assertEqual(json.loads(content)['pattern_results'][0]['example_id'], '123')
        self.assertFalse(api.failed)
        self.assertEqual(len((api.path / 'attempts.jsonl').read_text().splitlines()), 2)
        self.assertEqual(len(list((api.path / 'api').glob('*.json'))), 1)

    async def test_wrong_known_id_and_truncated_response_are_retried(self):
        api = self.api()
        model = self.model([response(payload('456')), response(payload(), finish='length'), response(payload())])
        with patch.object(annotation.asyncio, 'sleep', new=AsyncMock()):
            await api(model, prompt())
        self.assertEqual(model.client.chat.completions.create.await_count, 3)
        self.assertFalse(api.failed)

    async def test_valid_cache_is_reused_without_new_model_request(self):
        api = self.api()
        model = self.model([response(payload())])
        first = await api(model, prompt())
        second = await self.api()(model, prompt())
        self.assertEqual(first, second)
        self.assertEqual(model.client.chat.completions.create.await_count, 1)

    async def test_bad_cache_is_archived_and_only_that_request_is_reissued(self):
        api = self.api()
        model = self.model([response(payload()), response(payload())])
        await api(model, prompt())
        cache = next((api.path / 'api').glob('*.json'))
        for corrupt in ['{broken', 'wrong_id', 'wrong_tokens']:
            original = json.loads(cache.read_text())
            if corrupt == '{broken':
                cache.write_text(corrupt)
            else:
                original['content'] = payload('456') if corrupt == 'wrong_id' else original['content']
                if corrupt == 'wrong_tokens':
                    original['tokens'] = [True, -1]
                cache.write_text(json.dumps(original))
            model = self.model([response(payload())])
            recovered = await self.api()(model, prompt())
            self.assertEqual(model.client.chat.completions.create.await_count, 1)
            self.assertEqual(json.loads(recovered[0])['pattern_results'][0]['example_id'], '123')
        self.assertEqual(len(list((api.path / 'invalid_cache').glob('*.json'))), 3)

    async def test_exhausted_retries_remain_failed_and_are_not_cached(self):
        api = self.api()
        model = self.model([response('{broken')] * 3)
        with patch.object(annotation.asyncio, 'sleep', new=AsyncMock()):
            with self.assertRaises(ValueError):
                await api(model, prompt())
        self.assertTrue(api.failed)
        self.assertFalse(list((api.path / 'api').glob('*.json')))
        self.assertEqual(len((api.path / 'attempts.jsonl').read_text().splitlines()), 3)

    async def test_stability_runs_do_not_share_response_cache(self):
        model = self.model([response(payload()), response(payload())])
        await self.api('run01')(model, prompt())
        await self.api('run02')(model, prompt())
        self.assertEqual(model.client.chat.completions.create.await_count, 2)


if __name__ == '__main__':
    unittest.main()
