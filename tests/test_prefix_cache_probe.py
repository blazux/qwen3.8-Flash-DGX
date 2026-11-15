"""Ensure the diagnostic cannot report success on missing reuse or changed probabilities."""
import copy
import hashlib
import importlib.util
import json
import contextlib
import io
import tempfile
import threading
import time
from unittest.mock import patch
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('probe', ROOT / 'scripts/probe-prefix-cache.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def response(cached):
    return {'usage': {'prompt_tokens': 10000, 'completion_tokens': 1, 'prompt_tokens_details': {'cached_tokens': cached}},
            'choices': [{'text': ' example', 'finish_reason': 'length', 'logprobs': {
                'tokens': [' example'], 'token_logprobs': [-0.2],
                'top_logprobs': [{' example': -0.2, ' other': -2.0}]}}]}


class ProbeTest(unittest.TestCase):
    def setUp(self):
        self.responses = {k: response(n) for k, n in [('cold', 0), ('hit', 9984), ('repeat', 9984), ('fresh', 0)]}

    def test_original_prompt_matches_saved_reproducer(self):
        digest = hashlib.sha256(json.dumps(probe.original_prompt()).encode()).hexdigest()
        self.assertEqual(digest, '03726a5b31f980d6acbc1bc980fc4233cf732ea2bd7f78739218c2f54810a5b8')

    def test_matching_cold_hit_and_independent_control_pass(self):
        self.assertTrue(probe.assess(self.responses)['passed'])

    def test_zero_reuse_is_failure_even_with_identical_text(self):
        self.responses['hit'] = response(0)
        self.assertFalse(probe.assess(self.responses)['passed'])

    def test_contaminated_fresh_control_fails(self):
        self.responses['fresh'] = response(9984)
        self.assertFalse(probe.assess(self.responses)['passed'])

    def test_same_first_token_but_changed_later_logprob_fails(self):
        for r in self.responses.values():
            lp = r['choices'][0]['logprobs']
            lp['tokens'].append(' next')
            lp['token_logprobs'].append(-0.3)
            lp['top_logprobs'].append({' next': -0.3})
        self.responses['hit']['choices'][0]['logprobs']['top_logprobs'][1][' next'] = -0.31
        result = probe.assess(self.responses)
        self.assertFalse(result['passed'])
        self.assertTrue(result['comparison_details']['cold_hit']['text'])
        self.assertFalse(result['comparison_details']['cold_hit']['top_logprobs'])

    def test_changed_top_token_set_fails(self):
        self.responses['hit']['choices'][0]['logprobs']['top_logprobs'][0][' new'] = -3.0
        self.assertFalse(probe.assess(self.responses)['passed'])

    def test_missing_usage_or_probabilities_never_pass(self):
        for field in ('usage', 'choices'):
            broken = copy.deepcopy(self.responses)
            del broken['hit'][field]
            with self.assertRaises(KeyError):
                probe.assess(broken)
        self.responses['hit']['choices'][0]['logprobs']['tokens'] = []
        with self.assertRaises(ValueError):
            probe.assess(self.responses)


class WavesTest(unittest.TestCase):
    def fake_transport(self, concurrency, fail_first=False):
        lock = threading.Lock()
        seen, calls = {}, []
        reached = threading.Event()
        state = {'active': 0, 'max_active': 0}

        def post(base, path, body, timeout):
            self.assertEqual(path, '/v1/completions')
            self.assertEqual(timeout, 7)
            self.assertNotIn('stream', body)
            with lock:
                salt = body['cache_salt']
                count = seen.get(salt, 0)
                seen[salt] = count + 1
                index = len(calls)
                calls.append((salt, count))
                state['active'] += 1
                state['max_active'] = max(state['max_active'], state['active'])
                if state['active'] == concurrency:
                    reached.set()
            try:
                if not reached.wait(2):
                    raise AssertionError('Requests did not overlap within a wave')
                time.sleep(.001)
                if fail_first and index == 0:
                    raise RuntimeError('Synthetic transport failure')
                return response(0 if count == 0 else 9984)
            finally:
                with lock:
                    state['active'] -= 1
        return post, calls, state

    def test_concurrent_waves_use_independent_salts_and_real_reuse(self):
        for concurrency in (1, 4, 8):
            with self.subTest(concurrency=concurrency), tempfile.TemporaryDirectory() as directory:
                post, calls, state = self.fake_transport(concurrency)
                output = Path(directory)
                with patch.object(probe, 'post', side_effect=post):
                    result = probe.run_case('http://fixture.invalid', {'prompt': 'synthetic'},
                                            7, concurrency, output, 'case')
                self.assertTrue(result['passed'], result)
                self.assertEqual(len(calls), 4 * concurrency)
                self.assertEqual(state['max_active'], concurrency)
                groups = [calls[i:i + concurrency] for i in range(0, len(calls), concurrency)]
                salts = [{s for s, _ in group} for group in groups]
                self.assertEqual(len(salts[0]), concurrency)
                self.assertEqual(salts[0], salts[1])
                self.assertEqual(salts[1], salts[2])
                self.assertTrue(salts[0].isdisjoint(salts[3]))
                self.assertEqual([[n for _, n in group] for group in groups],
                                 [[n] * concurrency for n in (0, 1, 2, 0)])
                self.assertEqual(len(list(output.glob('*.json'))), 1 + 4 * concurrency)
                for label, wave in result['waves'].items():
                    self.assertEqual(wave['completed_requests'], concurrency)
                    self.assertEqual(wave['completion_tokens'], concurrency)
                    self.assertEqual(wave['errors'], 0)
                    self.assertGreater(wave['aggregate_output_tokens_per_second_e2e'], 0)
                self.assertTrue(all(c['exact_matches']['cold_fresh'] for c in result['clients']))

    def test_one_failed_request_is_archived_and_other_waves_continue(self):
        with tempfile.TemporaryDirectory() as directory:
            post, calls, _ = self.fake_transport(2, fail_first=True)
            output = Path(directory)
            with patch.object(probe, 'post', side_effect=post):
                result = probe.run_case('http://fixture.invalid', {'prompt': 'synthetic'},
                                        7, 2, output, 'case')
            self.assertFalse(result['passed'])
            self.assertEqual(len(calls), 8)
            self.assertEqual(len(list(output.glob('*-cold.json'))), 2)
            records = [json.loads(p.read_text()) for p in output.glob('*-cold.json')]
            self.assertEqual(sum('error' in r for r in records), 1)
            self.assertEqual(result['waves']['cold']['errors'], 1)
            self.assertEqual(result['waves']['cold']['completed_requests'], 1)
            self.assertEqual(result['waves']['cold']['completion_tokens'], 1)
            self.assertEqual(sum(c['passed'] for c in result['clients']), 1)

    def test_invalid_metrics_and_incomplete_finish_cannot_be_successes(self):
        for value in (None, -1, True):
            r = response(0)
            r['usage']['completion_tokens'] = value
            with self.assertRaises(ValueError):
                probe.usage_metrics(r, 1)
        r = response(0)
        r['choices'][0]['finish_reason'] = None
        with self.assertRaises(ValueError):
            probe.signature(r)

    def test_cli_continues_to_next_case_after_a_case_error(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'new'
            argv = ['probe', '--output', str(output), '--tokens', '4']
            with patch('sys.argv', argv), contextlib.redirect_stdout(io.StringIO()), \
                 patch.object(probe, 'post', return_value={'tokens': list(range(10))}), \
                 patch.object(probe, 'run_case', side_effect=[RuntimeError('Synthetic case failure'),
                                                            {'case': 'tokens-4', 'passed': True, 'errors': []}]) as run:
                code = probe.main()
            self.assertEqual(code, 1)
            self.assertEqual(run.call_count, 2)
            summary = json.loads((output / 'summary.json').read_text())
            self.assertFalse(summary['passed'])
            self.assertEqual([c['passed'] for c in summary['cases']], [False, True])
            self.assertTrue((output / 'original-error.json').is_file())

    def test_existing_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            receipt = output / 'summary.json'
            receipt.write_text('previous result')
            with patch('sys.argv', ['probe', '--output', str(output), '--tokens']):
                with self.assertRaises(FileExistsError):
                    probe.main()
            self.assertEqual(receipt.read_text(), 'previous result')


if __name__ == '__main__':
    unittest.main()
