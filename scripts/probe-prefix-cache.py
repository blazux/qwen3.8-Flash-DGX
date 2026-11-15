#!/usr/bin/env python3
"""Cold/hit/fresh controls and end-to-end timings for a direct vLLM API (stdlib)."""
import argparse
import concurrent.futures
import hashlib
import json
import math
import random
import threading
import time
import urllib.request
import uuid
from pathlib import Path

LABELS = ('cold', 'hit', 'repeat', 'fresh')
SIGNATURE_FIELDS = ('text', 'tokens', 'token_logprobs', 'top_logprobs', 'finish_reason')


def original_prompt():
    """Reproduce the synthetic prompt used in the original cache discrepancy."""
    rng = random.Random(2)
    words = 'ledger invoice payroll contract clause annex schedule amount date vendor total net gross tax due paid'.split()
    text = ' '.join(rng.choice(words) + (str(rng.randint(1, 9999)) if rng.random() < 0.2 else '')
                    for _ in range(5800))
    return text + ('\n\nQuestion: list three numbers that appear right after the word '
                   'invoice, then name the most frequent word. Answer:')


def post(base_url, path, body, timeout):
    request = urllib.request.Request(base_url + path, data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def cached_tokens(response):
    usage = response['usage']
    prompt = usage['prompt_tokens']
    count = usage['prompt_tokens_details']['cached_tokens']
    if type(prompt) is not int or prompt < 0 or type(count) is not int or not 0 <= count <= prompt:
        raise ValueError('Invalid prompt/cached token usage')
    return count


def signature(response):
    """Require usable probabilities; compare every reported generated position."""
    choice = response['choices'][0]
    logprobs = choice['logprobs']
    tokens, probs, tops = (logprobs[k] for k in ('tokens', 'token_logprobs', 'top_logprobs'))
    if not tokens or len(tokens) != len(probs) or len(tokens) != len(tops):
        raise ValueError('Missing or inconsistent generated-token logprobs')
    for prob, top in zip(probs, tops):
        if not isinstance(top, dict) or not top:
            raise ValueError('Missing top-token probability distribution')
        if not all(type(v) in (int, float) and math.isfinite(v) for v in [prob, *top.values()]):
            raise ValueError('Missing or non-finite logprob')
    if choice['finish_reason'] not in ('stop', 'length'):
        raise ValueError('Missing or unsuccessful completion finish_reason')
    return choice['text'], tokens, probs, tops, choice['finish_reason']


def assess(responses):
    """A matching answer with zero cache reuse must never count as a pass."""
    counts = {label: cached_tokens(responses[label]) for label in LABELS}
    signatures = {label: signature(responses[label]) for label in LABELS}
    errors = []
    for label in ('cold', 'fresh'):
        if counts[label] != 0:
            errors.append(label + ' control unexpectedly reused a prefix')
    for label in ('hit', 'repeat'):
        if counts[label] <= 0:
            errors.append(label + ' did not reuse any prefix tokens')
    comparisons, details = {}, {}
    for left, right in (('cold', 'hit'), ('hit', 'repeat'), ('cold', 'fresh')):
        key = left + '_' + right
        details[key] = {field: a == b for field, a, b in
                        zip(SIGNATURE_FIELDS, signatures[left], signatures[right])}
        comparisons[key] = all(details[key].values())
        if not comparisons[key]:
            errors.append(left + '/' + right + ' text or full reported logprobs differ')
    return {'passed': not errors, 'cached_tokens': counts, 'exact_matches': comparisons,
            'comparison_details': details, 'errors': errors}


def usage_metrics(response, seconds):
    """Nonstream response latency includes queueing, prefill and generation."""
    usage = response['usage']
    completion = usage['completion_tokens']
    if type(completion) is not int or completion < 0:
        raise ValueError('Invalid completion_tokens usage')
    return {'prompt_tokens': usage['prompt_tokens'], 'completion_tokens': completion,
            'cached_tokens': cached_tokens(response),
            'output_tokens_per_second_e2e': completion / seconds if seconds > 0 else None}


def run_case(base, request, timeout, concurrency, output, name):
    """Each client owns a salt; complete one whole wave before starting the next."""
    salts = [uuid.uuid4().hex for _ in range(concurrency)]
    fresh_salts = [uuid.uuid4().hex for _ in range(concurrency)]
    samples = {i: {} for i in range(concurrency)}
    waves = {}
    (output / (name + '-request.json')).write_text(json.dumps(request, indent=2) + '\n')
    for label in LABELS:
        barrier = threading.Barrier(concurrency)

        def one(client):
            salt = fresh_salts[client] if label == 'fresh' else salts[client]
            record = {'client': client, 'label': label, 'cache_salt': salt}
            barrier.wait(timeout=30)
            started = time.monotonic()
            try:
                response = post(base, '/v1/completions', dict(request, cache_salt=salt), timeout)
                record['response'] = response
                record['seconds'] = time.monotonic() - started
                record['metrics'] = usage_metrics(response, record['seconds'])
            except Exception as error:
                record.setdefault('seconds', time.monotonic() - started)
                record['error'] = f'{type(error).__name__}: {error}'
            suffix = '' if concurrency == 1 else '-client-' + str(client)
            (output / (name + suffix + '-' + label + '.json')).write_text(json.dumps(record, indent=2) + '\n')
            return record, started + record['seconds']

        started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            finished = list(pool.map(one, range(concurrency)))
        # Exclude response-file serialization from the wave's observed request interval.
        elapsed = max(end for _, end in finished) - started
        records = [record for record, _ in finished]
        for record in records:
            samples[record['client']][label] = record
        good = [r for r in records if 'error' not in r]
        tokens = sum(r['metrics']['completion_tokens'] for r in good)
        waves[label] = {'wall_seconds': elapsed, 'completed_requests': len(good),
                        'errors': len(records) - len(good), 'completion_tokens': tokens,
                        'aggregate_output_tokens_per_second_e2e': tokens / elapsed if elapsed > 0 else None,
                        'request_seconds': [r['seconds'] for r in records]}
    clients = []
    for client, records in samples.items():
        result = {'client': client, 'seconds': {k: v['seconds'] for k, v in records.items()},
                  'metrics': {k: v['metrics'] for k, v in records.items() if 'metrics' in v}}
        errors = [label + ': ' + r['error'] for label, r in records.items() if 'error' in r]
        if not errors:
            try:
                result.update(assess({label: r['response'] for label, r in records.items()}))
            except Exception as error:
                errors.append(f'{type(error).__name__}: {error}')
        if errors:
            result.update(passed=False, errors=errors)
        clients.append(result)
    # Preserve the original solo result fields, with additional per-client/wave diagnostics.
    result = dict(clients[0]) if concurrency == 1 else {
        'passed': all(c['passed'] for c in clients),
        'errors': [f"client {c['client']}: {e}" for c in clients for e in c['errors']]}
    result.update(case=name, concurrency=concurrency, clients=clients, waves=waves)
    return result


def positive(value):
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:18300', help='direct vLLM URL, with or without /v1')
    parser.add_argument('--model', default='qwen3.8-flash-next')
    parser.add_argument('--output', type=Path, required=True, help='new results directory (never overwrites)')
    parser.add_argument('--tokens', type=positive, nargs='*', default=[4096, 9984, 10000, 10582, 16384, 20000],
                        help='additional exact token lengths; original prompt is always tested')
    parser.add_argument('--max-tokens', type=positive, default=1)
    parser.add_argument('--concurrency', type=int, choices=(1, 2, 4, 8), default=1,
                        help='independent clients per wave; select lengths that fit the KV pool')
    parser.add_argument('--timeout', type=positive, default=600)
    args = parser.parse_args()
    if len(args.tokens) != len(set(args.tokens)):
        parser.error('--tokens must be unique to preserve each case archive')
    base = args.base_url.rstrip('/')
    if base.endswith('/v1'):
        base = base[:-3]
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {'passed': False, 'cases': [], 'model': args.model, 'base_url': base,
               'max_tokens': args.max_tokens, 'concurrency': args.concurrency,
               'timing_scope': 'Nonstream end-to-end request/wave latency, including queueing, prefill and generation; not TTFT or steady-state decode.',
               'probe_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        prompt = original_prompt()
        cases = [('original', prompt)]
        if args.tokens:
            expanded = prompt * (max(args.tokens) // 10000 + 2)
            ids = post(base, '/tokenize', {'model': args.model, 'prompt': expanded, 'add_special_tokens': False}, args.timeout)['tokens']
            if len(ids) < max(args.tokens):
                raise ValueError('Tokenizer returned too few tokens for requested lengths')
            cases.extend((f'tokens-{n}', ids[:n]) for n in args.tokens)
        for name, content in cases:
            request = {'model': args.model, 'prompt': content, 'max_tokens': args.max_tokens,
                       'temperature': 0, 'seed': 42, 'logprobs': 5}
            try:
                result = run_case(base, request, args.timeout, args.concurrency, args.output, name)
            except Exception as error:
                result = {'case': name, 'concurrency': args.concurrency, 'passed': False,
                          'errors': [f'{type(error).__name__}: {error}']}
                (args.output / (name + '-error.json')).write_text(json.dumps(result, indent=2) + '\n')
            summary['cases'].append(result)
            # Keep an aggregate even if a later case is interrupted.
            (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
            print(json.dumps(result), flush=True)
        summary['passed'] = all(c['passed'] for c in summary['cases'])
    except Exception as error:
        summary['error'] = f'{type(error).__name__}: {error}'
        print(summary['error'], flush=True)
    finally:
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
