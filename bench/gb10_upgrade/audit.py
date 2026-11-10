#!/usr/bin/env python3
"""Audit actual coverage and paired identities before reporting medians."""
import argparse
import collections
import hashlib
import itertools
import json
from pathlib import Path
import statistics

LABELS = ('v031-current', 'v033-matched', 'v033-fast')
EXPECTED = set(itertools.product((1024, 8192, 32768), (1, 4, 8), range(2), ('fresh', 'repeat')))

def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []

def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None

def arm(folder):
    matrix = rows(folder / 'matrix.jsonl')
    waves = [r for r in matrix if r.get('kind') == 'wave']
    requests = [r for r in matrix if r.get('kind') == 'request']
    keys = [(r['target_input_tokens'], r['concurrency'], r['repetition'], r['phase']) for r in waves]
    counts = collections.Counter(keys)
    request_keys = [(r['target_input_tokens'], r['concurrency'], r['repetition'], r['phase'], r['client']) for r in requests]
    expected_requests = {(*cell, client) for cell in EXPECTED for client in range(cell[1])}
    cells = []
    for target, concurrency, phase in itertools.product((1024, 8192, 32768), (1, 4, 8), ('fresh', 'repeat')):
        samples = [r for r in waves if (r['target_input_tokens'], r['concurrency'], r['phase']) == (target, concurrency, phase)]
        rr = [r for r in requests if (r['target_input_tokens'], r['concurrency'], r['phase']) == (target, concurrency, phase)]
        good = [r for r in rr if r['ok']]
        cells.append(dict(context_target=target, clients=concurrency, phase=phase,
            repetitions=len(samples), requests=len(rr), successful=len(good),
            aggregate_output_tok_s=median(r['output_tokens_per_s'] for r in samples),
            wave_ttft_p50_s=median(r['ttft_p50_s'] for r in samples),
            client_decode_tok_s=median(r.get('decode_tokens_per_s') for r in good),
            cached_tokens=sum(r['usage'].get('prompt_tokens_details', {}).get('cached_tokens', 0) or 0 for r in good),
            prompt_tokens=sum(r['usage']['prompt_tokens'] for r in good),
            output_tokens=sum(r['usage']['completion_tokens'] for r in good)))
    needles = [r for r in rows(folder / 'needles.jsonl') if r.get('kind') == 'needle']
    repair = [r for r in rows(folder / 'repair.jsonl') if r.get('kind') == 'repair']
    tools = rows(folder / 'tool-json.jsonl')
    soak = rows(folder / 'soak.jsonl')
    soak_rows = [r for r in soak if r.get('kind') == 'request']
    complete = [r for r in soak if r.get('kind') == 'soak_complete']
    generation = rows(folder / 'humaneval/raw.jsonl')
    evaluation_file = folder / 'humaneval/evaluation/evaluation.json'
    evaluation = json.loads(evaluation_file.read_text()) if evaluation_file.exists() else {}
    evaluated = evaluation.get('eval', {})
    quality = dict(generated=len(generation), errors=sum(not r['ok'] for r in generation),
        missing_answers=sum(not r.get('content', '').strip() for r in generation),
        truncations=sum(r.get('finish_reason') == 'length' for r in generation),
        evaluated=len(evaluated),
        base_pass=sum(r[0]['base_status'] == 'pass' for r in evaluated.values()),
        plus_pass=sum(r[0]['base_status'] == r[0]['plus_status'] == 'pass' for r in evaluated.values()),
        complete=set(evaluated) == {f'HumanEval/{i}' for i in range(164)} and all(len(v) == 1 for v in evaluated.values())
            and collections.Counter(r['task_id'] for r in generation) == collections.Counter({f'HumanEval/{i}':1 for i in range(164)}))
    return dict(matrix=dict(waves=len(waves), requests=len(requests),
        errors=sum(not r['ok'] for r in requests),
        complete=set(keys) == EXPECTED and all(v == 1 for v in counts.values()) and len(requests) == 156
            and set(request_keys) == expected_requests,
        missing_cells=[list(k) for k in sorted(EXPECTED-set(keys))], cells=cells),
        tools=dict(passed=sum(r.get('passed', False) for r in tools), total=len(tools),
            complete=len(tools) == 6 and {(r['case'],r['repetition']) for r in tools} == set(itertools.product(('tool','json'), range(3)))),
        repair=dict(passed=sum(r['passed'] for r in repair), total=len(repair),
            complete=len(repair) == 6 and {(r['task'],r['repetition']) for r in repair} ==
                {(name,0) for name in ('atomic-ledger','deep-config','retry-boundary','stable-dag','ttl-cache','window-split')},
            attempts=[{k:r.get(k) for k in ('task','repetition','passed','elapsed_s','tool_calls','tool_errors','error')} for r in repair]),
        needles=dict(passed=sum(r['passed'] for r in needles), total=len(needles),
            complete=len(needles) == 9 and {(r['target_input_tokens'],r['position']) for r in needles} == set(itertools.product((32768,131072,262144),(.1,.5,.9)))),
        soak=dict(complete=bool(complete) and complete[-1]['elapsed_s'] >= 600, seconds=complete[-1]['elapsed_s'] if complete else None,
            requests=len(soak_rows), errors=sum(not r['ok'] for r in soak_rows),
            deliberate_disconnects=sum(r.get('kind') == 'client_disconnect' for r in soak)),
        humaneval=quality)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent / 'results')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    result = {label:arm(a.root / label) for label in LABELS}
    pairs = {}
    def identities(label, filename, kind):
        data = rows(a.root / label / filename)
        return {((r['target_input_tokens'],r['concurrency'],r['repetition'],r['phase'],r['client']) if kind == 'request' else r['task_id']):r['prompt_sha256']
                for r in data if kind == 'quality' or r.get('kind') == kind}
    for label in LABELS[1:]:
        pairs[label] = {}
        for filename, kind in (('matrix.jsonl', 'request'), ('humaneval/raw.jsonl', 'quality')):
            reference = identities(LABELS[0], filename, kind)
            candidate = identities(label, filename, kind)
            pairs[label][kind] = dict(reference_count=len(reference), candidate_count=len(candidate),
                identical=bool(reference) and reference == candidate)
        def outcomes(name):
            f = a.root / name / 'humaneval/evaluation/evaluation.json'
            data = json.loads(f.read_text()).get('eval', {}) if f.exists() else {}
            return {k:(v[0]['base_status'] == 'pass',
                       v[0]['base_status'] == v[0]['plus_status'] == 'pass') for k,v in data.items()}
        reference = outcomes(LABELS[0])
        candidate = outcomes(label)
        common = sorted(reference.keys() & candidate.keys(), key=lambda k:int(k.split('/')[-1]))
        pairs[label]['outcome_changes'] = {
            name:dict(gained=[k for k in common if not reference[k][index] and candidate[k][index]],
                      lost=[k for k in common if reference[k][index] and not candidate[k][index]])
            for index,name in enumerate(('base','plus'))}
    result['paired_prompts'] = pairs
    if a.output:
        a.output.write_text(json.dumps(result, indent=2) + '\n')
    for label in LABELS:
        r = result[label]
        print(label, 'matrix',r['matrix']['requests'], 'errors',r['matrix']['errors'],
              'complete',r['matrix']['complete'], 'tools',r['tools'],
              'repairs',r['repair']['passed'], '/',r['repair']['total'],
              'needles',r['needles'], 'soak',r['soak'], 'HumanEval',r['humaneval'])
    print('Paired prompts:',pairs)

if __name__ == '__main__':
    main()
