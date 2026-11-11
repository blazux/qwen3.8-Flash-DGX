import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def summarize(name):
    rows = [json.loads(line) for line in (ROOT / name).read_text().splitlines()]
    result = {}
    for words in (1600, 6400):
        for phase in ('fresh', 'repeat'):
            samples = [r for r in rows if r['kind'] == 'sample'
                       and r['words'] == words and r['phase'] == phase]
            if not samples:
                continue
            result[f'{words}/{phase}'] = {
                'n': len(samples),
                'prompt_tokens_median': statistics.median(r['usage']['prompt_tokens'] for r in samples),
                'ttft_s_median': statistics.median(r['ttft_s'] for r in samples),
                'decode_tokens_per_s_median': statistics.median(r['decode_tokens_per_s'] for r in samples),
                'e2e_output_tokens_per_s_median': statistics.median(r['usage']['completion_tokens'] / r['elapsed_s'] for r in samples),
                'prompt_tokens_over_ttft_median': statistics.median(r['usage']['prompt_tokens'] / r['ttft_s'] for r in samples),
                'cached_tokens': [r['usage'].get('prompt_tokens_details', {}).get('cached_tokens', 0) for r in samples],
                'output_tokens': [r['usage']['completion_tokens'] for r in samples],
                'finish_reasons': [r.get('finish_reason') for r in samples],
            }
    concurrent = [r['tokens_per_s'] for r in rows if r['kind'] == 'concurrent']
    result['concurrent/4'] = {'n': len(concurrent), 'aggregate_tokens_per_s': concurrent,
                              'median': statistics.median(concurrent) if concurrent else None}
    result['determinism'] = [r['identical'] for r in rows if r['kind'] == 'determinism']
    result['finished'] = any(r['kind'] == 'finished' for r in rows)
    result['errors'] = [r for r in rows if r['kind'] == 'error']
    result['sample_count'] = sum(r['kind'] == 'sample' for r in rows)
    return result


results = {name: summarize(name + '.jsonl') for name in ('before', 'after')
           if (ROOT / (name + '.jsonl')).exists()}
(ROOT / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
print(json.dumps(results, indent=2))
