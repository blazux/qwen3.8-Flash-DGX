#!/usr/bin/env python3
"""Execute the predeclared paired suite against one already running deployment."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser()
    p.add_argument('label')
    p.add_argument('--container', default='qwen38-flash')
    p.add_argument('--soak-seconds', type=int, default=600)
    a = p.parse_args()
    out = ROOT / 'results' / a.label
    out.mkdir(exist_ok=False)
    journal = out / 'stages.jsonl'
    def emit(**row):
        row['utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with journal.open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
    def run(name, script, arguments, limit):
        command = [sys.executable, str(ROOT / 'harness' / script),
                   '--label', a.label, '--output', str(out / (name + '.jsonl')), *arguments]
        emit(event='start', stage=name, command=command, timeout_s=limit)
        with (out / (name + '.log')).open('x') as log:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            try:
                rc = proc.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                emit(event='timeout', stage=name)
                return False
        emit(event='end', stage=name, returncode=rc)
        return rc == 0
    common = ['--container', a.container, '--seed', '20261110', '--request-timeout', '420']
    stages = []
    stages.append(run('matrix', 'bench_matrix.py', ['matrix', *common,
        '--contexts', '1024,8192,32768', '--concurrencies', '1,4,8',
        '--repetitions', '2', '--output-tokens', '128', '--seconds', '1800'], 2250))
    stages.append(run('tool-json', 'bench_tool_json.py', [], 360))
    stages.append(run('repair', 'bench_swe_micro.py', ['--repetitions', '1',
        '--concurrency', '2', '--seed', '20261110'], 900))
    stages.append(run('needles', 'bench_matrix.py', ['needles', *common,
        '--contexts', '32768,131072,262144'], 1200))
    stages.append(run('soak', 'bench_matrix.py', ['soak', *common,
        '--contexts', '1024,8192', '--concurrencies', '1,4,8',
        '--output-tokens', '128', '--seconds', str(a.soak_seconds)], a.soak_seconds + 500))
    emit(event='suite_finished', stages_successful=stages,
         note='Stage exit codes do not establish coverage or quality; inspect rows.')
    return 0 if all(stages) else 1

if __name__ == '__main__':
    raise SystemExit(main())
