"""Apply the opt-in checkpoint-boundary workaround only to the inspected source."""
import ast
import hashlib
import json
import sys
from pathlib import Path


def apply(path):
    manifest = json.loads(Path(__file__).with_name('manifest.json').read_text())
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest == manifest['patched_sha256']:
        print('Already patched')
        return
    if digest != manifest['original_sha256']:
        raise SystemExit('Source differs from the inspected v0.33 file; refusing to patch')
    old = '    alignment = 1\n    for layer in get_layers_from_vllm_config('
    new = '''    # Local diagnostic opt-out: use ordinary boundary-split prefills in every
    # Mamba group, keeping scheduler, allocation, and checkpoint export in sync.
    if os.environ.get("VLLM_DGX_SPLIT_PREFILL") == "1":
        return None
    alignment = 1
    for layer in get_layers_from_vllm_config('''
    text = source.decode()
    if text.count(old) != 1:
        raise SystemExit("Expected one capability-query match; refusing to patch")
    patched = text.replace('from abc import abstractmethod\n', 'import os\nfrom abc import abstractmethod\n', 1).replace(old,new)
    ast.parse(patched)
    if hashlib.sha256(patched.encode()).hexdigest() != manifest['patched_sha256']:
        raise SystemExit('Patched source hash mismatch; original left untouched')
    path.write_text(patched)
    print('Patched; enabled only when VLLM_DGX_SPLIT_PREFILL=1')


if __name__ == '__main__':
    apply(Path(sys.argv[1]))
