#!/usr/bin/env python3
"""CPU checks for src/patch_dflash_qwen4exp.py (patch 23) - no GPU, no vllm import.

    python3 src/test_dflash_patch_cpu.py

Fixtures are syntactically valid windows of the v0.32.0 files around every
anchor the patch edits (anchors copied byte-exact from
https://github.com/vllm-project/vllm/tree/v0.32.0). Checks:
  1. the patcher runs clean and idempotent over a fake dist-packages tree;
  2. every edit landed in the right file with the right shape;
  3. a drifted anchor refuses loudly instead of half-patching;
  4. qsa_ring_capacity keeps the MTP depths byte-familiar (same capacity the
     old assert accepted) and rescues exactly the depths the assert killed
     (the PR #54912 test table);
  5. the capture/guard lines only ever fire when aux layers are configured.
"""
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PATCH = os.path.join(HERE, "patch_dflash_qwen4exp.py")

CFG_FIXTURE = '''\
class Qwen4ExpForConditionalGenerationConfig:
    @staticmethod
    def verify_and_update_config(vllm_config):
        spec_config = vllm_config.speculative_config
        if spec_config is not None and spec_config.method not in {
            "mtp",
            "ngram",
            "ngram_gpu",
        }:
            raise NotImplementedError(
                "Qwen4Exp speculative decoding supports only its native MTP "
                "checkpoint and linear n-gram proposers"
            )
'''

MDL_FIXTURE = '''\
import torch
from vllm.model_executor.models.interfaces import (
    HasInnerState,
)
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
)


class Qwen4ExpModel(nn.Module):
    def forward(self, positions, input_ids, query_start_loc, ngram_context,
                deepstack_input_embeds):
        hidden_states = positions
        full_num_tokens = positions.shape[-1]
        block_output = None
        injection = None
        for layer_idx, layer in enumerate(()):
            hidden_states, block_output, injection = layer(
                hidden_states=hidden_states,
                positions=positions,
                input_ids=input_ids,
                query_start_loc=query_start_loc,
                ngram_context=ngram_context,
            )
            if deepstack_input_embeds is not None and layer_idx < len(
                deepstack_input_embeds
            ):
                hidden_states = hidden_states + deepstack_input_embeds
        sample_hidden_states = hidden_states
        multi_hidden = hidden_states
        num_tokens = 1
        self._mtp_hidden_buffer = None
        if self._mtp_hidden_buffer is not None:
            self._mtp_hidden_buffer[:num_tokens].copy_(multi_hidden)
        return sample_hidden_states


class Qwen4ExpForCausalLM(nn.Module):
    pass


class Qwen4ExpForConditionalGeneration(nn.Module):
    pass


__all__ = [
    "Qwen4ExpModel",
]
'''

QSA_FIXTURE = '''\
import torch
from vllm.config import CacheConfig, VllmConfig
from vllm.config.cache import CacheDType
from vllm.utils.math_utils import cdiv


def canonical_qsa_rope_positions(positions: torch.Tensor) -> torch.Tensor:
    return positions


class _QSAStateCache:
    pass


class QSAKeyStateCache(_QSAStateCache):
    def get_kv_cache_spec(self, vllm_config):
        # Hold the open group's committed keys plus every row a speculative
        # step stores before acceptance is known, rounded up to whole groups so
        # the ring divides the attention block size (it joins the LCM that sets
        # the scheduler block size). Anything narrower lets a rejected draft row
        # overwrite a committed key the next step needs to close the group.
        span = self.compress_ratio + vllm_config.num_speculative_tokens
        capacity = self.compress_ratio * cdiv(span, self.compress_ratio)
        assert self.cache_config.block_size % capacity == 0, (
            f"QSA ring capacity {capacity} must divide the attention block "
            f"size {self.cache_config.block_size}"
        )
        return capacity
'''


def make_tree(root, mdl=MDL_FIXTURE, cfg=CFG_FIXTURE, qsa=QSA_FIXTURE):
    for sub, name, body in (
        ("vllm/model_executor/models", "config.py", cfg),
        ("vllm/models/qwen4_exp/nvidia", "model.py", mdl),
        ("vllm/models/qwen4_exp/common", "qsa_cache.py", qsa),
    ):
        if body is None:
            continue
        d = os.path.join(root, sub.replace("/", os.sep))
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, name), "w").write(body)


def run(root):
    p = subprocess.run([sys.executable, PATCH, root], capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        # 1. apply + idempotence on real anchors
        tree = os.path.join(td, "a")
        os.makedirs(tree)
        make_tree(tree)
        rc, out = run(tree)
        assert rc == 0, out
        assert out.count("applied: ") == 12, out  # everything new, nothing skipped
        rc2, out2 = run(tree)
        assert rc2 == 0 and out2.count("already applied: ") == 12, out2
        cfg = open(os.path.join(tree, "vllm/model_executor/models/config.py")).read()
        mdl = open(os.path.join(tree, "vllm/models/qwen4_exp/nvidia/model.py")).read()
        qsa = open(os.path.join(tree, "vllm/models/qwen4_exp/common/qsa_cache.py")).read()
        # 2. shapes
        assert '"dflash",\n            "dspark",\n            "ngram",' in cfg, cfg
        assert "class Qwen4ExpModel(nn.Module, EagleModelMixin):" in mdl
        assert "EagleModelMixin,\n    HasInnerState," in mdl
        assert "import SupportsEagle3" in mdl
        assert "if layer_idx + 1 in self.aux_hidden_state_layers:" in mdl
        assert "return (sample_hidden_states, aux_hidden_states)" in mdl
        assert "_q4e_cls.set_aux_hidden_state_layers = SupportsEagle3.set_aux_hidden_state_layers" in mdl
        assert "def qsa_ring_capacity(" in qsa and "assert self.cache_config.block_size % capacity" not in qsa
        print("1-2 OK: apply/idempotence/shapes")

        # 3. drift on ANY anchor => loud failure, and the file keeps BOTH
        #    old-shape and no partial mix inside the one drifted file.
        tree2 = os.path.join(td, "b")
        os.makedirs(tree2)
        drifted = CFG_FIXTURE.replace('"ngram_gpu",\n        }:', '"ngram_gpu",\n        }:\n        pass  # drifted\n')
        make_tree(tree2, cfg=drifted)
        rc3, out3 = run(tree2)
        assert rc3 != 0 and "anchor for" in out3, out3
        kept = open(os.path.join(tree2, "vllm/model_executor/models/config.py")).read()
        assert '"dflash"' not in kept  # no half-patch
        print("3 OK: drift refused")

        # 4. ring math (PR #54912 expectations incl. GB10 block sizes)
        import logging
        logging.Logger.info_once = lambda self, *a: None  # vllm logger extension
        body = re.search(r"QSA_RING_MAX_WIDENING = 2.*?class QSAKeyStateCache", qsa, re.S).group(0)
        ns = {"logger": logging.getLogger("qsa")}
        exec("def cdiv(a, b):\n    return -(-a // b)\n" + body[: body.index("class QSA")], ns)
        f = ns["qsa_ring_capacity"]
        for k, expect in ((1, 8), (2, 8), (3, 8), (4, 8)):  # MTP-era depths on 848
            assert f(4, k, 848) == expect, (k, f(4, k, 848))
            assert f(4, k, 1616) == expect, (k, f(4, k, 1616))
        assert f(4, 5, 848) == 16 and f(4, 5, 16) == 16 and f(4, 8, 1616) == 16
        assert f(4, 7, 1616) == 16  # the DRAFT_TOKENS default, rescued
        for bs in (16, 848, 1616):
            try:
                f(4, 13, bs)
            except ValueError as e:
                assert "QSA ring" in str(e)
            else:
                raise AssertionError(f"13/4-{bs} must raise")
        print("4 OK: ring math (MTP untouched, K=7 rescued, refusal bound kept)")

        # 5. inert-unless-configured: capture guard + return guard read the
        #    (empty default) layers tuple; no combine call without taps
        assert "if self.aux_hidden_state_layers:" in mdl
        assert mdl.count("mlp_hyper_connection.combine") == 1  # only inside the tap guard
        print("5 OK: inert for the MTP path")
    print("dflash patch CPU checks: ALL OK")


if __name__ == "__main__":
    main()
