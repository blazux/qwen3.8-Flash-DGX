#!/usr/bin/env python3
"""dflash/DFlash2: open the Qwen4Exp drafter seam in the vLLM v0.32.0 image.

Measured 2026-10-27 on the GB10: with a DFlash2 draft configured,
method=dflash parses and vLLM resolves the DFlash2DraftModel class, then
config.py refuses the whole architecture
("Qwen4Exp speculative decoding supports only its native MTP checkpoint and
linear n-gram proposers"). The blockers are catalogued upstream with patches at
vllm-project/vllm#56088 (working recipe: PixelML/deepspec-qwen38-flash-next/
serving). Against THIS pinned 0.32.0, three of the five are already moot:

  * blocker 1 (V1 runner cannot prepare PLE inputs): this image boots the
    target on the V2 model runner with no override (the everyday MTP runs prove
    it), and scripts/serve.sh pins VLLM_USE_V2_MODEL_RUNNER=1 for the draft path;
  * blocker 2 (None MTP hidden-state getter): since fixed upstream - the V2
    runner now calls get_drafter_hidden_states(), which treats a None buffer as
    "use the regular hidden states" (verified at v0.32.0, spec_decode/utils.py);
  * blocker 3 (sample_from_anchor raise): DFlash2's speculators-format config
    carries no sample_from_anchor, so the stock 1+N query layout applies and the
    raise is not reached. The DeepSpec-only overlay/plugin files of the working
    recipe are therefore NOT ported.
  * blocker 5 (adaptive verification vs GDN backends): a dspark option
    (config/speculative.py gates it on method=="dspark"); dflash never asks.

What remains is what this script does:

  (a) config.py     - admit "dflash"/"dspark" next to mtp/ngram in the Qwen4Exp
                      speculative-method allowlist (upstream blocker 4, their
                      patches/03 - its anchor text matches 0.32.0 byte-for-byte);
  (b) nvidia/model.py - the target-side auxiliary hidden states that every
                      DFlash-family drafter conditions on (upstream
                      "not a bug but is needed"). The stock runner wires them
                      through the EAGLE-3 protocol:
                      spec_decode/eagle/eagle3_utils.set_eagle3_aux_hidden_
                      state_layers(model, spec) requires supports_eagle3(model)
                      and an EagleModelMixin decoder, stores the tap layers, and
                      expects forward to return (hidden, [tap, ...]). Qwen 3.5
                      ships exactly that for its own classes; Qwen4Exp declares
                      neither. So: Qwen4ExpModel gains the mixin, both top
                      classes gain the protocol members (the protocol's default
                      unwrap methods, assigned at the module tail), and the
                      decoder forward grows the capture itself. Tap semantics
                      are incoai's own DFlash2 recipe (the draft repo's
                      qwen38-dflash2-capture.patch): COMPLETED layer outputs
                      averaged over the hc_count HyperConnection streams. The
                      helper reads dflash_config.target_layer_ids with +1
                      ("entry of layer L+1"), which in this loop is exactly the
                      materialized (combined) output of layer L - taken via the
                      pure mlp_hyper_connection.combine the deepstack/PP paths
                      already use, leaving the pending tuple for the next layer
                      untouched, so an unconfigured boot (method=mtp etc.) is
                      byte-identical: the guard tuple is empty and no combine
                      call is issued.
  (c) common/qsa_cache.py - the ring widening of vllm#54912, verbatim port
                      (open upstream, needed for num_speculative_tokens >= 5):
                      a K whose minimal whole-group ring does not divide the
                      attention block size used to assert-killed the boot on
                      GB10's 848/1616 blocks; it now widens within a bounded
                      factor. Same returned capacity whenever the old assert
                      passed, so the MTP path is untouched.

    python3 patch_dflash_qwen4exp.py /usr/local/lib/python3.12/dist-packages
"""
import ast
import sys

SP = sys.argv[1] if len(sys.argv) > 1 else "/usr/local/lib/python3.12/dist-packages"
CFG = f"{SP}/vllm/model_executor/models/config.py"
MDL = f"{SP}/vllm/models/qwen4_exp/nvidia/model.py"
QSA = f"{SP}/vllm/models/qwen4_exp/common/qsa_cache.py"


def edit(path, old, new, why):
    s = open(path).read()
    if new in s:
        print(f"  already applied: {why}")
        return
    n = s.count(old)
    assert n == 1, f"{path}: anchor for {why!r} found {n} times, expected 1:\n{old}"
    open(path, "w").write(s.replace(old, new))
    print(f"  applied: {why}")


# --- (a) config.py: admit dflash/dspark to the Qwen4Exp speculative allowlist ---
edit(
    CFG,
    """        if spec_config is not None and spec_config.method not in {
            "mtp",
            "ngram",
            "ngram_gpu",
        }:
            raise NotImplementedError(
                "Qwen4Exp speculative decoding supports only its native MTP "
                "checkpoint and linear n-gram proposers"
            )""",
    """        if spec_config is not None and spec_config.method not in {
            "mtp",
            # qwen38-flash-dgx (vllm#56088): drafters paired with this target;
            # the aux-hidden-state seam in nvidia/model.py supplies their
            # conditioning and the V2 runner is pinned by scripts/serve.sh.
            "dflash",
            "dspark",
            "ngram",
            "ngram_gpu",
        }:
            raise NotImplementedError(
                "Qwen4Exp speculative decoding supports its native MTP "
                "checkpoint, paired DFlash/DSpark drafters, and linear "
                "n-gram proposers"
            )""",
    "Qwen4Exp speculative-method allowlist (dflash, dspark)",
)

# --- (b) nvidia/model.py: the EAGLE-3 style aux hidden-state seam ---
# 0.1 the mixin import (needed where the decoder class is defined)
edit(
    MDL,
    "from vllm.model_executor.models.interfaces import (\n    HasInnerState,\n",
    "from vllm.model_executor.models.interfaces import (\n    EagleModelMixin,\n    HasInnerState,\n",
    "model.py: EagleModelMixin import",
)
# 0.2 the Protocol import for the module-tail mount
edit(
    MDL,
    "from vllm.model_executor.models.qwen3_5 import (\n",
    "from vllm.model_executor.models.interfaces import SupportsEagle3  # qwen38: dflash seam\nfrom vllm.model_executor.models.qwen3_5 import (\n",
    "model.py: SupportsEagle3 import",
)
# 1. the decoder must BE an EagleModelMixin (eagle3_utils asserts isinstance;
#    the mixin also brings the aux_hidden_state_layers=() default and setter)
edit(
    MDL,
    "class Qwen4ExpModel(nn.Module):",
    "class Qwen4ExpModel(nn.Module, EagleModelMixin):  # qwen38: dflash seam",
    "Qwen4ExpModel inherits EagleModelMixin",
)
# 2. collect the per-tap list in the decoder forward
edit(
    MDL,
    """        full_num_tokens = positions.shape[-1]
        block_output = None""",
    """        full_num_tokens = positions.shape[-1]
        aux_hidden_states: list[torch.Tensor] = []  # qwen38: dflash taps below
        block_output = None""",
    "Qwen4ExpModel.forward: aux_hidden_states collection list",
)
# 3. capture after every completed layer whose NEXT index was configured
edit(
    MDL,
    """                ngram_context=ngram_context,
            )
            if deepstack_input_embeds is not None and layer_idx < len(""",
    """                ngram_context=ngram_context,
            )
            if layer_idx + 1 in self.aux_hidden_state_layers:
                # qwen38-flash-dgx dflash tap (vllm#56088): the COMPLETED output
                # of layer layer_idx - the +1-shifted ids the eagle3 helper
                # passes are its NEXT layer's entry. Materialized by the same
                # pure HC combine the deepstack/PP paths use (the pending tuple
                # the loop threads on is left exactly as it was), then averaged
                # over the hc_count streams: incoai's capture recipe from the
                # draft repo's qwen38-dflash2-capture.patch.
                _hc_done = layer.mlp_hyper_connection.combine(
                    hidden_states, block_output, injection
                )
                aux_hidden_states.append(
                    _hc_done.unflatten(
                        -1, (self.config.hc_count, self.config.hidden_size)
                    ).mean(-2)
                )
            if deepstack_input_embeds is not None and layer_idx < len(""",
    "Qwen4ExpModel.forward: completed-layer HC-mean capture",
)
# 4. hand the list out alongside the sampled stream (only ever reached when
#    taps are configured, so the untouched MTP path keeps returning one tensor)
edit(
    MDL,
    """            self._mtp_hidden_buffer[:num_tokens].copy_(multi_hidden)
        return sample_hidden_states""",
    """            self._mtp_hidden_buffer[:num_tokens].copy_(multi_hidden)
        if self.aux_hidden_state_layers:
            if len(aux_hidden_states) != len(self.aux_hidden_state_layers):
                raise RuntimeError(
                    f"dflash capture: collected {len(aux_hidden_states)} of "
                    f"{len(self.aux_hidden_state_layers)} taps (a configured "
                    "layer was skipped - PP stage out of range?)"
                )
            return (sample_hidden_states, aux_hidden_states)
        return sample_hidden_states""",
    "Qwen4ExpModel.forward: (hidden, aux) return under capture",
)
# 5. module tail: mount the protocol members the runner helper checks for. It
#    never touches weights or the MTP path; has_own_* stay False (they are read
#    off the DRAFT for embed/lm_head aliasing, and DFlash2 ships neither).
TAIL = '''

# --- qwen38-flash-dgx: dflash target seam (vllm#56088) -----------------------
# spec_decode/eagle/eagle3_utils.set_eagle3_aux_hidden_state_layers refuses the
# drafter boot unless supports_eagle3(model); its default methods unwrap
# self[.get_language_model()].model to the decoder (an EagleModelMixin since
# the base edit above) and store the tap layers there. The capture itself lives
# in Qwen4ExpModel.forward. Inert unless a dflash/dspark/eagle3 drafter asks.
for _q4e_cls in (Qwen4ExpForCausalLM, Qwen4ExpForConditionalGeneration):
    _q4e_cls.supports_eagle3 = True
    _q4e_cls.has_own_lm_head = False
    _q4e_cls.has_own_embed_tokens = False
    _q4e_cls.set_aux_hidden_state_layers = SupportsEagle3.set_aux_hidden_state_layers
    _q4e_cls.get_eagle3_default_aux_hidden_state_layers = (
        SupportsEagle3.get_eagle3_default_aux_hidden_state_layers
    )
del _q4e_cls
'''
s = open(MDL).read()
if "qwen38-flash-dgx: dflash target seam" not in s:
    assert s.rstrip().endswith("]"), "model.py no longer ends with __all__"
    open(MDL, "w").write(s.rstrip("\n") + "\n" + TAIL)
    print("  applied: model.py dflash seam tail")
else:
    print("  already applied: model.py dflash seam tail")

# --- (c) common/qsa_cache.py: ring widening for depths >= 5 (vllm#54912) ---
edit(
    QSA,
    "from vllm.config.cache import CacheDType\n",
    "from vllm.config.cache import CacheDType\nfrom vllm.logger import init_logger\n",
    "qsa_cache logger import",
)
edit(
    QSA,
    "\n\n\ndef canonical_qsa_rope_positions(positions: torch.Tensor) -> torch.Tensor:",
    "\n\n\nlogger = init_logger(__name__)\n\n\ndef canonical_qsa_rope_positions(positions: torch.Tensor) -> torch.Tensor:",
    "qsa_cache module logger",
)
edit(
    QSA,
    "\n\nclass QSAKeyStateCache(_QSAStateCache):",
    '''

# The ring may be widened past the minimal whole-group size by at most this
# factor. Widening keeps the scheduler LCM unchanged, but every request holds
# one ring block for its lifetime, so a large divisor must not be picked
# silently (e.g. 20 rows -> 212 on a block size of 848 = 16 * 53).
# Port of vllm-project/vllm#54912 (open upstream): speculation depths whose
# minimal ring does not divide the attention block size used to assert there.
QSA_RING_MAX_WIDENING = 2


def qsa_ring_capacity(
    compress_ratio: int, num_speculative_tokens: int, block_size: int
) -> int:
    """Rows of the QSA raw-key ring for a given attention block size.

    The ring holds the open group's committed keys plus every row a
    speculative step stores before acceptance is known (``span``). Anything
    narrower lets a rejected draft row overwrite a committed key the next
    step needs to close the group, so ``span`` is a lower bound; a wider ring
    is slack. The ring is rounded up to whole groups, and a whole-group size
    that divides ``block_size`` is preferred so that adding the circular-buffer
    group does not increase the scheduler block size (the LCM over all
    groups). When the minimal size does not divide ``block_size``, the next
    multiple of ``compress_ratio`` that does is used, up to
    ``QSA_RING_MAX_WIDENING`` times the minimal size; beyond that a
    ``ValueError`` is raised rather than allocating a disproportionate ring.
    """
    span = compress_ratio + num_speculative_tokens
    minimal = compress_ratio * cdiv(span, compress_ratio)
    limit = min(block_size, QSA_RING_MAX_WIDENING * minimal)
    capacity = minimal
    while capacity <= limit:
        if block_size % capacity == 0:
            if capacity != minimal:
                logger.info_once(
                    "QSA ring widened from %d to %d rows so that it divides the "
                    "attention block size %d (num_speculative_tokens=%d).",
                    minimal,
                    capacity,
                    block_size,
                    num_speculative_tokens,
                )
            return capacity
        capacity += compress_ratio
    raise ValueError(
        f"QSA ring needs {minimal} rows for num_speculative_tokens="
        f"{num_speculative_tokens} (compress ratio {compress_ratio}), and no "
        f"multiple of {compress_ratio} up to {limit} divides the attention "
        f"block size {block_size}; choose a depth whose ring divides it"
    )


class QSAKeyStateCache(_QSAStateCache):''',
    "qsa_cache qsa_ring_capacity",
)
edit(
    QSA,
    """        # Hold the open group's committed keys plus every row a speculative
        # step stores before acceptance is known, rounded up to whole groups so
        # the ring divides the attention block size (it joins the LCM that sets
        # the scheduler block size). Anything narrower lets a rejected draft row
        # overwrite a committed key the next step needs to close the group.
        span = self.compress_ratio + vllm_config.num_speculative_tokens
        capacity = self.compress_ratio * cdiv(span, self.compress_ratio)
        assert self.cache_config.block_size % capacity == 0, (
            f"QSA ring capacity {capacity} must divide the attention block "
            f"size {self.cache_config.block_size}"
        )""",
    """        # qwen38-flash-dgx (vllm#54912): the minimal whole-group ring when it
        # divides the attention block size, else the smallest widening that
        # does. Same capacity whenever the old assert accepted.
        capacity = qsa_ring_capacity(
            self.compress_ratio,
            vllm_config.num_speculative_tokens,
            self.cache_config.block_size,
        )""",
    "qsa_cache ring widening in get_kv_cache_spec",
)

for path in (CFG, MDL, QSA):
    ast.parse(open(path).read(), path)
print("patch_dflash_qwen4exp: all files parse OK")
