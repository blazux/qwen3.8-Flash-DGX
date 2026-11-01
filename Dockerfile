# Qwen3.8-Flash-Next on a DGX Spark — the official vLLM v0.32.0 image plus these patches:
#   1 PLE mmap, 2 GB10 FLA fixes, 4 prefix-caching block_size fix, 5 exact top-k, 6 hybrid mode,
#   8 deterministic top-k kernel, 10 reduced draft vocabulary, 12 Qwen tool preamble fix,
#   13 quoted Qwen tool markers stay text, 14 MoE weight-load clone (#33), 15-18 the rest of weight
#   loading (pread, chunked embedding copy, MTP name prefilter), 19 token identity and Responses output,
#   20 isolate the MTP drafter from target auxiliary outputs, 21 schema-validated tool arguments,
#   22 cover the served M-RoPE window in the target and MTP drafter, 23 dflash/DFlash2 drafter seam
#   for Qwen4Exp (vllm#56088).
# Upstream now: 3 (vllm#50729), 7 (vllm#55557), 9 (vllm#52775),
# 11 (vllm#55513), 16 (vllm#58720). Older bases remain in git history.
#
#   docker build -t qwen38-flash-dgx:v0.32 .
FROM vllm/vllm-openai:v0.32.0
LABEL qwen38.base=v0.32
ARG SP=/usr/local/lib/python3.12/dist-packages
ARG PKG=${SP}/vllm/models/qwen4_exp/nvidia
ARG PLE=${PKG}/ngram_embedding.py

# --- 1. PLE n-gram table from disk (VLLM_PLE_MMAP=1) ---
COPY src/vllm_ple_mmap.py ${SP}/vllm_ple_mmap.py
RUN cp ${PLE} ${PLE}.orig \
 && printf '\n\n# --- qwen38-flash-dgx: serve the PLE n-gram table from disk (VLLM_PLE_MMAP=1) ---\nfrom vllm_ple_mmap import apply as _ple_mmap_apply\n_ple_mmap_apply(Qwen4ExpNGramEmbedding)\n' >> ${PLE} \
 && python3 -c "import ast; ast.parse(open('${PLE}').read()); print('ngram_embedding.py patched OK')"

# --- 2. GB10 FLA fixes (@Saren-Arterius) ---
ARG FLA_UTILS=${SP}/vllm/third_party/flash_linear_attention/ops/utils.py
ARG FLA_CDH=${SP}/vllm/third_party/flash_linear_attention/ops/chunk_delta_h.py
RUN sed -i 's|DEFAULT = 102400|DEFAULT = 101376  # spark-fla-shmem: GB10 99KiB, big GDN tiles fit|' ${FLA_UTILS} \
 && grep -q "spark-fla-shmem" ${FLA_UTILS} && echo "fla shmem gate patched" \
 && sed -i 's|for num_warps in \[2, 4\]|for num_warps in [2]  # spark-fla-warps: fla#953 Blackwell tl.dot race|' ${FLA_CDH} \
 && grep -q "spark-fla-warps" ${FLA_CDH} && echo "fla num_warps pinned"

# --- 4. Prefix caching: block_size fix (core.py takes the min group block) ---
COPY src/patch_mamba_block_size.py /tmp/patch_mamba_block_size.py
RUN python3 /tmp/patch_mamba_block_size.py ${SP} && rm /tmp/patch_mamba_block_size.py

# --- 5. Exact QSA top-k fallback (VLLM_QSA_EXACT_TOPK=1) ---
COPY src/patch_qsa_exact_topk.py /tmp/patch_qsa_exact_topk.py
RUN python3 /tmp/patch_qsa_exact_topk.py ${PKG}/ops/qsa_indexer.py && rm /tmp/patch_qsa_exact_topk.py

# --- 6. Hybrid mode: NVFP4 experts + blockwise-fp8 side layers (VLLM_FP8_HYBRID=1) ---
ARG MO=${SP}/vllm/model_executor/layers/quantization/modelopt.py
ARG QSA=${PKG}/qsa.py
COPY src/vllm_fp8_hybrid_modelopt.py ${SP}/vllm_fp8_hybrid_modelopt.py
RUN cp ${MO} ${MO}.orig \
 && printf '\n\n# --- qwen38-flash-dgx: NVFP4 + blockwise-fp8 side layers (VLLM_FP8_HYBRID=1) ---\nfrom vllm_fp8_hybrid_modelopt import apply as _fp8_hybrid_apply\n_fp8_hybrid_apply()\n' >> ${MO} \
 && python3 -c "import ast; ast.parse(open('${MO}').read()); print('modelopt.py hooked OK')" \
 && cp ${QSA} ${QSA}.orig \
 && sed -i 's/quant_config=model\.without_modelopt_fp4(quant_config)/quant_config=_fp8_hybrid_excluded(quant_config)/' ${QSA} \
 && sed -i 's/^from \. import model$/from . import model\nfrom vllm_fp8_hybrid_modelopt import excluded_quant_config as _fp8_hybrid_excluded, qsa_requires_separate_projections as _fp8_hybrid_separate_qsa/' ${QSA} \
 && sed -i 's/self.fuse_indexer_projection = vllm_config.lora_config is None/self.fuse_indexer_projection = (vllm_config.lora_config is None and not _fp8_hybrid_separate_qsa(quant_config, prefix))/' ${QSA} \
 && grep -q '_fp8_hybrid_separate_qsa(quant_config, prefix)' ${QSA} && grep -q "_fp8_hybrid_excluded(quant_config)" ${QSA} && grep -q "^from vllm_fp8_hybrid_modelopt import" ${QSA} \
 && python3 -c "import ast; ast.parse(open('${QSA}').read()); print('qsa.py hooked OK')"

# --- 7. fp8_e4m3 main KV cache on the QSA path (--kv-cache-dtype fp8_e4m3; KV_DTYPE=fp8_e4m3) ---
# The earlier local implementation followed @Nanetnounou's original (issue #6).
# Use upstream support now, with assertions against accidental base drift.
# Native since v0.31 (vllm#55557); do not apply the older local patch.
RUN grep -q 'IS_FP8=is_fp8' ${PKG}/ops/qsa.py && grep -q '"fp8_e4m3"' ${QSA}

# --- 8. Deterministic persistent_topk kernel (@jschmied, vllm#55122), built with the image's nvcc ---
# v0.30 changed the stock kernel (vllm#54110 low-smem fallback, vllm#56346 sampled filtering): whether GB10
# still needs this one is re-measured by scripts/smoke-test.sh (determinism) on every boot.
ARG KDET_SHA=e0ef69d4f5575dad00d34e05479eaf4c6547bace
ARG KDET=https://raw.githubusercontent.com/jschmied/qwen38-flash-next-gb10/${KDET_SHA}
ARG DET_ARCH=121a
ADD --checksum=sha256:138cacfc5eb117f0922d53c88727e4d0dc26dcfb246c3d401fc280cfc726cc71 ${KDET}/patches/kernel-det/build_det.py /opt/llm/kernel-det/src/build_det.py
ADD --checksum=sha256:b103fbeaf7589b9468471142ad0b30012a076f93d20ba11fc5ff6dcb1ecd32a6 ${KDET}/patches/kernel-det/bindings_det.cpp /opt/llm/kernel-det/src/bindings_det.cpp
ADD --checksum=sha256:19e1d53425ea9a839445722fd1dac1c41727128eebdce84508c1bfb8592afecf ${KDET}/patches/kernel-det/topk_det.cu /opt/llm/kernel-det/src/topk_det.cu
ADD --checksum=sha256:16939700ae389750782ff5c0d5b9caef59aa0ff8b869b64ec94fa72c814910ee ${KDET}/patches/kernel-det/torch_utils.h /opt/llm/kernel-det/src/torch_utils.h
ADD --checksum=sha256:b4ef9ce298d43d6c0e6db9fcca451df20815b2cfe33791919c1ad9c0e84f0ba7 ${KDET}/patches/kernel-det/persistent_topk.cuh /opt/llm/kernel-det/src/persistent_topk.cuh
COPY src/patch_qsadet.py /tmp/patch_qsadet.py
RUN cd /opt/llm/kernel-det/src && DET_BUILD_DIR=/opt/llm/kernel-det/build DET_ARCH=${DET_ARCH} python3 build_det.py 2>&1 | tail -2 \
 && cp /opt/llm/kernel-det/build/_C_det.so /opt/llm/kernel-det/_C_det.so \
 && python3 /tmp/patch_qsadet.py ${PKG}/ops/qsa_indexer.py && rm /tmp/patch_qsadet.py \
 && python3 -c "import ast; ast.parse(open('${PKG}/ops/qsa_indexer.py').read()); print('qsadet wired OK')"

# --- 10. Reduced draft vocabulary for the MTP drafter (VLLM_MTP_DRAFT_VOCAB) ---
COPY src/patch_mtp_draft_vocab.py /tmp/patch_mtp_draft_vocab.py
COPY src/draft_vocab_65536.npy /opt/llm/draft_vocab_65536.npy
RUN python3 /tmp/patch_mtp_draft_vocab.py ${PKG}/mtp.py && rm /tmp/patch_mtp_draft_vocab.py

# --- 12. Preserve quoted/malformed Qwen tool markers in reasoning and content ---
# Confirm a tool preamble before switching output channels.
COPY src/patches/qwen-tool-preamble.patch /tmp/qwen-tool-preamble.patch
RUN cd ${SP} && patch --batch --forward --fuzz=0 -p1 < /tmp/qwen-tool-preamble.patch \
 && rm /tmp/qwen-tool-preamble.patch

# --- 13. Quoted Qwen tool markers the model is documenting, not calling ---
# A marker inside a fenced code block, or a wrapper-less <function= header mid-line,
# stays text and opens no tool call. Needs 12 applied first.
COPY src/patches/qwen-tool-marker-guard.patch /tmp/qwen-tool-marker-guard.patch
RUN cd ${SP} && patch --batch --forward --fuzz=0 -p1 < /tmp/qwen-tool-marker-guard.patch \
 && rm /tmp/qwen-tool-marker-guard.patch

# --- 19. Qwen token identity and truthful Responses terminal events (issue #31) ---
# Ported from Chaos2522's source bundle to v0.32; requires patches 12 and 13.
# No sampling/weight/KV changes. Responses buffers complete XML arguments;
# Chat Completions retains its incremental argument streaming.
COPY src/patches/qwen-safe-output.patch /tmp/qwen-safe-output.patch
RUN cd ${SP} && patch --batch --forward --fuzz=0 -p1 < /tmp/qwen-safe-output.patch \
 && rm /tmp/qwen-safe-output.patch

# --- 14. MoE weight loading: clone the mmap view before the H2D copy (VLLM_LOAD_CLONE=0 disables) ---
# PR #33 (@Willian-Zhang): the two call sites in routed_experts.py; byte-identical weights.
COPY src/patch_moe_load_clone.py /tmp/patch_moe_load_clone.py
RUN python3 /tmp/patch_moe_load_clone.py ${SP} && rm /tmp/patch_moe_load_clone.py

# --- 15-18. The rest of weight loading (docs/HOW-IT-WORKS.md, "The rest of weight loading") ---
# 15: pread tensors <= 64 MiB into ordinary memory, not mmap views; never the PLE table (needs 14).
# 16: indexed expert-name matching is native upstream since v0.31.
# 17: embed_tokens / lm_head copied to the GPU in 64 MiB pieces via ordinary memory.
# 18: the MTP drafter skips non-MTP tensors by name before they are read.
COPY src/patch_load_pread.py src/patch_embed_chunked_copy.py \
     src/patch_mtp_name_prefilter.py /tmp/
RUN python3 /tmp/patch_load_pread.py ${SP} \
 && grep -q 'mapping_by_expert = _index_expert_mapping' ${SP}/vllm/model_executor/layers/fused_moe/routed_experts.py \
 && python3 /tmp/patch_embed_chunked_copy.py ${SP} && python3 /tmp/patch_mtp_name_prefilter.py ${SP} \
 && rm /tmp/patch_load_pread.py /tmp/patch_embed_chunked_copy.py \
       /tmp/patch_mtp_name_prefilter.py

# --- 20. Routed-expert auxiliary outputs belong to the target, not MTP (issue #55) ---
# Diagnosis/fix reported by @rp-tn. Keep target capture enabled; clear only the draft copy.
COPY src/patches/qwen-mtp-aux-output.patch /tmp/qwen-mtp-aux-output.patch
RUN cd ${SP} && patch --batch --forward --fuzz=0 -p1 < /tmp/qwen-mtp-aux-output.patch \
 && rm /tmp/qwen-mtp-aux-output.patch

# --- 21. Repair accidental Qwen arguments wrappers using the offered schema (issue #31) ---
# Ported from Chaos2522's schema2 bundle; requires patch 19 including the #56 follow-up.
# Keep Chat compatibility and incremental normal arguments; validate Responses before emission.
COPY src/patches/qwen-tool-argument-schema.patch /tmp/qwen-tool-argument-schema.patch
RUN cd ${SP} && patch --batch --forward --fuzz=0 -p1 < /tmp/qwen-tool-argument-schema.patch \
 && python3 -c "import ast, jsonschema, referencing; ast.parse(open('${SP}/vllm/parser/engine/argument_validation.py').read())" \
 && rm /tmp/qwen-tool-argument-schema.patch

# --- 22. M-RoPE cache covers served positions, including the MTP draft path (issue #60) ---
# Adapted from wtogami's vllm#62409; preserve the original YaRN frequency base and cache rows.
COPY src/patches/qwen-mrope-cache-window.patch /tmp/qwen-mrope-cache-window.patch
RUN cd ${SP} && patch --batch --forward --fuzz=0 -p1 < /tmp/qwen-mrope-cache-window.patch \
 && python3 -c "import ast; ast.parse(open('${SP}/vllm/model_executor/layers/rotary_embedding/mrope.py').read())" \
 && rm /tmp/qwen-mrope-cache-window.patch

# --- 23. dflash/DFlash2 drafter seam on the Qwen4Exp target (vllm#56088) ---
# Open the speculative-method allowlist, mount the EAGLE-3-style target
# aux-hidden-state capture that DFlash drafting needs, and port the QSA ring
# widening (vllm#54912) for speculation depths >= 5. Inert for the MTP path; the
# scripts/serve.sh dflash branch pins the V2 runner that init_speculator requires.
COPY src/patch_dflash_qwen4exp.py /tmp/patch_dflash_qwen4exp.py
RUN python3 /tmp/patch_dflash_qwen4exp.py ${SP} && rm /tmp/patch_dflash_qwen4exp.py

# Advertise DFlash only after patch 23 has applied successfully (issue #64).
LABEL qwen38.dflash=1
