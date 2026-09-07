# Qwen3.8-Flash-Next on a single DGX Spark / GB10, via vLLM.
#
# Starts from the official Qwen3.8-Flash-Next vLLM image and layers a handful of
# patches on it. Every one of them is a no-op unless its runtime flag is set, so
# the image still behaves like upstream with the flags off:
#
#   1. PLE table served from disk via mmap            (VLLM_PLE_MMAP=1)      — the one that makes it fit
#   2. GB10 FLA fixes                                  (always on, harmless elsewhere)
#   3. Mamba state-copy race fix + bounds guard        (always on)
#   4. Prefix-caching block_size fix                   (always on; needed for --enable-prefix-caching)
#   5. Exact, deterministic QSA top-k                  (VLLM_QSA_EXACT_TOPK=1)
#   6. NVFP4 experts + fp8 side-layers "hybrid" mode   (VLLM_FP8_HYBRID=1)
#   7. fp8_e4m3 KV cache on the QSA path              (--kv-cache-dtype fp8_e4m3)
#   8. Deterministic persistent_topk kernel           (VLLM_QSA_DET_TOPK=1) — replaces 5 at no prefill cost
#   9. M%4 padding for the blockwise-fp8 GEMM         (VLLM_FP8_PAD_M4=1)   — hybrid mode with prefix caching OFF
#
#   docker build -t qwen38-flash-dgx .
#
# The base image is multi-arch (arm64 for the Spark's Grace CPU). Pinned by digest
# for reproducibility; bump the tag below if the upstream recipe moves.
FROM vllm/vllm-openai:qwen38-flash-next@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8

# Package layout inside the official image (vLLM 0.1.dev20073, torch 2.13 cu130,
# numpy 2.2.6 — the patch needs numpy, already present).
ARG SP=/usr/local/lib/python3.12/dist-packages
ARG PLE=${SP}/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py

# --- 1. PLE n-gram table from disk ------------------------------------------------
COPY src/vllm_ple_mmap.py ${SP}/vllm_ple_mmap.py
RUN cp ${PLE} ${PLE}.orig \
 && printf '\n\n# --- qwen38-flash-dgx: serve the PLE n-gram table from disk (VLLM_PLE_MMAP=1) ---\nfrom vllm_ple_mmap import apply as _ple_mmap_apply\n_ple_mmap_apply(Qwen3_8FlashNextNGramEmbedding)\n' >> ${PLE} \
 && python3 -c "import ast; ast.parse(open('${PLE}').read()); print('ple_layer.py patched OK')"

# --- 2. GB10 FLA fixes, contributed by @Saren-Arterius ----------------------------
#     (https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound)
# 1) sm_121 reports 99 KiB of shared memory per block; the flash-linear-attention
#    gate asks for 100 KiB, so all 36 GDN layers silently fell back to small tiles.
# 2) fla#953: tl.dot race on Blackwell with num_warps=4 in chunk_delta_h -> pin 2.
ARG FLA_UTILS=${SP}/vllm/third_party/flash_linear_attention/ops/utils.py
ARG FLA_CDH=${SP}/vllm/third_party/flash_linear_attention/ops/chunk_delta_h.py
RUN sed -i 's|DEFAULT = 102400|DEFAULT = 101376  # spark-fla-shmem: GB10 99KiB, big GDN tiles fit|' ${FLA_UTILS} \
 && grep -q "spark-fla-shmem" ${FLA_UTILS} && echo "fla shmem gate patched" \
 && sed -i 's|for num_warps in \[2, 4\]|for num_warps in [2]  # spark-fla-warps: fla#953 Blackwell tl.dot race|' ${FLA_CDH} \
 && grep -q "spark-fla-warps" ${FLA_CDH} && echo "fla num_warps pinned"

# --- 3. Mamba state copy: vllm#50729 + bounds guard --------------------------------
# src/mamba_utils_guarded.py = vLLM's v1/worker/mamba_utils.py with
#   - vllm#50729 "[Bugfix][Mamba] Fix overlapping state copy race" (@AndreasKaratzas), and
#   - a bounds guard by @Saren-Arterius: a state copy with an out-of-range block id is
#     skipped and counted ("mamba state-copy guard: N out-of-range ...") instead of
#     taking down the CUDA context with an illegal memory access.
ARG MU=${SP}/vllm/v1/worker/mamba_utils.py
RUN cp ${MU} ${MU}.orig
COPY src/mamba_utils_guarded.py ${MU}
RUN python3 -c "import ast; ast.parse(open('${MU}').read()); print('mamba_utils.py guarded OK')"

# --- 4. Prefix caching: block_size fix ----------------------------------------------
# vLLM's EngineCore overwrites cache_config.block_size with the smallest KV-group block
# size (here the QSA raw-key ring: 8/16 tokens) while the Mamba block is 1600. Two
# consumers used the former as the latter, so a prefix hit restored an all-zero Mamba
# state. See docs/HOW-IT-WORKS.md. With this, --enable-prefix-caching is correct.
COPY src/patch_mamba_block_size.py /tmp/patch_mamba_block_size.py
RUN python3 /tmp/patch_mamba_block_size.py ${SP} && rm /tmp/patch_mamba_block_size.py

# --- 5. Exact QSA top-k (VLLM_QSA_EXACT_TOPK=1) ---------------------------------------
# The stock persistent_topk kernel is non-deterministic on GB10 and can drop real
# top-k candidates (vllm#51782; reported here by @k3dani, issue #3). The exact path
# uses torch.topk over the visible columns. Opt-in; scripts/serve.sh enables it.
COPY src/patch_qsa_exact_topk.py /tmp/patch_qsa_exact_topk.py
RUN python3 /tmp/patch_qsa_exact_topk.py ${SP}/vllm/models/qwen3_8_flash_next/nvidia/ops/qsa.py && rm /tmp/patch_qsa_exact_topk.py

# --- 6. Hybrid mode: NVFP4 experts + blockwise-fp8 side layers (VLLM_FP8_HYBRID=1) ----
# Lets the GDN in/out projections, QSA q/k/v/o and shared experts be stored as
# blockwise fp8 (DeepSeek layout, produced by tools/fp8_convert.py) while the MoE
# experts stay NVFP4. Port of @Saren-Arterius's int4+fp8 dispatch to ModelOpt-NVFP4.
# No-op unless VLLM_FP8_HYBRID=1 (and harmless on a plain NVFP4 checkpoint).
ARG MO=${SP}/vllm/model_executor/layers/quantization/modelopt.py
ARG QSA=${SP}/vllm/models/qwen3_8_flash_next/nvidia/qsa.py
COPY src/vllm_fp8_hybrid_modelopt.py ${SP}/vllm_fp8_hybrid_modelopt.py
RUN cp ${MO} ${MO}.orig \
 && printf '\n\n# --- qwen38-flash-dgx: NVFP4 + blockwise-fp8 side layers (VLLM_FP8_HYBRID=1) ---\nfrom vllm_fp8_hybrid_modelopt import apply as _fp8_hybrid_apply\n_fp8_hybrid_apply()\n' >> ${MO} \
 && python3 -c "import ast; ast.parse(open('${MO}').read()); print('modelopt.py hooked OK')" \
 && cp ${QSA} ${QSA}.orig \
 && sed -i 's/quant_config=model\.without_modelopt_fp4(quant_config)/quant_config=_fp8_hybrid_excluded(quant_config)/' ${QSA} \
 && sed -i 's/^from \. import model$/from . import model\nfrom vllm_fp8_hybrid_modelopt import excluded_quant_config as _fp8_hybrid_excluded/' ${QSA} \
 && grep -q "_fp8_hybrid_excluded(quant_config)" ${QSA} && grep -q "^from vllm_fp8_hybrid_modelopt import" ${QSA} \
 && python3 -c "import ast; ast.parse(open('${QSA}').read()); print('qsa.py hooked OK')"

# --- 7. fp8_e4m3 KV cache on the QSA path (--kv-cache-dtype fp8_e4m3) ------------------
# Contributed by @Nanetnounou (issue #6, vllm#54426). Dequantizes on the read side of the
# QSA Triton kernels (main KV and the indexer's raw-key ring), relaxes the bf16-only guards.
# Inert with --kv-cache-dtype auto (the Triton branch is compiled out). ~1.9x KV pool,
# 1M context on one box, at a speed and quality cost — see README before using it.
COPY src/patch_qsa_fp8_kv.py /tmp/patch_qsa_fp8_kv.py
RUN python3 /tmp/patch_qsa_fp8_kv.py ${SP} && rm /tmp/patch_qsa_fp8_kv.py

# --- 8. Deterministic persistent_topk kernel (VLLM_QSA_DET_TOPK=1) -----------------------
# Kernel-side fix for the non-deterministic / candidate-dropping QSA top-k (issue #3, vllm#51782):
# @jschmied's deterministic persistent_topk, upstream as vllm#55122, built here as a standalone
# extension (_C_det.so) with the image's nvcc — no vLLM rebuild. Same determinism as patch 5's
# exact torch.topk, but at kernel speed: on the GX10 it recovers the whole prefill penalty
# (8k: 1,476 -> 2,436 tok/s; 32k: 1,794 -> 2,904; needle 92k: 69 s -> 48 s) with decode unchanged.
# Sources are fetched from https://github.com/jschmied/qwen38-flash-next-gb10 at a pinned commit
# (Apache-2.0; attribution: @jschmied). Set DET_ARCH=120a for
# an x86 Blackwell (RTX 5090). Inert unless VLLM_QSA_DET_TOPK=1; VLLM_QSA_EXACT_TOPK=1 still wins.
ARG KDET_SHA=e0ef69d4f5575dad00d34e05479eaf4c6547bace
ARG KDET=https://raw.githubusercontent.com/jschmied/qwen38-flash-next-gb10/${KDET_SHA}
ARG DET_ARCH=121a
# Pinned by commit AND sha256 (ADD --checksum needs BuildKit, the default since Docker 23).
ADD --checksum=sha256:138cacfc5eb117f0922d53c88727e4d0dc26dcfb246c3d401fc280cfc726cc71 ${KDET}/patches/kernel-det/build_det.py /opt/llm/kernel-det/src/build_det.py
ADD --checksum=sha256:b103fbeaf7589b9468471142ad0b30012a076f93d20ba11fc5ff6dcb1ecd32a6 ${KDET}/patches/kernel-det/bindings_det.cpp /opt/llm/kernel-det/src/bindings_det.cpp
ADD --checksum=sha256:19e1d53425ea9a839445722fd1dac1c41727128eebdce84508c1bfb8592afecf ${KDET}/patches/kernel-det/topk_det.cu /opt/llm/kernel-det/src/topk_det.cu
ADD --checksum=sha256:16939700ae389750782ff5c0d5b9caef59aa0ff8b869b64ec94fa72c814910ee ${KDET}/patches/kernel-det/torch_utils.h /opt/llm/kernel-det/src/torch_utils.h
ADD --checksum=sha256:b4ef9ce298d43d6c0e6db9fcca451df20815b2cfe33791919c1ad9c0e84f0ba7 ${KDET}/patches/kernel-det/persistent_topk.cuh /opt/llm/kernel-det/src/persistent_topk.cuh
ADD --checksum=sha256:70905073fe3fa361030bf1cb469b74610766bdfe361419cd7df50af2561322e3 ${KDET}/tools/determinism/qsadet_patch.py /tmp/qsadet_patch.py
RUN cd /opt/llm/kernel-det/src && DET_BUILD_DIR=/opt/llm/kernel-det/build DET_ARCH=${DET_ARCH} python3 build_det.py 2>&1 | tail -2 \
 && cp /opt/llm/kernel-det/build/_C_det.so /opt/llm/kernel-det/_C_det.so \
 && VLLM_QSA_PY=${SP}/vllm/models/qwen3_8_flash_next/nvidia/ops/qsa.py python3 /tmp/qsadet_patch.py && rm /tmp/qsadet_patch.py \
 && python3 -c "import ast; ast.parse(open('${SP}/vllm/models/qwen3_8_flash_next/nvidia/ops/qsa.py').read()); print('qsadet wired OK')"

# --- 9. M%4 padding for the sm_12x blockwise-FP8 GEMM (VLLM_FP8_PAD_M4=1, opt-in) --------
# The preview image's cutlass blockwise-fp8 dispatch routes any GEMM whose row count M is not a
# multiple of 4 (or <= 64) to a swap_ab path: ~1.7x slower below 2,048 rows, ~10x above (measured
# on a GX10; fixed upstream in C++ by vllm#52775, after this image was cut). Only the hybrid mode's
# fp8 side layers use this GEMM. @jschmied's drop-in pads M to a multiple of 4 inside an opaque
# custom op (issue #3). With --enable-prefix-caching (our default) prefill chunks are already
# clipped to the 1,600-token Mamba block, so M % 4 == 0 and the patch is a no-op: OFF by default.
# With PREFIX_CACHE=0 in hybrid mode it is worth about -40% TTFT at 8k. Fetched at a pinned commit
# (Apache-2.0; attribution: @jschmied). NOTE: the patch itself
# defaults to ON on sm_12x when the env var is unset, so scripts/serve.sh always sets it explicitly.
ARG KM4_SHA=d9705bde5a5b294478a5baf82b888a64000a16ef
ADD --checksum=sha256:deb7b7865c84ab9921e1f8e7d5c60d366910092a7888f8f570ddf5d35d83eff8 https://raw.githubusercontent.com/jschmied/qwen38-flash-next-gb10/${KM4_SHA}/tools/main/fp8_m4pad_patch.py /tmp/fp8_m4pad_patch.py
RUN python3 /tmp/fp8_m4pad_patch.py && rm /tmp/fp8_m4pad_patch.py \
 && python3 -c "import ast; p='${SP}/vllm/model_executor/kernels/linear/scaled_mm/cutlass.py'; s=open(p).read(); ast.parse(s); assert 'fp8m4pad::scaled_mm_padded' in s; print('fp8 m4pad wired OK')"
