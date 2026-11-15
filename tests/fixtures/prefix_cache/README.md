# vLLM source fixture

`abstract_v033.py` is the unmodified inspected vLLM 0.33 source file
`vllm/model_executor/layers/mamba/abstract.py`, extracted from the pinned image
used by this recipe. Its SHA-256 is recorded in
`src/prefix_cache_boundaries/manifest.json`. Copyright belongs to the vLLM
contributors, as stated in the retained SPDX header; license: Apache-2.0.

Tests parse the capability function from this fixture without importing
PyTorch or vLLM. Keep it unchanged unless deliberately reviewing a new version.
