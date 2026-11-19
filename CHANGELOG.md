# Changelog

## Unreleased

### Added
- `letter_ids`: at load, every letter A..Z must be one token after the chat template, else `ValueError`; a model
  without a chat template is rejected. Letter ids for `gemma-4-26b-a4b-it-4bit` are unchanged.
- `decide(..., raw=True)`, `{"raw": true}`, `--raw`: every answer carries `logprobs`, the option log-probabilities
  before calibration.
- `fit_calibration(answers, labels)`: fits the calibration (`[a, b]` by Platt scaling for noul, `T` by the least
  log-loss for choice / score) from labelled raw answers.
- README: "Other models" — requirements and the models checked.

## 0.1.0 — 2026-11-13

First standalone release, extracted from an internal speech-analytics project. The core logic is unchanged: the same
request gives a byte-identical answer through the original module and through the package.

### Differences from the original module
- No global `HF_HUB_OFFLINE` on import; `DecisionGem(..., offline=True)` takes the model only from the local HF
  cache (`snapshot_download(local_files_only=True)`).
- `repo` can be a local model folder.
- The answer computation is a pure function `answer()`, with tests that need no model.
- An error in any question is raised before the model runs.
- `gem(request)` takes a whole Jev request.

### Added
- Keyed states: `prefill` / `decide(key=...)` / `forget` / `save` / `load` — a text is read once, then asked any
  number of questions in any order.
- Open states: `prefill(..., open=True)`, `extend`, `split_open`.
- `generate()` and `{"prompt": ...}` requests: the same model answers ordinary prompts.
- `unload()` and `with DecisionGem() as gem:` — unload the model without ending the process.
- `cache_limit` — the MLX buffer cache limit as a parameter (was a hard-coded `mx.set_cache_limit(0)`).
- `orders="cyclic"` — every rotation of the options, so position bias cancels fully.

### Fixed
- bf16 noise on long states: `float32=True` by default casts every bf16 parameter to float32; `save` records the
  compute `dtype` and `load` rejects a mismatch.
- TF32 on M5-class GPUs: `float32=True` sets `MLX_ENABLE_TF32=0` and checks float32 matmuls against the CPU.
