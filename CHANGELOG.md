# Changelog

## 0.1.0 — 2026-11-19

First release.

### Added
- Jev requests (noul / choice / score) answered from option-letter logits, averaged over option orders;
  `orders_agree` in every answer.
- `DecisionGem(..., offline=True)` takes the model only from the local HF cache; `repo` can be a local model folder.
- The answer computation is a pure function `answer()`; an error in any question is raised before the model runs.
- Keyed states: `prefill` / `decide(key=...)` / `forget` / `save` / `load` — a text is read once, then asked any
  number of questions in any order.
- Open states: `prefill(..., open=True)`, `extend`, `split_open`.
- `generate()` and `{"prompt": ...}` requests: the same model answers ordinary prompts.
- `unload()` and `with DecisionGem() as gem:` — unload the model without ending the process.
- `cache_limit` — the MLX buffer cache limit (`mx.set_cache_limit`), 0 by default.
- `orders="cyclic"` — every rotation of the options, so position bias cancels fully.
- `letter_ids`: at load, every letter A..Z must be one token after the chat template, else `ValueError`; a model
  without a chat template is rejected.
- `decide(..., raw=True)`, `{"raw": true}`, `--raw`: every answer carries `logprobs`, the option log-probabilities
  before calibration.
- `fit_calibration(answers, labels)`: fits the calibration (`[a, b]` by Platt scaling for noul, `T` by the least
  log-loss for choice / score) from labelled raw answers.
- `float32=True` (default): every bf16 parameter is cast to float32, so a text read whole, in chunks or with
  `extend` gives the same answer on long states; `save` records the compute `dtype` and `load` rejects a mismatch.
- On GPUs with neural accelerators (M5) `float32=True` sets `MLX_ENABLE_TF32=0` and checks float32 matmuls against
  the CPU.
