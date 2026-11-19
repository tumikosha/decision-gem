# decision-gem

A decision model in the style of [TypeSafe Jev](https://developers.cloudflare.com/ai/models/typesafe/jev/) on a local
Gemma (MLX). It answers three kinds of questions about a text (the "state"): yes / no, a choice among options, a score
on a scale. No text is generated: the answer is read from the probabilities of the options' letters. Runs offline.
The same model also answers ordinary prompts by generating (`generate`, see "Prompts").

Russian version: [README.md](README.md).

## What it does

Request: `{"state": text or JSON object, "questions": {id: question}}`.

| Type | Question | Answer |
|---|---|---|
| `noul` | `instructions`, `criteria` `{"true": …, "false": …}` (optional) | `noul` = P(true) |
| `choice` | `instructions`, `criteria` `{option: description}` or `[option, …]`, 2..26 | `choice`, `confidence`, `probabilities` |
| `score` | `instructions`, `criteria` `{value: description}` or `[description, …]` (levels 1..N), 2..10 | `score` = Σ p·value, `confidence`, `legend`, `probabilities` |

Beyond Jev, every answer has `orders_agree`: whether the choice is the same in every option order (below).
`false` means an unsure answer, worth checking. The whole response: `{"model", "answers": {id: answer},
"usage": {"input_tokens", "output_tokens": 0}}`.

How it works:
1. The state is read once into a KV cache.
2. Each question is appended to a copy of the cache, its options under the letters A, B, C …
3. The logits of those letters at the first answer position are read.
4. Each question is read in several option orders. Each option's log-probabilities are averaged, so a
   preference for a letter or a position cancels out.

### Option orders

`DecisionGem(orders=...)`, per request `decide(..., orders=...)`, on the CLI `--orders`:

| `orders` | Passes per question | What cancels |
|---|---|---|
| `"reverse"` (default) | 2: in order and reversed | with 2 options, everything; with 3+ the middle stays in the middle (of 3 options, B is B both times), so a pull to or from the middle letters stays |
| `"cyclic"` | n: every rotation | each option once under every letter: position cancels fully |

For noul (2 options) both are the same two passes.

Under `cyclic`, `orders_agree` means all n passes agree, stricter than two; its `false` rate is not comparable
across modes. Calibration `T` was fitted on `reverse`; fit it again for `cyclic`.

### Calibration

Raw LLM probabilities are overconfident: `noul` 1.0 does not mean "yes" 100% of the time. Calibration remaps them so
that 0.8 means "yes" in about 80% of such answers. Each question has its own bias, so the coefficients are kept per
question id: `[a, b]` for noul (`p = sigmoid(a·logit + b)`, `logit` being the log-probability of true minus that of
false), `T` for choice / score (`softmax(logprob / T)`, `T > 1` softens).

```json
{"spam": [0.35, -0.8], "kind": 2.1}
```

Pass it as `DecisionGem(calibration=...)` or `--calibration cal.json`. To fit it:

1. Label 100–300 of your own texts by hand: the answer to every question.
2. Get answers with the raw log-probabilities: `decide(..., raw=True)`, `{"raw": true}` in a request or `--raw` in the
   CLI. Every answer gets `logprobs` {option: log-probability before calibration}.
3. `fit_calibration(answers, labels)` returns the calibration.

```python
from decision_gem import DecisionGem, fit_calibration

with DecisionGem() as gem:
    answers = [gem.decide(text, qs, raw=True)["answers"] for text in texts]
cal = fit_calibration(answers, labels)   # labels: [{"spam": True, "kind": "phishing"}, ...] in the same order
```

A label is `True` / `False` for noul, the option for choice, the level for score; a question with no label in a text is
skipped. noul is fitted by Platt scaling (logistic regression on `logit`), `T` by the least log-loss. Both noul labels
and some model mistakes must be in the set: where the model is always right, `T` goes to 0.01 and `a` grows.


## Differences from Jev

The request and the response are Jev's (noul / choice / score). Everything under the contract differs. Facts about Jev
come from the Cloudflare and TypeSafe docs (`docs.typesafe.ai/models.md`), November 2026; they do not disclose how Jev
works inside.

| | Jev (TypeSafe) | decision-gem |
|---|---|---|
| Where it runs | a cloud API (Cloudflare Workers AI, TypeSafe `POST /v1/systemone`), $0.042 per 1M input tokens | locally on Apple silicon (MLX), offline, data stays on the machine |
| Model | its own, trained to return calibrated decisions (RLCD), architecture not disclosed | any instruct LLM in mlx-lm format, see "Other models" |
| How it computes | not disclosed | the logits of the option letters on a copy of the state's KV cache, averaged over option orders |
| Calibration | built in by training | fitted on your own labels: `fit_calibration`, see "Calibration" |
| Question types | noul, choice, score, Box (from Jev 1.14) | noul, choice, score |
| Images (`assets`) | yes (beta) | no: text or JSON |
| Limits | 64k tokens per request, 32k per question (262k with a header) | the model's context and memory; choice 2..26 options, score 2..10 levels |

Only in decision-gem:

- `orders_agree` in every answer and `orders="cyclic"` (see "Option orders");
- `raw=True`: the raw option `logprobs`, and `fit_calibration` on them;
- keyed states: `prefill`, `decide(key=...)`, `forget`, `save` / `load`;
- open states and `extend`: text is read into the cache as it grows;
- `generate`: the same model answers ordinary prompts;
- score `criteria` also as `{value: description}`, choice `criteria` also as a list.

## Keyed states

`decide(state, questions)` reads the state, answers and drops its cache. To ask one email many questions at different
times, or to switch between emails, read the state once and keep it under a key:

| Method | What it does |
|---|---|
| `prefill(key, state)` | reads the state into a KV cache and keeps it under `key`; returns the number of tokens |
| `decide(key=key, questions=qs)` | questions on a kept state; exactly one of `state` / `key`, else `ValueError` |
| `gem({"key": key, "questions": qs})` | the same as a Jev request |
| `forget(key)` | frees the state's memory |
| `save(key, path)` / `load(key, path)` | the state to a `.safetensors` file and back; `load` takes only a file saved with the same model |

Each question runs on its own copy of the cache, so the kept state does not change: questions are independent and the
model does not remember earlier questions (it is not a chat). An answer by key is byte-identical to the answer of
`decide(state, ...)` — checked on 3 calls in reverse order and after `save` + `load`. Copying the cache is almost
free: MLX arrays are immutable, the data is shared until a question is appended to the copy.

A state file is the email's text in another form: keep it only where the text itself may be kept.

### Open states

`prefill(key, state, open=True)` stops the cache right after the state text; the closing `</state>` is not read yet.
`extend(key, text)` reads more text into that cache for good (it returns the tokens added; `ValueError` on a closed
state), so a growing thread can be kept open and new emails read into it. `decide` on an open state puts `</state>` and the question after the
cache on every copy and never writes the kept cache. `save` / `load` keep the open flag. `split_open(prompt)` is the
pure helper that cuts a rendered prompt before the closing tag.

Seam rule: `extend` equals one read only when the new text starts at a token boundary. Put the leading space or newline
in the new text, never split a word, and never end a chunk in whitespace (`extend` raises `ValueError`): `"hello. "` +
`"Operator"` tokenizes differently from `"hello. Operator"`. `prefill` strips the state, so it never ends in whitespace.

Precision: the gem computes in float32 by default (`float32=True`), see "The bf16 bug" below. `extend` equals one
read only in float32.

## Prompts

`generate(prompt, state=None, *, key=None, max_tokens=512, temperature=0.0)` runs the model as an ordinary chat model:
the answer to the prompt is generated (`mlx_lm.stream_generate`) by the model already loaded.

| Call | What the model sees |
|---|---|
| `generate(prompt)` | the prompt only |
| `generate(prompt, state)` | the state, then the prompt where a question would go |
| `generate(prompt, key=key)` | the kept state (`prefill` / `load`), not read again; generated on a copy of its cache |
| `gem({"prompt": …, "state" or "key", "max_tokens", "temperature"})` | the same as a request; the CLI takes this JSON too |

Response: `{"model", "text", "finish_reason": "stop" | "length", "usage": {"input_tokens", "output_tokens"}}`.
`temperature` 0 is greedy: the same prompt gives the same text. Generating leaves a kept state as it was:
`decide(key=…)` before and after `generate(key=…)` gives byte-identical answers (the `-m models` test); the answer on a
key equals the answer on the same `state`.

Limits: the answer is not constrained to a JSON schema (mlx-lm has no structured output) — check JSON and ask again
on an error; do not run beside Ollama with the same Gemma (memory).

## The bf16 bug: noisy answers on long states (found 2026-10-19)

**Symptom.** The same text gave a different answer depending on how it was read: whole, in chunks with `extend`,
or with `</state>` in another place. Up to ~460 tokens the answers agreed, beyond that they drifted:

| State tokens | Whole | In chunks (`extend`) |
|---|---|---|
| ≤ 462 | same | same |
| 692 | 0.75 | 0.37 |
| ~1400 | 0.45 | 0.22 |

Open vs closed state on one long text: 0.45 vs 0.61. One 1590-token text read in different ways gave `noul` from
0.14 to 0.65: at a 0.5 threshold the yes / no answer depended on the way of reading, not on the text.

**Cause.** bf16 compute flips the expert choice of the MoE router. bf16 rounding depends on how many tokens one
read holds. Each layer's router makes a hard top-8 expert pick (`argpartition`). When the 8th and 9th expert scores
are nearly equal (at layer 8 the gap was 2.8e-5), rounding flips the pick, and the difference grows through the
later layers. The norms and the quantization scales and biases were bf16; the weights themselves are 4-bit.

**Fix.** `DecisionGem(..., float32=True)` by default. After loading, every bf16 parameter
(quantization scales and biases, norms, embeddings, router) is cast to float32; the packed 4/8-bit weights stay as
they are. In float32 the spread is 0.568..0.574 and `extend` matches one read to within 0.006 (not bit-exact).
`float32=False` keeps the old behaviour.

Cost: +1.5 GiB for the model (13.2 → 14.7 GiB), twice the KV cache (≈0.43 MiB per token), reads about 1.3x slower,
a short +1.5 GiB at load. The "Resources" numbers were measured in bf16.

**Saved states.** `save` records the compute dtype (`dtype`), `load` raises `ValueError` when it
differs from the gem's. A file without the field counts as bf16 (saved before the fix). Without the check a bf16
state loaded silently into a float32 gem and gave a mixed cache with the same noise.

**What it touched.**
- Everything computed before 2026-10-19 (before version 0.1.0) ran in bf16.
- Operator / client roles on 145 calls (state = first 20 lines): in float32 no call changed its operator, accuracy
  is the same; 163 s vs 127 s. On short states the noise does not show.
- Answers on long calls (the checklist, the whole call in the state) may have depended on the noise. Calibrations
  fitted on bf16 answers need refitting.

**TF32 on M5 (found 2026-10-19).** Cause: on GPUs with neural accelerators (M5) MLX multiplies float32 matrices in
TF32 (relative error ~5e-4 instead of ~1e-6) from 128 rows up (dense: from 2), with another kernel below that. So the
same tokens get different rounding depending on how many one read holds, and the router's near-ties flip as in bf16.
Symptom on an M5 Max: 0.347 in chunks vs 0.546 whole. `DecisionGem(float32=True)` sets `MLX_ENABLE_TF32=0` and
checks a float32 matmul against the CPU (`float32_matmuls`); MLX reads the variable once, at the process's first
matmul, so if one already ran with TF32 on the gem raises `RuntimeError`. With it: 0.5739 in chunks vs 0.5741 whole.
Cost: reads 2x slower than in TF32 (1594 tokens 1.6 s vs 0.8 s, a question 0.30 s vs 0.18 s).

**Check.** `tests/test_open_model.py` (with the model): a long state in chunks equals one read, open equals closed,
tolerance 0.02. In bf16 these two tests fail (0.2306 and 0.1544). Model-free tests in `tests/test_core.py` cover the
cast and `dtype` in `save` / `load`.

## Where it was used

Speech analytics of call-centre calls, model `mlx-community/gemma-4-26b-a4b-it-4bit`.

- Operator / client roles: the right operator in 99 of the 100 calls of the roles reference.
- Checklist item statuses, 199 calls, 2250 items: 7.1 s per call, 9 of 9 manual rulings matched, the option orders
  agree on 2094 of 2250.
- The full checklist with no analysis prompt: scenario, outcome, call flags, then the items; 199 calls, 15.5 s per
  call, 9 of 9 manual rulings.

## Resources

Measured on the longest call (3051 tokens of state) and on 199 calls, 2026-10-13, an Apple silicon Mac with 36 GB:

| What | Time / memory |
|---|---|
| model load | about 6 s, 13.2 GiB |
| reading a state (`prefill` or the start of `decide`) | 4.2 s for 3051 tokens (0.8 s for 882) |
| a state's KV cache | ≈0.2 MiB per token: 656 MiB for 3051 tokens; 3 calls of 882–2430 tokens — 1.11 GiB |
| cache copy per question | 0.1 ms |
| one question (two passes, options in order and reversed) | 221 ms |
| generation (`generate`) | 86–97 tokens/s; a call's summary (653 state tokens, 71 answer tokens) in 0.7 s (16.10) |
| `save` / `load` of a state | 0.34 s / 0.02 s, file 656 MiB |
| peak memory on long calls | about 17 GiB (27.8 GiB with the MLX buffer cache) |
| a call in the runs above | 2–30 s, depends on the number of questions |

On a 36 GB Mac do not run it next to another large model. The GPU gets about 27 GiB: besides the model, about 15
kept long calls fit in memory; more through `save` / `load`.

## Memory

- `cache_limit` (default 0) — bytes of freed buffers MLX keeps for reuse (`mx.set_cache_limit`); `None` — leave the
  MLX limit as it is. With the buffers the peak was 27.8 GiB instead of 17. The limit is process-wide: `unload()`
  puts back the previous one (the MLX default is 34.2 GiB on this Mac).
- `gem.unload()` frees the model and all kept states (13.2 GiB → 0) without ending the process; `decide()` after it
  raises `RuntimeError`. Or `with DecisionGem() as gem: ...` — unloaded on exit.
- `forget(key)` — one state's memory (3 calls: 1.11 GiB, after `forget` of one 0.59 GiB).

## Other models

The model is set by `DecisionGem(repo=...)` or `--model`: an HF repo id or a local folder in mlx-lm format.

```python
gem = DecisionGem("mlx-community/Qwen3-14B-4bit")
```

```bash
decision-gem request.json --model mlx-community/Qwen3-14B-4bit
```

What a model needs:

- mlx-lm format (`mlx_lm.load` loads it), Apple silicon;
- instruction-tuned, with a chat template;
- every letter A..Z after the answer template is exactly one token. Checked at load (`letter_ids`): otherwise
  `ValueError` before the first question;
- the first answer token is a letter, not the start of reasoning. The template gets `enable_thinking=False`. A model
  that always reasons is read wrong, and the check does not catch it.

**Small models.** Models up to ~8B parameters, with no fine-tuning and no calibration, answer poorly. In JevBench
v1.6.0 (Benchmark Heaven, 2026-10-05) raw Qwen3 with a logit readout scored a Capability of 34.6 (4B Instruct),
30.1 (8B), 8.3 (1.7B) and 8.2 (0.6B); frozen Gemma-4-12B-it and Gemma-4-31B-it scored 70.9 and 77.6, Jev 1.13.0 76.5.
This covers `gemma-4-e2b` and `gemma-4-e4b` in "Model comparison" too: the tests on them check the package, not the
quality of the answers. Take a small model only with a calibration fitted on your own labels and its accuracy checked
on them.

Calibration (`[a, b]`, `T`) is per model. The numbers in this README were measured on `gemma-4-26b-a4b-it-4bit`.

### Model comparison

Measured on 2026-11-13, an Apple silicon Mac with 36 GB, `float32=True`, `orders="reverse"`. Set: 3 short calls,
6 questions each — 3 noul, 2 choice, 1 score, 18 in all. Tests: `DECISION_GEM_TEST_MODEL=<repo> pytest -m models`.

| Model | `-m models` tests | Load | 18 questions |
|---|---|---|---|
| `mlx-community/gemma-4-e2b-it-4bit` | 6 of 6 | 1.4 s | 1.7 s |
| `mlx-community/gemma-4-e4b-it-4bit` | 6 of 6 | 1.8 s | 3.1 s |
| `mlx-community/Qwen3-14B-4bit` | 6 of 6 | 3.1 s | 9.2 s |
| `mlx-community/gemma-4-26b-a4b-it-4bit` (default) | 6 of 6 | 4.6 s | 6.7 s |

Gemma 26B is an MoE (~4B active parameters), so it answers faster than the dense Qwen3-14B.

## Install and run

Apple silicon only (MLX). The model must already be in the HF cache, or run with `--online` / `offline=False`.

```bash
pip install decision-gem                  # or from source: pip install -e ".[test]"
python -m decision_gem request.json       # or: decision-gem request.json; - reads stdin
python -m decision_gem request.json --calibration cal.json --model <repo or folder>
pytest                                   # tests without the model; pytest -m models — on the model
```

```python
from decision_gem import DecisionGem

email = {"from": '"Secure Bank" <security@secure-bank-verify.top>',   # the state: text or a JSON object
         "subject": "Your account has been limited!",
         "body": "Dear customer, we noticed unusual activity. Verify your identity within 24 hours: "
                 "http://secure-bank-verify.top/login or your account will be permanently suspended."}
qs = {"spam":    {"type": "noul", "instructions": "Is this email spam or phishing?"},
      "kind":    {"type": "choice", "instructions": "What kind of email is it?",
                  "criteria": {"phishing": "tries to steal logins or payment data", "ads": "unsolicited advertising",
                               "scam": "promises money or prizes", "normal": "an ordinary email"}},
      "urgency": {"type": "score", "instructions": "How hard does the email push the reader to act right away?",
                  "criteria": ["no pressure", "some pressure", "strong pressure"]}}

gem = DecisionGem()                       # mlx-community/gemma-4-26b-a4b-it-4bit from the local HF cache
gem({"state": email, "questions": qs})
```

The answer (Gemma 26B, 474 input tokens; `legend` and `probabilities` cut):

```json
{"spam":    {"type": "noul", "noul": 1.0, "orders_agree": true},
 "kind":    {"type": "choice", "choice": "phishing", "confidence": 1.0, "orders_agree": true},
 "urgency": {"type": "score", "score": 3.0, "confidence": 1.0, "orders_agree": true}}
```

On an ordinary email from a colleague about meeting notes the same questions give `spam` 0.0, `kind` `normal`,
`urgency` 1.01.

Several states at once (see "Keyed states"):

```python
with DecisionGem() as gem:
    gem.prefill("mail-1", email1)            # number of tokens; ≈0.2 MiB of memory per token until forget / unload
    gem.prefill("mail-2", email2)
    gem.decide(key="mail-2", questions=qs)   # the same answer as decide(email2, qs), without reading again
    gem({"key": "mail-1", "questions": qs})  # a Jev request with key instead of state
    gem.save("mail-1", "mail-1.safetensors") # to disk; load() reads it back in ~0.02 s
    gem.forget("mail-1")
    gem.load("mail-1", "mail-1.safetensors") # only a file saved with the same model
```

The full contract is also in the docstring of `decision_gem/core.py`.
