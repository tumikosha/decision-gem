"""DecisionGem: a local LLM (Gemma, MLX) read as a decision model, with the request / response contract of TypeSafe's
Jev (https://developers.cloudflare.com/ai/models/typesafe/jev/). Runs fully offline.

Request: {"state": text or JSON object, "questions": {id: question}}; a question is
- noul:   {"type": "noul", "instructions": ..., "criteria": {"true": ..., "false": ...}}  (criteria optional)
- choice: {"type": "choice", "instructions": ..., "criteria": {option: description} or [option, ...]}  (2..26)
- score:  {"type": "score", "instructions": ..., "criteria": {value: description} or [description, ...]}  (2..10
          levels, in order; list levels are worth 1..N)
Response: {"model", "answers": {id: answer}, "usage": {"input_tokens", "output_tokens": 0}}; an answer is
- noul:   {"type", "noul": P(true)}
- choice: {"type", "choice": option, "confidence", "probabilities": {option: p}}
- score:  {"type", "score": sum of p * value, "confidence", "legend": {value: description}, "probabilities"}
plus, beyond Jev, "orders_agree": whether every option order (below) picks the same answer.

How: nothing is generated. The state goes first in the prompt and is read once into a KV cache; every question is
appended to a copy of that cache, its options listed under the letters A, B, C …, and the logits of those letters
at the first answer position are read. Every question is read in several option orders and the log-probabilities
of each option are averaged, so a preference for a letter or a position cancels out. orders="reverse" (default): twice,
in order and reversed — 2 passes, but with 3+ options the middle stays in the middle, so a pull to or from the middle
letters is not cancelled. orders="cyclic": every rotation, each option once under every letter — n passes.
orders="single": in order only — 1 pass, the fastest, nothing cancelled, and orders_agree is None.
confidence is the probability of the answer. Calibration (Platt scaling on hand-labelled calls) per question id:
noul p = sigmoid(a * logit + b) from [a, b]; choice / score probabilities = softmax(logprob / T) from T.

Beyond Jev, generate() runs the model as an ordinary chat model: a prompt (after a state, or a kept state's key, or
alone) and the generated answer; a request with "prompt" instead of "questions" does the same.

Everything but DecisionGem itself is pure Python; the model needs mlx and mlx-lm (Apple silicon).
"""
import copy
import json
import math
import os
import re
import string

MODEL = "mlx-community/gemma-4-26b-a4b-it-4bit"
LETTERS = string.ascii_uppercase
_QUESTION = "\x00question\x00"   # where the question goes in the rendered prompt
STATE_INTRO = "Read the state below, then answer the question after it.\n\n<state>\n"
STATE_CLOSE = "\n</state>\n\n"   # what follows the state text; an open state leaves it out of the cache
KINDS = ("noul", "choice", "score")
ORDERS = ("reverse", "cyclic", "single")


def options_of(question):
    """[(key, description)] of a question's options, in order: noul true / false, choice options, score levels."""
    kind, criteria = question.get("type"), question.get("criteria")
    if kind not in KINDS:
        raise ValueError(f"question type must be one of {', '.join(KINDS)}, got {kind!r}")
    if kind == "noul":
        criteria = criteria or {}
        return [("true", criteria.get("true", "yes")), ("false", criteria.get("false", "no"))]
    if isinstance(criteria, dict):
        items = [(str(k), str(v)) for k, v in criteria.items()]
    elif isinstance(criteria, list):
        items = [(str(x), str(x)) for x in criteria] if kind == "choice" else \
                [(str(i + 1), str(x)) for i, x in enumerate(criteria)]
    else:
        raise ValueError(f"{kind} question needs criteria")
    low, high = (2, len(LETTERS)) if kind == "choice" else (2, 10)
    if not low <= len(items) <= high:
        raise ValueError(f"{kind} question needs {low}..{high} options, got {len(items)}")
    return items


def question_text(question, items):
    """The question as appended after the state, its options under the letters A, B, C …"""
    lines = [f"Question: {question.get('instructions', '').strip()}"]
    if question["type"] == "score":
        lines.append("The options are levels of an ordered scale.")
    for letter, (key, desc) in zip(LETTERS, items):
        label = key if desc == key else f"{key} ({desc})" if question["type"] != "noul" else f"{key}: {desc}"
        lines.append(f"{letter}: {label}")
    lines.append(f"Answer with one letter: {', '.join(LETTERS[:len(items)])}.")
    return "\n".join(lines)


def orders_of(n, how="reverse"):
    """The option orders a question of n options is read in, each a list of option indexes by letter."""
    if how == "reverse":
        return [list(range(n)), list(range(n))[::-1]]
    if how == "cyclic":
        return [[(s + i) % n for i in range(n)] for s in range(n)]
    if how == "single":
        return [list(range(n))]
    raise ValueError(f"orders must be one of {', '.join(ORDERS)}, got {how!r}")


def split_open(prompt):
    """(head, close) of a rendered prompt: head ends right after the state text, close is what follows it."""
    at = prompt.find(STATE_CLOSE)
    if at < 0:
        raise ValueError("the prompt has no closing </state>")
    return prompt[:at], STATE_CLOSE


def float32_matmuls(mx):
    """Whether this process's MLX multiplies float32 matrices on the GPU in float32: the GPU's product against the
    CPU's. On GPUs with neural accelerators (M5) MLX multiplies them in TF32 (relative error ~5e-4, float32 ~1e-6)
    from 128 rows up (dense: from 2), unless MLX_ENABLE_TF32=0 was set before the process's first matmul."""
    a = mx.random.normal((256, 256), key=mx.random.key(0))
    gpu, cpu = a @ a.T, mx.matmul(a, a.T, stream=mx.cpu)
    return (mx.abs(gpu - cpu).max() / mx.abs(cpu).max()).item() < 1e-5


def letter_ids(tok):
    """Token ids of the letters A..Z as the first token of the answer. Each letter is appended to the tail of the
    chat prompt (what follows the question) and must add exactly one token there. A tokenizer that splits a letter or
    merges it with the tail cannot be read this way: ValueError, before any question runs."""
    if not getattr(tok, "chat_template", None):
        raise ValueError("the model has no chat template: decision-gem needs an instruction-tuned (chat) model")
    prompt = tok.apply_chat_template([{"role": "user", "content": _QUESTION}], add_generation_prompt=True,
                                     tokenize=False, enable_thinking=False)
    tail = prompt.split(_QUESTION)[1]
    base = tok.encode(tail, add_special_tokens=False)
    ids = []
    for x in LETTERS:
        full = tok.encode(tail + x, add_special_tokens=False)
        if full[:len(base)] != base or len(full) != len(base) + 1:
            raise ValueError(f"letter {x!r} after the chat prompt is not one token in this tokenizer "
                             f"({len(full) - len(base)} tokens): the answer cannot be read from letter logits")
        ids.append(full[-1])
    if len(set(ids)) != len(ids):
        raise ValueError("two letters share a token in this tokenizer")
    return ids


def softmax(xs):
    top = max(xs)
    es = [math.exp(x - top) for x in xs]
    return [e / sum(es) for e in es]


def answer(question, items, logprobs, agree, cal=None, raw=False):
    """The Jev answer from the averaged log-probabilities of the options; cal is the question's calibration.
    raw: add "logprobs" {option: log-probability before calibration}, the input of fit_calibration()."""
    kind = question["type"]
    extra = {"logprobs": {k: round(x, 6) for (k, _), x in zip(items, logprobs)}} if raw else {}
    if kind == "noul":
        a, b = cal or (1.0, 0.0)
        p = 1 / (1 + math.exp(-(a * (logprobs[0] - logprobs[1]) + b)))
        return {"type": kind, "noul": round(p, 4), "orders_agree": agree, **extra}
    probs = softmax([x / (cal or 1.0) for x in logprobs])
    best = max(range(len(items)), key=lambda i: probs[i])
    out = {"type": kind}
    if kind == "choice":
        out["choice"] = items[best][0]
    else:
        values = [float(k) if re.fullmatch(r"-?\d+(\.\d+)?", k) else i + 1 for i, (k, _) in enumerate(items)]
        out["score"] = round(sum(p * v for p, v in zip(probs, values)), 4)
        out["legend"] = {k: d for k, d in items}
    out.update(confidence=round(probs[best], 4),
               probabilities={k: round(p, 4) for (k, _), p in zip(items, probs)}, orders_agree=agree, **extra)
    return out


def _fit_platt(xs, ys):
    """[a, b] of p = sigmoid(a * x + b) by Newton's method on Platt's smoothed targets, which keep a and b finite
    on separable labels."""
    pos = sum(ys)
    neg = len(ys) - pos
    hi, lo = (pos + 1) / (pos + 2), 1 / (neg + 2)
    ts = [hi if y else lo for y in ys]

    def loss(a, b):
        out = 0.0
        for x, t in zip(xs, ts):
            z = a * x + b
            out += max(z, 0) + math.log1p(math.exp(-abs(z))) - t * z   # -t·log p - (1-t)·log(1-p)
        return out

    a, b = 0.0, math.log((pos + 1) / (neg + 1))
    for _ in range(100):
        ga = gb = haa = hab = hbb = 0.0
        for x, t in zip(xs, ts):
            p = 1 / (1 + math.exp(-(a * x + b))) if a * x + b > -700 else 0.0
            d, w = p - t, max(p * (1 - p), 1e-12)
            ga, gb, haa, hab, hbb = ga + d * x, gb + d, haa + w * x * x, hab + w * x, hbb + w
        det = haa * hbb - hab * hab + 1e-12
        da, db = (hbb * ga - hab * gb) / det, (haa * gb - hab * ga) / det
        step, before = 1.0, loss(a, b)
        while step > 1e-10 and loss(a - step * da, b - step * db) > before + 1e-12:
            step /= 2
        a, b = a - step * da, b - step * db
        if abs(step * da) < 1e-9 and abs(step * db) < 1e-9:
            break
    return [round(a, 4), round(b, 4)]


def _fit_temperature(rows):
    """T of softmax(logprob / T) with the least log-loss: the loss is convex in 1/T, so its derivative is bisected.
    rows: [(logprobs, index of the label)]."""
    def slope(beta):   # d loss / d beta = sum of E[logprob] under softmax(beta·logprob) - logprob of the label
        out = 0.0
        for lp, y in rows:
            ws = [beta * x for x in lp]
            top = max(ws)
            es = [math.exp(w - top) for w in ws]
            out += sum(e * x for e, x in zip(es, lp)) / sum(es) - lp[y]
        return out
    lo, hi = math.log(1e-3), math.log(100.0)   # T from 0.01 to 1000
    if slope(math.exp(hi)) < 0:
        return round(1 / math.exp(hi), 4)
    if slope(math.exp(lo)) > 0:
        return round(1 / math.exp(lo), 4)
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if slope(math.exp(mid)) < 0 else (lo, mid)
    return round(1 / math.exp((lo + hi) / 2), 4)


def fit_calibration(answers, labels):
    """The calibration {question id: [a, b] (noul) or T (choice / score)} fitted on labelled answers.
    answers: the "answers" of responses decided with raw=True, one per state; labels: a {question id: label} per
    state, in the same order. A label is True / False for noul, the option for choice, the level (value) for score;
    a question with no label in a state is skipped there. Pass the result as DecisionGem(calibration=...) or in the
    --calibration file."""
    if len(answers) != len(labels):
        raise ValueError(f"answers and labels must have the same length, got {len(answers)} and {len(labels)}")
    rows = {}
    for got, want in zip(answers, labels):
        for qid, label in want.items():
            a = got.get(qid)
            if a is None:
                continue
            if "logprobs" not in a:
                raise ValueError(f"answer {qid!r} has no logprobs: decide with raw=True")
            rows.setdefault(qid, (a["type"], []))[1].append((a["logprobs"], label))
    out = {}
    for qid, (kind, pairs) in rows.items():
        if kind == "noul":
            xs = [lp["true"] - lp["false"] for lp, _ in pairs]
            ys = []
            for _, label in pairs:
                if label not in (True, False, "true", "false"):
                    raise ValueError(f"noul label of {qid!r} must be True / False, got {label!r}")
                ys.append(label in (True, "true"))
            if all(ys) or not any(ys):
                raise ValueError(f"noul {qid!r} needs both labels, true and false")
            out[qid] = _fit_platt(xs, ys)
        else:
            fit = []
            for lp, label in pairs:
                keys = list(lp)
                if str(label) not in keys:
                    raise ValueError(f"label {label!r} of {qid!r} is not one of {keys}")
                fit.append(([lp[k] for k in keys], keys.index(str(label))))
            out[qid] = _fit_temperature(fit)
    return out


class DecisionGem:
    def __init__(self, repo=MODEL, calibration=None, offline=True, cache_limit=0, orders="reverse", float32=True):
        """repo: a Hugging Face repo id or a local model folder. offline: load only what is in the local HF cache.
        orders: the option orders every question is read in, "reverse", "cyclic" or "single" (see the module doc).
        cache_limit: bytes of freed buffers MLX may keep for reuse (mx.set_cache_limit), None to leave MLX's own.
        MLX's limit is the process's: unload() puts back the one there was.
        float32: compute in float32 (the bf16 norms and quantization scales cast up; +1.5 GiB, KV cache 2x, reads
        about 1.3x slower). In bf16 a read's rounding depends on how many tokens it holds and flips the MoE router's
        top-8 on near-ties, so one text read whole, in chunks or with </state> cut elsewhere gave noul 0.14..0.65
        (1590 tokens, 10.19); in float32 0.568..0.574. extend() equals one read only in float32.
        float32 also needs float32 matmuls: on an M5 MLX multiplies in TF32 by default, and the same 1590 tokens
        read whole or in chunks gave 0.546 vs 0.347 (10.19). So the gem sets MLX_ENABLE_TF32=0; MLX reads it at the
        process's first matmul, and a RuntimeError says so if one ran before with TF32 on. Cost on the M5 Max: reads
        2x slower than in TF32 (1594 tokens 1.6 s vs 0.8 s, a question 0.30 s vs 0.18 s)."""
        import mlx.core as mx
        from mlx_lm import load
        orders_of(2, orders)   # a bad value fails before the model loads
        self.orders = orders
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        if float32:
            os.environ.setdefault("MLX_ENABLE_TF32", "0")
            if not float32_matmuls(mx):
                raise RuntimeError("MLX multiplies float32 in TF32 in this process: set MLX_ENABLE_TF32=0 before the "
                                   "first MLX matmul (MLX reads it once), or pass float32=False")
        path = repo
        if offline and not os.path.isdir(repo):
            from huggingface_hub import snapshot_download
            path = snapshot_download(repo, local_files_only=True)
        self.mx, self.repo, self.float32 = mx, repo, float32
        # MLX keeps freed buffers for reuse; the cache copies per question piled 13 GiB of them on the 13 GiB model
        # (peak footprint 27.8 GiB on the 5 longest calls, 13.10). With no buffer cache: 17.2 GiB, same time.
        self._cache_limit_before = None if cache_limit is None else mx.set_cache_limit(cache_limit)
        self.model, self.tok = load(path)
        if float32:
            from mlx.utils import tree_map
            self.model.update(tree_map(lambda p: p.astype(mx.float32) if p.dtype == mx.bfloat16 else p,
                                       self.model.parameters()))
            mx.eval(self.model.parameters())
        self.letter_ids = letter_ids(self.tok)
        self.calibration = calibration or {}
        self.states = {}   # key → a state read by prefill() / load(): {"cache", "tokens", "tail", "close"}

    @property
    def dtype(self):
        """The compute dtype, saved with a state: load() takes a state only into a gem of the same dtype."""
        return "float32" if self.float32 else "bfloat16"

    def unload(self):
        """Free the model's memory without ending the process (one big model at a time); decide() fails after."""
        import gc
        if self.model is None:
            return
        self.model = self.tok = None
        self.states = {}
        gc.collect()
        self.mx.clear_cache()
        if self._cache_limit_before is not None:
            self.mx.set_cache_limit(self._cache_limit_before)
            self._cache_limit_before = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.unload()

    def _encode(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    def _read(self, state, open=False):
        """The state read into a fresh KV cache: (cache, its tokens, the prompt's tail after the question, close).
        open: the cache stops after the state text and close is what decide() puts before each question; a closed
        state has the closing tag in its cache and close is ''."""
        from mlx_lm.models.cache import make_prompt_cache
        if self.model is None:
            raise RuntimeError("the model is unloaded")
        if not isinstance(state, str):
            state = json.dumps(state, ensure_ascii=False, indent=1)
        content = f"{STATE_INTRO}{state.strip()}{STATE_CLOSE}{_QUESTION}"
        prompt = self.tok.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                              tokenize=False, enable_thinking=False)
        head, tail = prompt.split(_QUESTION)
        close = ""
        if open:
            head, close = split_open(head)
        prefix = self._encode(head)
        cache = make_prompt_cache(self.model)
        self.mx.eval(self.model(self.mx.array(prefix)[None], cache=cache))
        return cache, len(prefix), tail, close

    def prefill(self, key, state, open=False):
        """Read a state once and keep it under `key`, for decide(key=...) later; the number of its tokens.
        open: stop after the state text, so extend(key, text) can go on; decide() adds the closing tag per question.
        Memory: about 0.43 MiB per token in float32 (691 MiB for 1590 tokens, 10.19; 0.2 MiB in bf16) until
        forget(key) or unload()."""
        cache, tokens, tail, close = self._read(state, open)
        self.states[key] = {"cache": cache, "tokens": tokens, "tail": tail, "close": close}
        return tokens

    def extend(self, key, text):
        """Read more text into an open state's cache, for good; the number of tokens added. ValueError on a closed one.
        Seam rule: extend = one read only if the text starts at a token boundary. The leading space or newline goes in
        the new text, a word is never split, and the text never ends in whitespace (ValueError): "hello. " + "Operator"
        tokenizes differently from "hello. Operator". prefill() strips the state, so it never ends in whitespace."""
        if self.model is None:
            raise RuntimeError("the model is unloaded")
        kept = self.states[key]
        if not kept["close"]:
            raise ValueError(f"state {key!r} is closed: prefill it with open=True")
        if text != text.rstrip():
            raise ValueError("extend text must not end in whitespace: put it at the start of the next text")
        ids = self._encode(text)
        if ids:
            self.mx.eval(self.model(self.mx.array(ids)[None], cache=kept["cache"]))
        kept["tokens"] += len(ids)
        return len(ids)

    def forget(self, key):
        """Drop a kept state; its memory goes back to the system."""
        del self.states[key]

    def save(self, key, path):
        """Write a kept state to a file (.safetensors); load() reads it back in about 0.02 s instead of a prefill.
        The file is the call's text in other form: keep it where the text may be."""
        from mlx_lm.models.cache import save_prompt_cache
        kept = self.states[key]
        meta = {"tokens": str(kept["tokens"]), "tail": kept["tail"], "close": kept["close"], "repo": self.repo,
                "dtype": self.dtype}
        save_prompt_cache(path, kept["cache"], meta)

    def load(self, key, path):
        """Keep the state saved in `path` under `key`; it must have been saved with the same model and dtype."""
        from mlx_lm.models.cache import load_prompt_cache
        cache, meta = load_prompt_cache(path, return_metadata=True)
        if meta.get("repo") != self.repo:
            raise ValueError(f"{path} was saved with {meta.get('repo')}, not {self.repo}")
        saved = meta.get("dtype", "bfloat16")   # no field: saved before float32 became the default
        if saved != self.dtype:
            raise ValueError(f"{path} was saved in {saved}, not {self.dtype}: a mixed-dtype cache reads wrong")
        self.states[key] = {"cache": cache, "tokens": int(meta["tokens"]), "tail": meta["tail"],
                            "close": meta.get("close", "")}
        return self.states[key]["tokens"]

    def decide(self, state=None, questions=None, *, key=None, orders=None, raw=False):
        """The Jev response for one request (see the module doc): on `state`, read now and dropped after, or on the
        state kept under `key` by prefill() / load() — exactly one of the two. orders: this request's, else the gem's.
        raw: every answer also carries "logprobs", the option log-probabilities before calibration (fit_calibration)."""
        if (state is None) == (key is None):
            raise ValueError("give either state or key")
        if orders is None:
            orders = self.orders
        orders_of(2, orders)
        options = {qid: options_of(q) for qid, q in (questions or {}).items()}  # a bad question fails before the model
        if key is None:
            cache, used, tail, close = self._read(state)
        else:
            if self.model is None:
                raise RuntimeError("the model is unloaded")
            kept = self.states[key]
            cache, used, tail, close = kept["cache"], kept["tokens"], kept["tail"], kept["close"]
        answers = {}
        for qid, question in (questions or {}).items():
            items = options[qid]
            n = len(items)
            logprobs, picks = [0.0] * n, []
            read = orders_of(n, orders)
            for order in read:
                suffix = self._encode(close + question_text(question, [items[i] for i in order]) + tail)
                used += len(suffix)
                logits = self.model(self.mx.array(suffix)[None], cache=copy.deepcopy(cache))[0, -1]
                z = [logits[t].item() for t in self.letter_ids[:n]]
                lse = max(z) + math.log(sum(math.exp(x - max(z)) for x in z))
                for pos, i in enumerate(order):
                    logprobs[i] += (z[pos] - lse) / len(read)
                picks.append(order[max(range(n), key=lambda pos: z[pos])])
            answers[qid] = answer(question, items, logprobs, len(set(picks)) == 1 if len(read) > 1 else None,
                                   self.calibration.get(qid), raw)
        return {"model": f"decision-gem/{self.repo.rstrip('/').split('/')[-1]}", "answers": answers,
                "usage": {"input_tokens": used, "output_tokens": 0}}

    def generate(self, prompt, state=None, *, key=None, max_tokens=512, temperature=0.0):
        """The model as an ordinary chat model: its answer to `prompt`, generated. With a `state` (read now) or the
        state kept under `key` (prefill / load — not read again; the answer is generated on a copy of its cache, as
        decide() asks), the prompt follows the state in the same user turn, where a question would; with neither, the
        prompt is the whole user turn. temperature 0: greedy, the same prompt gives the same text.
        {"model", "text", "finish_reason": "stop" | "length", "usage": {"input_tokens", "output_tokens"}}."""
        from mlx_lm import stream_generate
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if state is not None and key is not None:
            raise ValueError("give state or key, not both")
        if self.model is None:
            raise RuntimeError("the model is unloaded")
        if key is not None:
            kept = self.states[key]
            cache, used = copy.deepcopy(kept["cache"]), kept["tokens"]
            tokens = self._encode(kept["close"] + prompt.strip() + kept["tail"])   # an open state closes here
        elif state is not None:
            cache, used, tail, _ = self._read(state)
            tokens = self._encode(prompt.strip() + tail)
        else:
            text = self.tok.apply_chat_template([{"role": "user", "content": prompt.strip()}],
                                                add_generation_prompt=True, tokenize=False, enable_thinking=False)
            cache, used, tokens = make_prompt_cache(self.model), 0, self._encode(text)
        parts, last = [], None
        for last in stream_generate(self.model, self.tok, self.mx.array(tokens), max_tokens=max_tokens,
                                    sampler=make_sampler(temp=temperature), prompt_cache=cache):
            parts.append(last.text)
        return {"model": f"decision-gem/{self.repo.rstrip('/').split('/')[-1]}", "text": "".join(parts).strip(),
                "finish_reason": last.finish_reason if last else "length",
                "usage": {"input_tokens": used + len(tokens), "output_tokens": last.generation_tokens if last else 0}}

    def __call__(self, request):
        """decide() on a whole Jev request {"state", "questions"}, or {"key", "questions"} on a kept state ("raw": true
        adds the uncalibrated "logprobs");
        generate() on {"prompt", "state" or "key" (optional), "max_tokens", "temperature"}."""
        if "prompt" in request:
            if "questions" in request:
                raise ValueError("a request has questions or a prompt, not both")
            return self.generate(request["prompt"], request.get("state"), key=request.get("key"),
                                 max_tokens=request.get("max_tokens", 512),
                                 temperature=request.get("temperature", 0.0))
        return self.decide(request.get("state"), request["questions"], key=request.get("key"),
                           raw=request.get("raw", False))
