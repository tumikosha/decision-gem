"""The pure parts of decision_gem: no model is loaded."""
import math
import types

import pytest

from decision_gem import answer, options_of, orders_of, question_text, split_open


def test_noul_default_and_custom_criteria():
    assert options_of({"type": "noul"}) == [("true", "yes"), ("false", "no")]
    q = {"type": "noul", "criteria": {"true": "booked", "false": "not booked"}}
    assert options_of(q) == [("true", "booked"), ("false", "not booked")]


def test_choice_list_and_dict():
    assert options_of({"type": "choice", "criteria": ["a", "b"]}) == [("a", "a"), ("b", "b")]
    assert options_of({"type": "choice", "criteria": {"x": "first", "y": "second"}}) == [("x", "first"), ("y", "second")]


def test_score_list_levels_are_1_to_n():
    assert options_of({"type": "score", "criteria": ["bad", "ok", "good"]}) == [("1", "bad"), ("2", "ok"), ("3", "good")]


@pytest.mark.parametrize("q", [
    {"type": "choice", "criteria": ["only"]},
    {"type": "choice"},
    {"type": "score", "criteria": [str(i) for i in range(11)]},
    {"type": "yesno"},
])
def test_bad_questions(q):
    with pytest.raises(ValueError):
        options_of(q)


def test_question_text_letters():
    q = {"type": "choice", "instructions": " Who? ", "criteria": {"x": "first", "y": "y"}}
    assert question_text(q, options_of(q)) == "Question: Who?\nA: x (first)\nB: y\nAnswer with one letter: A, B."


def test_noul_answer_with_calibration():
    q = {"type": "noul"}
    out = answer(q, options_of(q), [math.log(0.8), math.log(0.2)], True)
    assert out == {"type": "noul", "noul": 0.8, "orders_agree": True}
    out = answer(q, options_of(q), [0.0, 0.0], False, cal=[1.0, 2.0])
    assert out["noul"] == round(1 / (1 + math.exp(-2)), 4)


def test_choice_answer():
    q = {"type": "choice", "criteria": ["a", "b", "c"]}
    out = answer(q, options_of(q), [math.log(0.2), math.log(0.7), math.log(0.1)], True)
    assert out["choice"] == "b" and out["confidence"] == 0.7
    assert out["probabilities"] == {"a": 0.2, "b": 0.7, "c": 0.1}


def test_score_is_expected_value():
    q = {"type": "score", "criteria": {"0": "none", "10": "full"}}
    out = answer(q, options_of(q), [math.log(0.25), math.log(0.75)], True)
    assert out["score"] == 7.5 and out["legend"] == {"0": "none", "10": "full"}


def test_decide_wants_state_or_key():
    from decision_gem import DecisionGem
    gem = object.__new__(DecisionGem)        # no model: the arguments fail first
    for args, kw in (((), {}), (("text", {}), {"key": "a"})):
        with pytest.raises(ValueError):
            gem.decide(*args, **kw)


def test_generate_checks_its_arguments_before_the_model():
    from decision_gem import DecisionGem
    gem = object.__new__(DecisionGem)        # no model: the arguments fail first
    for args, kw in ((("",), {}), (("Say hi", "text"), {"key": "a"})):
        with pytest.raises(ValueError):
            gem.generate(*args, **kw)
    with pytest.raises(ValueError):
        gem({"prompt": "Say hi", "questions": {}})


@pytest.mark.models
def test_generate_on_a_kept_state_equals_on_the_state():
    import os
    from decision_gem import MODEL, DecisionGem
    state = "SPEAKER_00: Добрый день, клиника.\nSPEAKER_01: Хочу записаться к кардиологу на четверг."
    prompt = "Which doctor does the caller want? Answer with one word."
    with DecisionGem(os.environ.get("DECISION_GEM_TEST_MODEL", MODEL)) as gem:
        once = gem.generate(prompt, state, max_tokens=8)
        gem.prefill("c", state)
        qs = {"q": {"type": "noul", "instructions": "Does the caller want to book a visit?"}}
        before = gem.decide(key="c", questions=qs)
        kept = gem.generate(prompt, key="c", max_tokens=8)
        alone = gem({"prompt": "Say the word yes and nothing else.", "max_tokens": 4})
        after = gem.decide(key="c", questions=qs)
    assert once["text"] == kept["text"] and any(w in once["text"].lower() for w in ("cardiolog", "кардиолог"))
    assert once["usage"]["input_tokens"] == kept["usage"]["input_tokens"] and once["usage"]["output_tokens"] <= 8
    assert "yes" in alone["text"].lower()
    assert after == before and after["answers"]["q"]["noul"] > 0.5      # generate() left the kept state as it was


def test_orders_reverse_and_cyclic():
    assert orders_of(3) == [[0, 1, 2], [2, 1, 0]]
    assert orders_of(3, "cyclic") == [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
    assert orders_of(2, "cyclic") == orders_of(2)            # two options: both ways are the same two orders
    for n in (3, 4, 7):                                       # cyclic: every option once under every letter
        read = orders_of(n, "cyclic")
        assert all(sorted(o[pos] for o in read) == list(range(n)) for pos in range(n))
    with pytest.raises(ValueError):
        orders_of(3, "shuffle")


class _Box:
    def __init__(self, x):
        self.x = x

    def __getitem__(self, _):
        return self

    def item(self):
        return self.x


def _biased_gem(orders="reverse"):
    """A DecisionGem with no model: letter logits = 1 for the option «good», + 3 for the letter B, whatever stands there."""
    from decision_gem import DecisionGem
    gem = object.__new__(DecisionGem)
    gem.mx, gem.repo, gem.orders, gem.calibration = types.SimpleNamespace(array=_Box), "fake", orders, {}
    gem.letter_ids = list(range(26))
    gem._encode = lambda text: text
    gem._read = lambda state: (None, 0, "", "")

    def model(box, cache):
        letters = {line[0]: line[3:] for line in box.x.splitlines() if line[1:3] == ": "}
        row = [_Box((letters.get(chr(65 + t)) == "good") + 3.0 * (t == 1)) for t in range(26)]
        return {(0, -1): row}
    gem.model = model
    return gem


def test_cyclic_cancels_a_pull_to_the_middle_letter():
    qs = {"q": {"type": "choice", "criteria": ["good", "mid", "bad"]}}
    rev = _biased_gem().decide("state", qs)["answers"]["q"]
    assert rev["choice"] == "mid" and rev["orders_agree"]          # mid is under B both times: the pull wins
    cyc = _biased_gem().decide("state", qs, orders="cyclic")["answers"]["q"]
    assert cyc["choice"] == "good" and not cyc["orders_agree"]     # every option under B once: the pull cancels
    assert cyc["probabilities"]["mid"] == cyc["probabilities"]["bad"]


def test_orders_fall_back_and_fail():
    qs = {"q": {"type": "choice", "criteria": ["good", "mid", "bad"]}}
    assert _biased_gem("cyclic").decide("state", qs)["answers"]["q"]["choice"] == "good"
    with pytest.raises(ValueError):
        _biased_gem().decide("state", qs, orders="")


def test_split_open_cuts_before_closing_tag():
    head, close = split_open("<u>Read…\n\n<state>\nA: hi\n</state>\n\n\x00question\x00<tail>")
    assert head == "<u>Read…\n\n<state>\nA: hi" and close == "\n</state>\n\n"


def test_split_open_needs_a_closing_tag():
    with pytest.raises(ValueError):
        split_open("<state>\nA: hi")


def test_extend_rejects_trailing_whitespace():
    from decision_gem import DecisionGem
    gem = object.__new__(DecisionGem)
    gem.model, gem.states = object(), {"a": {"close": "\n</state>\n\n"}}
    for text in ("hello. ", "hello.\n", " "):
        with pytest.raises(ValueError):
            gem.extend("a", text)


def _stub_gem(float32):
    from decision_gem import DecisionGem
    gem = object.__new__(DecisionGem)
    gem.model, gem.repo, gem.float32, gem.states = object(), "repo/x", float32, {}
    return gem


def _tiny_state():
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache
    c = KVCache()
    c.update_and_fetch(mx.zeros((1, 1, 2, 4)), mx.zeros((1, 1, 2, 4)))
    return {"cache": [c], "tokens": 2, "tail": "<t>", "close": ""}


@pytest.mark.parametrize("saved, loaded", [(True, False), (False, True)])
def test_load_rejects_a_state_of_the_other_dtype(tmp_path, saved, loaded):
    gem = _stub_gem(saved)
    gem.states["a"] = _tiny_state()
    gem.save("a", str(tmp_path / "a.safetensors"))
    with pytest.raises(ValueError):
        _stub_gem(loaded).load("b", str(tmp_path / "a.safetensors"))


@pytest.mark.parametrize("float32", [True, False])
def test_load_takes_a_state_of_the_same_dtype(tmp_path, float32):
    gem = _stub_gem(float32)
    gem.states["a"] = _tiny_state()
    gem.save("a", str(tmp_path / "a.safetensors"))
    assert _stub_gem(float32).load("b", str(tmp_path / "a.safetensors")) == 2


def test_a_state_without_dtype_is_bf16(tmp_path):
    from mlx_lm.models.cache import save_prompt_cache
    save_prompt_cache(str(tmp_path / "old.safetensors"), _tiny_state()["cache"],
                      {"tokens": "2", "tail": "<t>", "close": "", "repo": "repo/x"})
    assert _stub_gem(False).load("b", str(tmp_path / "old.safetensors")) == 2
    with pytest.raises(ValueError):
        _stub_gem(True).load("c", str(tmp_path / "old.safetensors"))


@pytest.mark.parametrize("float32", [True, False])
def test_float32_casts_bf16_params_only_when_asked(monkeypatch, tmp_path, float32):
    import mlx.core as mx
    import mlx_lm
    from decision_gem import DecisionGem

    class Model:   # a stand-in: one bf16 and one uint32 parameter
        params = {"norm": mx.ones((2,), mx.bfloat16), "w": mx.ones((2,), mx.uint32)}
        def parameters(self):
            return self.params
        def update(self, params):
            self.params = params

    class Tok:
        chat_template = "x"
        def apply_chat_template(self, messages, **kw):
            return messages[0]["content"] + "<t>"
        def encode(self, text, add_special_tokens=False):
            return [ord(c) for c in text]

    monkeypatch.setattr(mlx_lm, "load", lambda path: (Model(), Tok()))
    gem = DecisionGem(str(tmp_path), cache_limit=None, float32=float32)
    assert gem.model.params["norm"].dtype == (mx.float32 if float32 else mx.bfloat16)
    assert gem.model.params["w"].dtype == mx.uint32
    assert gem.dtype == ("float32" if float32 else "bfloat16")


def _in_fresh_process(code, **env):
    import os
    import subprocess
    import sys
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={k: v for k, v in {**os.environ, **env}.items() if v is not None})
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_float32_matmuls_without_tf32():
    assert _in_fresh_process("import mlx.core as mx; from decision_gem.core import float32_matmuls;"
                             "print(float32_matmuls(mx))", MLX_ENABLE_TF32="0") == "True"


def test_a_float32_gem_fails_once_tf32_is_on():
    # MLX reads MLX_ENABLE_TF32 at its first matmul: a matmul before the gem leaves TF32 on, where the GPU has it (M5)
    code = """
import mlx.core as mx, mlx_lm
from decision_gem import DecisionGem
from decision_gem.core import float32_matmuls
mx.eval(mx.ones((256, 256)) @ mx.ones((256, 256)))
if float32_matmuls(mx):
    print("no tf32")
else:
    mlx_lm.load = lambda path: (_ for _ in ()).throw(AssertionError("the model loads after the check"))
    try:
        DecisionGem(".", cache_limit=None)
    except RuntimeError as e:
        print("raised" if "MLX_ENABLE_TF32" in str(e) else e)
"""
    got = _in_fresh_process(code, MLX_ENABLE_TF32=None)
    if got == "no tf32":
        pytest.skip("this GPU multiplies float32 in float32 anyway")
    assert got == "raised"


class _Tok:
    """A tokenizer stand-in: one token per character, except the pairs in `merge`; the chat tail is "<t>"."""
    chat_template = "x"

    def __init__(self, merge=()):
        self.merge = merge

    def apply_chat_template(self, messages, **kw):
        return "<u>" + messages[0]["content"] + "<t>"

    def encode(self, text, add_special_tokens=False):
        out, i = [], 0
        while i < len(text):
            pair = text[i:i + 2]
            out.append(pair if pair in self.merge else text[i])
            i += 2 if pair in self.merge else 1
        return out


def test_letter_ids_are_the_letters_after_the_chat_tail():
    from decision_gem import letter_ids
    assert letter_ids(_Tok()) == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def test_letter_merged_with_the_tail_is_rejected():
    from decision_gem import letter_ids
    with pytest.raises(ValueError, match="'A'"):
        letter_ids(_Tok(merge=(">A",)))


def test_a_model_without_chat_template_is_rejected():
    from decision_gem import letter_ids
    tok = _Tok()
    tok.chat_template = None
    with pytest.raises(ValueError, match="chat template"):
        letter_ids(tok)


def test_raw_answer_carries_the_uncalibrated_logprobs():
    q = {"type": "noul"}
    lp = [math.log(0.9), math.log(0.1)]
    out = answer(q, options_of(q), lp, True, cal=[0.5, 0.0], raw=True)
    assert out["logprobs"] == {"true": pytest.approx(lp[0], abs=1e-6), "false": pytest.approx(lp[1], abs=1e-6)}
    assert "logprobs" not in answer(q, options_of(q), lp, True)
    q = {"type": "choice", "criteria": ["x", "y"]}
    assert set(answer(q, options_of(q), [-0.1, -2.3], True, raw=True)["logprobs"]) == {"x", "y"}


def _raw(kind, keys, lps):
    return {"type": kind, "logprobs": dict(zip(keys, lps))}


def test_fit_noul_recovers_platt_coefficients():
    import random
    from decision_gem import fit_calibration
    rnd, a, b = random.Random(0), 0.4, -1.0
    answers, labels = [], []
    for _ in range(4000):
        x = rnd.uniform(-12, 12)                         # logit = logprob(true) - logprob(false)
        answers.append({"spam": _raw("noul", ["true", "false"], [x, 0.0])})
        labels.append({"spam": rnd.random() < 1 / (1 + math.exp(-(a * x + b)))})
    fa, fb = fit_calibration(answers, labels)["spam"]
    assert fa == pytest.approx(a, abs=0.06) and fb == pytest.approx(b, abs=0.2)


def test_fit_noul_on_separable_labels_stays_finite():
    from decision_gem import fit_calibration
    answers = [{"q": _raw("noul", ["true", "false"], [x, 0.0])} for x in (-9, -7, -5, 5, 7, 9)]
    labels = [{"q": x > 0} for x in (-9, -7, -5, 5, 7, 9)]
    a, b = fit_calibration(answers, labels)["q"]
    assert 0 < a < 10 and math.isfinite(b)


def test_fit_choice_recovers_temperature():
    import random
    from decision_gem import fit_calibration
    rnd, t, keys = random.Random(1), 2.5, ["phishing", "ads", "normal"]
    answers, labels = [], []
    for _ in range(3000):
        lp = [rnd.uniform(-15, 0) for _ in keys]
        p = [math.exp(x / t) for x in lp]
        label = rnd.choices(keys, weights=p)[0]
        answers.append({"kind": _raw("choice", keys, lp)})
        labels.append({"kind": label})
    assert fit_calibration(answers, labels)["kind"] == pytest.approx(t, rel=0.1)


def test_fit_score_takes_int_labels_and_skips_unlabelled():
    from decision_gem import fit_calibration
    keys = ["1", "2", "3"]
    answers = [{"u": _raw("score", keys, [-0.1, -3, -6])}, {"u": _raw("score", keys, [-6, -3, -0.1])},
               {"u": _raw("score", keys, [-6, -0.1, -3])}, {"u": _raw("score", keys, [-3, -0.1, -6])}]
    labels = [{"u": 1}, {"u": 3}, {}, {"u": 2}]
    assert fit_calibration(answers, labels)["u"] > 0


def test_fit_needs_raw_answers_and_both_classes():
    from decision_gem import fit_calibration
    with pytest.raises(ValueError, match="raw"):
        fit_calibration([{"q": {"type": "noul", "noul": 0.9}}], [{"q": True}])
    one_class = [{"q": _raw("noul", ["true", "false"], [x, 0.0])} for x in (1, 2)]
    with pytest.raises(ValueError, match="both"):
        fit_calibration(one_class, [{"q": True}, {"q": True}])
    with pytest.raises(ValueError, match="same length"):
        fit_calibration(one_class, [{"q": True}])
