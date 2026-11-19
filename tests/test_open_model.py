"""Open states on the real model (Gemma 26B, ~16 GB from the HF cache): run explicitly, one big model at a time.
DECISION_GEM_TEST_MODEL picks another model."""
import os

import pytest

from decision_gem import MODEL, DecisionGem

pytestmark = pytest.mark.models

Q = {"done": {"type": "noul", "instructions": "Has the caller finished speaking?"}}


@pytest.fixture(scope="module")
def gem():
    with DecisionGem(os.environ.get("DECISION_GEM_TEST_MODEL", MODEL)) as g:
        yield g


def noul(gem, key):
    return gem.decide(key=key, questions=Q)["answers"]["done"]["noul"]


def test_extend_equals_one_read(gem):
    gem.prefill("a", "Operator: Good day, how can I help?\nCaller: I would like", open=True)
    gem.extend("a", " to book a visit.")
    gem.prefill("b", "Operator: Good day, how can I help?\nCaller: I would like to book a visit.", open=True)
    pa, pb = noul(gem, "a"), noul(gem, "b")
    assert abs(pa - pb) < 0.02
    assert noul(gem, "a") == pytest.approx(pa, abs=1e-4)   # decide twice: the kept cache is not written


def test_extend_closed_state_fails(gem):
    gem.prefill("c", "x")
    with pytest.raises(ValueError):
        gem.extend("c", "y")


def test_open_state_survives_save_and_load(gem, tmp_path):
    gem.prefill("d", "Operator: Good day.\nCaller: I need", open=True)
    before = noul(gem, "d")
    gem.save("d", str(tmp_path / "d.safetensors"))
    gem.load("e", str(tmp_path / "d.safetensors"))
    assert noul(gem, "e") == pytest.approx(before, abs=1e-4)
    gem.extend("e", " a visit.")   # still open after load


def test_long_open_state_extended_in_chunks_equals_one_read(gem):
    # over 512 tokens: the sliding-window layers wrap; chunks start at line starts, the last one is one token (".")
    turns = [f"Caller: I would like a visit on day {i}\nOperator: Noted, day {i}" for i in range(1, 70)]
    third = len(turns) // 3
    first = "\n".join(turns[:third])
    second = "\n" + "\n".join(turns[third:2 * third])
    third_chunk = "\n" + "\n".join(turns[2 * third:])
    assert gem.prefill("long-b", first + second + third_chunk + ".", open=True) > 512
    gem.prefill("long-a", first, open=True)
    gem.extend("long-a", second)
    gem.extend("long-a", third_chunk)
    assert gem.extend("long-a", ".") == 1
    pa, pb = noul(gem, "long-a"), noul(gem, "long-b")
    assert abs(pa - pb) < 0.02
    for _ in range(2):   # decide again: the kept cache is not written
        assert noul(gem, "long-a") == pytest.approx(pa, abs=1e-4)


def test_long_open_state_decides_as_closed_state(gem):
    # one read each: the closing tag goes in with the question (open) or sits in the kept cache (closed)
    text = "\n".join(f"Caller: I would like a visit on day {i}\nOperator: Noted, day {i}" for i in range(1, 70)) + "."
    assert gem.prefill("long-open", text, open=True) > 1024
    gem.prefill("long-closed", text)
    pc = noul(gem, "long-closed")
    assert abs(noul(gem, "long-open") - pc) < 0.02
    assert gem.decide(text, Q)["answers"]["done"]["noul"] == pytest.approx(pc, abs=1e-4)
