"""Tests for step 7: logs and replay (brief Q2).

A fake LLM returns different text for the same prompt every time, like a
sampled LLM. Same seed is NOT enough to get the same texts; replaying from the
record is."""
import os

from sim.log import load_events, save_run
from sim.network import build
from sim.simulate import run
from sim.textgen import TemplateText, TextGen


class FakeSamplingLLM:
    model_id = "fake-sampling-llm"

    def generate(self, prompt):
        return TemplateText().generate(prompt) + " #" + os.urandom(3).hex()


def texts(sim):
    return [v.text for v in sim.tree.versions.values()]


def first_seed_with_edits():
    for s in range(1, 40):
        G, agents, start = build(s)
        if len(run(G, agents, start, seed=s).tree) > 2:
            return s


def test_same_seed_is_not_enough_for_llm_text():
    s = first_seed_with_edits()
    G, agents, start = build(s)
    a = run(G, agents, start, seed=s, text_gen=TextGen(FakeSamplingLLM()))
    b = run(G, agents, start, seed=s, text_gen=TextGen(FakeSamplingLLM()))
    assert a.events == b.events            # the rules are reproducible ...
    assert texts(a) != texts(b)            # ... the LLM text is not


def test_replay_from_record_is_identical(tmp_path):
    s = first_seed_with_edits()
    G, agents, start = build(s)
    rec = tmp_path / "texts.json"
    tg = TextGen(FakeSamplingLLM(), record_path=str(rec), mode="record")
    original = run(G, agents, start, seed=s, text_gen=tg)
    tg.save()
    replay = run(G, agents, start, seed=s,
                 text_gen=TextGen(FakeSamplingLLM(), record_path=str(rec), mode="replay"))
    assert texts(replay) == texts(original)
    assert replay.events == original.events


def test_saved_log_matches_run(tmp_path):
    G, agents, start = build(2)
    sim = run(G, agents, start, seed=2)
    save_run(sim, str(tmp_path / "run"), {"seed": 2})
    assert load_events(str(tmp_path / "run")) == sim.events
    for name in ("config.json", "events.jsonl", "versions.json", "rounds.csv"):
        assert (tmp_path / "run" / name).exists()
