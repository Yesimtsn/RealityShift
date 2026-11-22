"""Tests for step 6 (detection and correction)."""
from sim.agents import State
from sim.claims import FACTS
from sim.intervention import EVIDENCE_SCHEDULE, Intervention, PlaceholderChecker, estimated_deviation
from sim.metrics import outcomes
from sim.network import build
from sim.simulate import run

ALL_EVIDENCE = set().union(*EVIDENCE_SCHEDULE.values())


def paired(seed, T=10):
    G, agents, start = build(seed)
    out = {}
    for cond in ("early", "late", "none"):
        iv = Intervention(cond)
        out[cond] = (run(G, agents, start, seed=seed, T=T, intervene=iv), iv)
    return out


def seeds_with_detection(n=15):
    return [s for s in range(1, 40) if paired(s)["none"][1].t_det is not None][:n]


def test_perfect_checker_with_all_evidence_gives_true_deviation():
    G, agents, start = build(2)
    sim = run(G, agents, start, seed=2)
    for v in sim.tree.versions.values():
        assert estimated_deviation(v, PlaceholderChecker(), ALL_EVIDENCE, 2, 10) == v.deviation


def test_condition_timing_and_pairing():
    for seed in seeds_with_detection(8):
        runs = paired(seed)
        t_dets = {c: runs[c][1].t_det for c in runs}
        assert len(set(t_dets.values())) == 1           # same detection round in all three
        t_det = t_dets["none"]
        assert runs["early"][1].t_c == t_det
        assert runs["late"][1].t_c == t_det + 4
        assert runs["none"][1].t_c is None
        # identical runs before the first correction
        before = lambda sim: [e for e in sim.events if e["round"] < t_det]
        assert before(runs["early"][0]) == before(runs["none"][0]) == before(runs["late"][0])


def test_none_never_corrects_and_corrected_agents_never_share():
    for seed in seeds_with_detection(8):
        runs = paired(seed)
        assert not any(e["type"] == "correction_delivered" for e in runs["none"][0].events)
        for cond in ("early", "late"):
            sim = runs[cond][0]
            assert sim.tree.check() == []
            assert outcomes(sim)["rule_violation_rate"] == 0
            corrected_at = {}
            for e in sim.events:
                if e["type"] == "state" and e["after"] == "C":
                    corrected_at.setdefault(e["agent"], e["round"])
            for e in sim.events:
                if e["type"] == "post" and e["kind"] == "claim" and e["agent"] in corrected_at:
                    assert e["round"] < corrected_at[e["agent"]]


def test_detection_recall_grows_as_evidence_is_released():
    # H2 at the level of one run: recall cannot go down while versions stay fixed
    for seed in seeds_with_detection(5):
        sim, iv = paired(seed)["none"]
        recalls = [d["recall"] for d in iv.detections if d["recall"] is not None]
        assert recalls[-1] >= recalls[0]
