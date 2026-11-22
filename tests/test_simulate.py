"""Tests for step 5 (the round loop).

The first two tests are the hand-calculated example from
'How to formulate your design', page 4."""
from sim.agents import State
from sim.network import build, distances_from, guide_network
from sim.rules import REFERENCE
from sim.simulate import run


def reach(sim):
    return [round(r["reach"], 2) for r in sim.rounds]


def test_hand_calculated_reach():
    G, agents, start = guide_network()
    sim = run(G, agents, start, rules=REFERENCE, T=4)
    assert reach(sim) == [0.17, 0.50, 0.67, 1.00, 1.00]


def test_hand_calculated_correction_round_2_vs_3():
    G, agents, start = guide_network()
    early = run(G, agents, start, rules=REFERENCE, T=6,
                intervene=lambda t, s: {3} if t == 2 else set())
    assert reach(early)[-1] == 0.50                 # agent 3 never shares, 4/5/6 never see it
    assert early.agents[3].state == State.C
    late = run(G, agents, start, rules=REFERENCE, T=6,
               intervene=lambda t, s: {3} if t == 3 else set())
    assert reach(late)[-1] == 1.00                  # too late: agent 3 already shared in round 2


def test_nobody_is_exposed_before_their_distance():
    for seed in range(10):
        G, agents, start = build(seed)
        sim = run(G, agents, start, seed=seed, T=10)
        d = distances_from(G, start)
        for a in sim.agents.values():
            if a.memory.first_exposure_round is not None:
                assert a.memory.first_exposure_round >= d[a.id]


def test_version_tree_valid_after_runs():
    for seed in range(20):
        G, agents, start = build(seed)
        sim = run(G, agents, start, seed=seed, T=10)
        assert sim.tree.check() == []
        assert not any(e["type"] == "violation" for e in sim.events)


def test_same_seed_same_run():
    G, agents, start = build(3)
    assert run(G, agents, start, seed=3).events == run(G, agents, start, seed=3).events


def test_runs_identical_before_the_correction():
    """Paired runs: with and without a correction in round 4, everything
    before round 4 is identical (keyed random numbers)."""
    G, agents, start = build(5)
    none = run(G, agents, start, seed=5, T=10)
    corr = run(G, agents, start, seed=5, T=10,
               intervene=lambda t, s: set(s.agents) if t == 4 else set())
    before = lambda sim: [e for e in sim.events if e["round"] < 4]
    assert before(none) == before(corr)
    assert none.rounds[:4] == corr.rounds[:4]
