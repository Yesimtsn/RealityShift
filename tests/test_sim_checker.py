"""The retrieval checker connected to the simulation's detector."""
from retrieval.sim_checker import RetrievalChecker
from sim.claims import FACTS, Verdict
from sim.intervention import EVIDENCE_SCHEDULE, Intervention, PlaceholderChecker
from sim.metrics import outcomes
from sim.network import build
from sim.simulate import run


def released_by(t):
    return set().union(*(ids for r, ids in EVIDENCE_SCHEDULE.items() if r <= t))


def test_same_verdicts_as_placeholder_in_every_round():
    rc, pc = RetrievalChecker(), PlaceholderChecker()
    for t in range(6):
        open_ = released_by(t)
        for fid in FACTS:
            assert rc.verdict(fid, open_, 1, t, 0) == pc.verdict(fid, open_, 1, t, 0), (fid, t)


def test_refuting_record_found_only_after_release():
    rc = RetrievalChecker()
    assert rc.verdict("cuts_next_week", released_by(2), 1, 2, 0) == Verdict.N
    assert rc.verdict("cuts_next_week", released_by(3), 1, 3, 0) == Verdict.R


def test_no_record_means_not_settled():
    rc = RetrievalChecker()
    assert rc.verdict("mayor_resigned", released_by(10), 1, 10, 0) == Verdict.N


def test_detection_and_outcomes_unchanged_when_connected():
    for seed in (1, 2, 3):
        G, agents, start = build(seed)
        for cond in ("early", "late", "none"):
            a, b = Intervention(cond), Intervention(cond, checker=RetrievalChecker())
            sa = run(G, agents, start, seed=seed, T=10, intervene=a)
            sb = run(G, agents, start, seed=seed, T=10, intervene=b)
            assert a.t_det == b.t_det and a.flagged == b.flagged
            assert outcomes(sa, a) == outcomes(sb, b)
