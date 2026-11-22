"""Verification tests for the retrieval side of the RealityShift prototype.
Run: python -m pytest -q
"""
import math
import pytest

from retrieval.rs_retrieval import (EvidenceItem, EvidenceStore, check_citations, evaluate,
                          hit_at_k, load_data, mentor_feedback, ndcg_at_k, retrieve,
                          verdict)

store, facts, queries = load_data()


# 1. Hand-calculated metrics (TA guide example and one nDCG example)
def test_hit_at_3_hand_example():
    rankings = [["d2", "d1", "d3"], ["d3", "d5", "d1"], ["d4", "d5", "d3"]]
    gold = [{"d2"}, {"d1"}, {"d1"}]
    assert hit_at_k(rankings, gold, 3) == pytest.approx(2 / 3)


def test_ndcg_hand_example():
    # single relevant document at rank 3: DCG = 1/log2(4) = 0.5, iDCG = 1
    assert ndcg_at_k(["a", "b", "g"], {"g"}, 3) == pytest.approx(0.5)
    # two relevant at ranks 1 and 3: (1 + 0.5) / (1 + 1/log2 3)
    assert ndcg_at_k(["g1", "b", "g2"], {"g1", "g2"}, 3) == pytest.approx(1.5 / (1 + 1 / math.log2(3)))


# 2. Retrieval example with a correct supporting source (TR-TR and EN-EN)
def test_correct_supporting_source_en():
    top = retrieve(store, "Did the reservoir level fall 12% this month?", "en", t=0, k=3)
    assert top[0][0].evidence_id == "E1"
    a = mentor_feedback(store, "The reservoir level fell 12% this month", "F1", "en", t=0)
    assert a.verdict == "S" and [d.evidence_id for d in a.citations] == ["E1"]


def test_correct_supporting_source_tr():
    top = retrieve(store, "Baraj doluluğu bu ay yüzde 12 mi düştü?", "tr", t=0, k=3)
    assert top[0][0].evidence_id == "E2"
    a = mentor_feedback(store, "Baraj doluluğu bu ay yüzde 12 düştü", "F1", "tr", t=0)
    assert a.verdict == "S" and a.citations[0].evidence_id == "E2"


def test_refuting_source():
    # the capacity data (E7) are released in round 2, as in the simulation
    assert mentor_feedback(store, "The reservoir is nearly empty", "F2", "en", t=1).verdict == "N"
    a = mentor_feedback(store, "The reservoir is nearly empty", "F2", "en", t=2)
    assert a.verdict == "R" and a.citations[0].evidence_id == "E7"


# 3. Missing, not-yet-released and conflicting evidence
def test_missing_evidence_gives_insufficient():
    a = mentor_feedback(store, "The mayor resigned", None, "en", t=10)
    assert a.verdict == "N" and a.citations == []
    assert "cannot be verified" in a.text


def test_evidence_not_released_yet():
    # E3 (no city-wide cuts) is released in round 3: before that the mentor must abstain
    assert mentor_feedback(store, "City-wide water cuts start next week", "F3", "en", t=2).verdict == "N"
    assert mentor_feedback(store, "City-wide water cuts start next week", "F3", "en", t=3).verdict == "R"


def test_conflicting_evidence_shows_both_sources():
    # E3 (authority) and E6 (newspaper) make opposite statements about the same
    # scope and the same week: a genuine conflict, shown with both publishers
    a = mentor_feedback(store, "City-wide water cuts start next week", "F3", "en", t=4)
    assert a.verdict == "CONFLICT"
    assert {d.evidence_id for d in a.citations} == {"E3", "E6"}
    assert "Northtown Water Authority" in a.text and "Northtown Daily" in a.text


def test_different_scope_is_not_a_conflict():
    # a district maintenance cut (E5) does not contradict "no city-wide cuts" (E3):
    # they settle different facts, so the district statement is simply supported
    a = mentor_feedback(store, "A water cut is scheduled in the Harbour district next week", "F6", "en", t=4)
    assert a.verdict == "S" and [d.evidence_id for d in a.citations] == ["E5"]


# 4. Citation check: a cited response must actually be supported
def test_wrong_citations_are_detected():
    # E1 is the top-ranked document for a question about the reservoir,
    # but it says nothing about water cuts: citing it for F3 is a wrong citation.
    good = check_citations([("F1", "S", ["E1"])], store)
    bad = check_citations([("F1", "S", ["E1"]), ("F3", "R", ["E1"])], store)
    assert good == {"citation_recall": 1.0, "citation_precision": 1.0}
    assert bad["citation_recall"] == 0.5 and bad["citation_precision"] == 0.5
    # E3 refutes F3: it is a correct citation for "the evidence contradicts F3",
    # but a wrong one if the mentor presents F3 as true
    assert check_citations([("F3", "R", ["E3"])], store)["citation_precision"] == 1.0
    assert check_citations([("F3", "S", ["E3"])], store) == {"citation_recall": 0.0, "citation_precision": 0.0}


# 5. Provenance: simulated content never enters the evidence store
def test_simulated_claim_rejected_from_store():
    fake = EvidenceItem("C3", "claim_version", "agent 7", "post", "2026-11-04", "en", 3, "",
                        "BREAKING: the reservoir is nearly empty")
    with pytest.raises(ValueError):
        store.add(fake)
    assert "C3" not in store.items
    # a real reference carries URL and access date
    ref = EvidenceItem("R1", "real_reference", "Example publisher", "Background", "2026-01-01", "en", 0, "HSK",
                       "Background text.", url="https://example.org", access_date="2026-11-22")
    EvidenceStore([ref])


# 6. Evaluation over the annotated set, including the cross-language gap
def test_evaluation_and_cross_language_gap():
    res = evaluate(store, queries, t=10, k=3)
    assert res["EN-EN"]["hit@3"] == 1.0 and res["TR-TR"]["hit@3"] == 1.0
    # BM25 cannot match a Turkish query to an English-only document
    assert res["TR-EN"]["hit@3"] == 0.0
    # unanswerable questions are excluded from hit@k and scored by abstention
    assert res["unanswerable"]["correct_abstention_rate"] == 1.0
    # in round 1 the water-cut evidence is not released yet: those questions
    # leave hit@k and count as questions the mentor must abstain on
    r1 = evaluate(store, queries, t=1, k=3)
    assert r1["EN-EN"]["n"] == 1 and r1["unanswerable"]["correct_abstention_rate"] == 1.0


# 7. Events and citations are logged
def test_mentor_answer_is_logged():
    log = []
    mentor_feedback(store, "City-wide water cuts start next week", "F3", "en", t=4, log=log)
    e = log[-1]
    assert e["actor"] == "mentor" and e["verdict"] == "CONFLICT"
    assert set(e["cited"]) == {"E3", "E6"} and e["retrieved"][0][0] == "E6"
