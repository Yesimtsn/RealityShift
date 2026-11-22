"""Retrieval-based fact checker for the simulation's detector.

Drop-in replacement for sim.intervention.PlaceholderChecker (same verdict()
signature). It uses the single evidence collection retrieval/evidence.json,
whose records carry the simulation's evidence ids (sim_id) and whose facts are
mapped to the simulation's fact ids (sim_facts). Instead of looking the answer
up in the fact base, it
  1. retrieves, with BM25, the English records released in the simulation
     (sim_id in evidence_open) that share terms with the statement, and
  2. returns the verdict from the reviewed evidence-claim relations of the
     retrieved records (rs_retrieval.verdict).
So a statement is S or R only if its settling record has been released AND
retrieval finds it; otherwise N.

    from sim.intervention import Intervention
    from retrieval.sim_checker import RetrievalChecker
    iv = Intervention("early", checker=RetrievalChecker())
"""
from __future__ import annotations

from sim.claims import FACTS, Verdict

from .rs_retrieval import EvidenceStore, load_data, load_sim_facts, retrieve, verdict


class RetrievalChecker:
    def __init__(self, k: int = 3, lang: str = "en"):
        self.store, _, _ = load_data()
        self.fact_map = load_sim_facts()
        self.k, self.lang = k, lang

    def verdict(self, fact_id: str, evidence_open, seed: int, t: int, key: int) -> Verdict:
        # seed, t, key: kept for the PlaceholderChecker interface; this checker is deterministic
        own_id = self.fact_map.get(fact_id)
        if own_id is None:                       # no fact of the collection covers it
            return Verdict.N
        released = EvidenceStore([d for d in self.store.items.values()
                                  if d.sim_id in evidence_open and d.lang == self.lang])
        if not released.items:
            return Verdict.N
        text = getattr(FACTS[fact_id], self.lang)
        # tau = 0 and score > 0: the collection is tiny, so any shared term selects a record
        hits = [d for d, s in retrieve(released, text, self.lang, t=10**9, k=self.k, tau=0.0) if s > 0]
        v, _ = verdict(own_id, hits)
        return Verdict.N if v == "CONFLICT" else Verdict(v)
