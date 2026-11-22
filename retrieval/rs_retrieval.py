"""RealityShift proof of concept: evidence retrieval, verification core and
evidence-grounded mentor feedback (retrieval side, H. S. Kara).

Notation follows the report:
  E_t          evidence released by round t (reviewed collection only)
  R_k(q, E_t)  top-k documents by score s(q, d)
  Z_t(q)       selected evidence: top-k, score >= tau, released by t
  v(a)         verdict of a statement a in {S, R, N} (+ CONFLICT for the mentor)

The prototype uses BM25 only. Dense / cross-language retrieval (BGE-M3) is part
of the design but not of this prototype; test_evaluation_and_cross_language_gap shows why it is
needed.

Pure Python, no third-party dependencies. Run: python -m pytest -q
"""
from __future__ import annotations

import json
import math
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

DATA = Path(__file__).with_name("evidence.json")

ALLOWED_TYPES = {"scenario_record", "real_reference"}


# --------------------------------------------------------------------------
# Tokenisation
# --------------------------------------------------------------------------
_TR_LOWER = str.maketrans({"I": "ı", "İ": "i"})
_STOP = {
    "en": {"the", "a", "an", "of", "in", "is", "and", "to", "this", "by", "for",
           "did", "does", "are", "was", "on", "it", "be", "will", "next", "according"},
    "tr": {"ve", "bu", "bir", "mi", "mı", "mu", "mü", "da", "de", "ile", "için",
           "göre", "gibi", "çok", "ne"},
}


def tokenize(text: str, lang: str) -> list[str]:
    """Lower-case (Turkish-aware), split, drop stop words.

    Turkish is agglutinative ("barajın", "barajdaki", "baraj"), so Turkish
    tokens are truncated to their first 5 characters, a simple prefix stemmer.
    Numbers such as "12%" are kept as tokens because they carry the facts.
    """
    if lang == "tr":
        text = text.translate(_TR_LOWER)
    text = text.lower()
    tokens = re.findall(r"\d+%?|[^\W\d_]+", text)
    tokens = [t for t in tokens if t not in _STOP.get(lang, set())]
    if lang == "tr":
        tokens = [t[:5] if not t[0].isdigit() else t for t in tokens]
    return tokens


# --------------------------------------------------------------------------
# Evidence collection
# --------------------------------------------------------------------------
@dataclass
class EvidenceItem:
    evidence_id: str
    type: str                 # scenario_record | real_reference
    publisher: str
    title: str
    pub_date: str
    lang: str
    release_round: int
    reviewer: str
    text: str
    relations: dict = field(default_factory=dict)   # fact_id -> "S" | "R" (reviewed)
    url: str = ""             # real references only
    access_date: str = ""     # real references only

    def cite(self) -> str:
        tag = "scenario record" if self.type == "scenario_record" else "reference"
        return f"[{self.evidence_id}] {self.publisher}, {self.pub_date} ({self.lang}, {tag})"


class EvidenceStore:
    """Reviewed evidence collection. Simulated claim versions are refused."""

    def __init__(self, items: list[EvidenceItem]):
        for it in items:
            if it.type not in ALLOWED_TYPES:
                raise ValueError(f"{it.evidence_id}: type '{it.type}' may not enter the evidence store")
            if not it.reviewer:
                raise ValueError(f"{it.evidence_id}: unreviewed item")
        self.items = {it.evidence_id: it for it in items}

    def add(self, item: EvidenceItem) -> None:
        EvidenceStore([item])          # same checks
        self.items[item.evidence_id] = item

    def released(self, t: int) -> list[EvidenceItem]:
        return [it for it in self.items.values() if it.release_round <= t]


def load_data(path: Path = DATA):
    raw = json.loads(path.read_text(encoding="utf-8"))
    store = EvidenceStore([EvidenceItem(**d) for d in raw["evidence"]])
    return store, raw["facts"], raw["queries"]


# --------------------------------------------------------------------------
# BM25 retrieval:  s(q,d) = sum_w IDF(w) f(w,d)(k1+1) / (f(w,d) + k1(1-b+b|d|/avgdl))
# --------------------------------------------------------------------------
class BM25:
    def __init__(self, docs: list[EvidenceItem], k1: float = 1.2, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.toks = {d.evidence_id: tokenize(d.title + " " + d.text, d.lang) for d in docs}
        n = len(docs)
        self.avgdl = sum(len(t) for t in self.toks.values()) / max(n, 1)
        df: dict[str, int] = {}
        for t in self.toks.values():
            for w in set(t):
                df[w] = df.get(w, 0) + 1
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def score(self, q_tokens: list[str], d: EvidenceItem) -> float:
        toks = self.toks[d.evidence_id]
        s = 0.0
        for w in q_tokens:
            f = toks.count(w)
            if f == 0:
                continue
            s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(toks) / self.avgdl))
        return s


def retrieve(store: EvidenceStore, query: str, lang: str, t: int, k: int = 3,
             tau: float = 1.0) -> list[tuple[EvidenceItem, float]]:
    """Z_t(q): top-k released documents with score >= tau."""
    docs = store.released(t)
    if not docs:
        return []
    bm = BM25(docs)
    q = tokenize(query, lang)
    ranked = sorted(((d, bm.score(q, d)) for d in docs), key=lambda x: -x[1])
    return [(d, s) for d, s in ranked[:k] if s >= tau]


# --------------------------------------------------------------------------
# Verification core and mentor
# --------------------------------------------------------------------------
def verdict(fact_id: str | None, evidence: list[EvidenceItem]) -> tuple[str, list[EvidenceItem]]:
    """Statement verdict from reviewed evidence-claim relations.

    S: some selected document supports it and none refutes it
    R: some refutes and none supports
    CONFLICT: both
    N: no selected document settles it (includes invented statements, fact_id None)
    """
    if fact_id is None:
        return "N", []
    sup = [d for d in evidence if d.relations.get(fact_id) == "S"]
    ref = [d for d in evidence if d.relations.get(fact_id) == "R"]
    if sup and ref:
        return "CONFLICT", sup + ref
    if sup:
        return "S", sup
    if ref:
        return "R", ref
    return "N", []


@dataclass
class MentorAnswer:
    statement: str
    verdict: str
    citations: list[EvidenceItem]
    text: str


EVENT_LOG: list[dict] = []


def mentor_feedback(store, statement: str, fact_id: str | None, lang: str, t: int,
                    k: int = 3, tau: float = 1.0, log: list | None = EVENT_LOG) -> MentorAnswer:
    """Constructed (template) feedback; every citation is a document whose reviewed
    relation settles the statement, so citation support holds by construction."""
    t0 = time.perf_counter()
    scored = retrieve(store, statement, lang, t, k, tau)
    selected = [d for d, _ in scored]
    v, cites = verdict(fact_id, selected)
    refs = "; ".join(d.cite() for d in cites)
    msgs = {
        "S": f"The released evidence supports this statement: {refs}.",
        "R": f"The released evidence contradicts this statement: {refs}.",
        "CONFLICT": f"The sources disagree; compare their dates and publishers: {refs}.",
        "N": "The evidence released so far does not settle this statement, so it cannot be verified yet.",
    }
    if log is not None:
        log.append({"round": t, "actor": "mentor", "action": "answer", "question": statement,
                    "lang": lang, "retrieved": [[d.evidence_id, round(x, 3)] for d, x in scored],
                    "verdict": v, "cited": [d.evidence_id for d in cites],
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 3)})
    return MentorAnswer(statement, v, cites, msgs[v])


def check_citations(answer: list[tuple[str | None, str, list[str]]], store: EvidenceStore) -> dict:
    """Citation recall / precision for an answer given as
    [(fact_id, asserted, [cited ids])], where asserted is "S" if the mentor
    presents the statement as supported and "R" if it presents it as refuted.

    A citation is correct only if its reviewed relation to the fact equals the
    asserted direction: citing a refuting document for a statement presented
    as true is a wrong citation.
    recall:    share of statements with at least one correct citation
    precision: share of citations that are correct for their statement
    (reviewed relations stand in for the NLI judge of ALCE in this prototype)
    """
    supported, total_c, good_c = 0, 0, 0
    for fact_id, asserted, cited in answer:
        ok = [c for c in cited if fact_id is not None and store.items[c].relations.get(fact_id) == asserted]
        supported += bool(ok)
        total_c += len(cited)
        good_c += len(ok)
    n = len(answer)
    return {"citation_recall": supported / n if n else 0.0,
            "citation_precision": good_c / total_c if total_c else 0.0}


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def hit_at_k(rankings: list[list[str]], gold: list[set[str]], k: int) -> float:
    return sum(bool(set(r[:k]) & g) for r, g in zip(rankings, gold)) / len(rankings)


def ndcg_at_k(ranking: list[str], gold: set[str], k: int) -> float:
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(ranking[:k]) if d in gold)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / idcg if idcg else 0.0


def latency_profile(fn, n: int = 50) -> dict:
    xs = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t0) * 1000)
    xs.sort()
    return {"median_ms": statistics.median(xs), "p95_ms": xs[math.ceil(0.95 * n) - 1]}


def evaluate(store, queries, t: int = 10, k: int = 3, tau: float = 1.0) -> dict:
    """hit@k and nDCG@k per setting over the queries answerable in round t
    (G_t(q) = G(q) & E_t non-empty), the TR-EN gap, and the correct abstention
    rate over the queries not answerable in round t: the mentor abstains if
    Z_t(q) is empty or no selected document settles the statement (verdict N)."""
    released = {d.evidence_id for d in store.released(t)}
    res: dict[str, dict] = {}
    for setting in sorted({q["setting"] for q in queries}):
        qs = [q for q in queries if q["setting"] == setting and set(q["gold"]) & released]
        if not qs:
            continue
        ranks = [[d.evidence_id for d, _ in retrieve(store, q["text"], q["lang"], t, k, tau=0.0)] for q in qs]
        golds = [set(q["gold"]) & released for q in qs]
        res[setting] = {"n": len(qs),
                        f"hit@{k}": hit_at_k(ranks, golds, k),
                        f"nDCG@{k}": sum(ndcg_at_k(r, g, k) for r, g in zip(ranks, golds)) / len(qs)}
    unans = [q for q in queries if not set(q["gold"]) & released]
    if unans:
        abstained = 0
        for q in unans:
            selected = [d for d, _ in retrieve(store, q["text"], q["lang"], t, k, tau)]
            abstained += (not selected) or verdict(q.get("fact"), selected)[0] == "N"
        res["unanswerable"] = {"n": len(unans), "correct_abstention_rate": abstained / len(unans)}
    if "EN-EN" in res and "TR-TR" in res:
        res["gap_EN_minus_TR"] = {m: res["EN-EN"][m] - res["TR-TR"][m] for m in (f"hit@{k}", f"nDCG@{k}")}
    return res


if __name__ == "__main__":
    store, facts, queries = load_data()
    res = evaluate(store, queries)
    print(json.dumps(res, indent=2))
    for f_id, st, lang, t in [("F1", "The reservoir level fell 12% this month", "en", 0),
                              ("F1", "Baraj doluluğu bu ay yüzde 12 düştü", "tr", 0),
                              ("F2", "The reservoir is nearly empty", "en", 0),
                              (None, "The mayor resigned", "en", 10),
                              ("F3", "City-wide water cuts start next week", "en", 1),
                              ("F3", "City-wide water cuts start next week", "en", 2),
                              ("F3", "City-wide water cuts start next week", "en", 4),
                              ("F6", "A water cut is scheduled in the Harbour district next week", "en", 4)]:
        a = mentor_feedback(store, st, f_id, lang, t=t)
        print(f"round {t}: {st!r} -> {a.verdict}: {a.text}")
    # preliminary microbenchmark: one template-based mentor answer (retrieval + verdict + template), 50 runs
    res["latency"] = latency_profile(lambda: mentor_feedback(store, "City-wide water cuts start next week", "F3", "en", t=4, log=None))
    out = Path(__file__).with_name("outputs"); out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (out / "events.jsonl").write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in EVENT_LOG) + "\n", encoding="utf-8")
    print("wrote outputs/results.json and outputs/events.jsonl")
