"""Step 6: detection and detection-triggered correction.

Detector (Shu et al. knowledge-based, adapted to versions and rounds):
    F_t(c) = 1[ D_hat_t(c) >= theta  and  R_hat_t(c) >= 1 ]
    D_hat_t(c) = (#R_hat + 0.5 #N_hat) / |A(c)|,  R_hat_t(c) = #R_hat (statements judged refuted)
where the verdicts v_hat_t(a) come from a fact checker that may only use the
evidence released by round t. The true D(c) is NOT used by the detector; it is
the gold label for evaluation.

Correction (Option B):
    t_det = first round in which any version is flagged
    Early: t_c = t_det      Late: t_c = t_det + LATE_DELAY      None: no correction
From t_c on, every agent who received a flagged version, or any version derived
from it, gets the official correction once.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .claims import FACTS, LAMBDA, Verdict, is_false
from .rules import rng
from .simulate import THETA, Sim

LATE_DELAY = 4

# Evidence release schedule (scenario records), the same in every condition.
EVIDENCE_SCHEDULE: Dict[int, Set[str]] = {
    0: {"ev_report"},       # monthly report: 12% drop, official source
    2: {"ev_capacity"},     # capacity data: not nearly empty, seasonal drop is normal
    3: {"ev_no_cuts"},      # authority statement: no water cuts planned
    4: {"ev_open_data"},    # raw data published openly
}
EXPERIMENT_ROUNDS = 15      # long enough for Late (t_det + 4) to take effect


class PlaceholderChecker:
    """Stand-in for Sila's evidence-grounded verification core.
    Returns the true verdict of a fact only if the evidence that settles it has
    been released; otherwise N (cannot be checked yet). With error_rate > 0 it
    flips a checkable verdict with that probability (keyed random)."""

    def __init__(self, error_rate: float = 0.0):
        self.error_rate = error_rate

    def verdict(self, fact_id: str, evidence_open: Set[str], seed: int, t: int, key: int) -> Verdict:
        f = FACTS[fact_id]
        if f.evidence is None or f.evidence not in evidence_open:
            return Verdict.N
        v = f.verdict
        if self.error_rate and rng(seed, key, t, "pick").random() < self.error_rate:
            v = Verdict.S if v == Verdict.R else Verdict.R
        return v


def estimate(version, checker, evidence_open, seed, t):
    """Returns (d_hat, r_hat): the estimated deviation and the number of
    statements the checker judges refuted, using only released evidence."""
    vs = [checker.verdict(f, evidence_open, seed, t, key=i) for i, f in enumerate(version.facts)]
    r_hat = sum(v == Verdict.R for v in vs)
    n_hat = sum(v == Verdict.N for v in vs)
    d_hat = (r_hat + LAMBDA * n_hat) / len(vs)
    return d_hat, r_hat


def estimated_deviation(version, checker, evidence_open, seed, t) -> float:
    return estimate(version, checker, evidence_open, seed, t)[0]


@dataclass
class Intervention:
    condition: str                        # "early", "late" or "none"
    checker: PlaceholderChecker = field(default_factory=PlaceholderChecker)
    schedule: Dict[int, Set[str]] = field(default_factory=lambda: dict(EVIDENCE_SCHEDULE))
    late_delay: int = LATE_DELAY
    flagged: Dict[str, int] = field(default_factory=dict)        # version id -> round flagged
    t_det: Optional[int] = None
    corrected: Set[int] = field(default_factory=set)            # agents that got the official correction
    detections: List[dict] = field(default_factory=list)        # per round, for evaluation

    @property
    def t_c(self) -> Optional[int]:
        if self.t_det is None or self.condition == "none":
            return None
        return self.t_det if self.condition == "early" else self.t_det + self.late_delay

    def __call__(self, t: int, sim: Sim) -> Set[int]:
        # 1. release evidence
        new = self.schedule.get(t, set()) - sim.evidence_open
        if new:
            sim.evidence_open |= new
            sim.log(round=t, type="evidence_released", ids=sorted(new))

        # 2. detection: check every version not flagged yet (runs in all conditions)
        for vid, v in sim.tree.versions.items():
            if vid in self.flagged:
                continue
            d_hat, r_hat = estimate(v, self.checker, sim.evidence_open, sim.seed, t)
            if d_hat >= THETA and r_hat >= 1:      # verifiably false, not just unsupported
                self.flagged[vid] = t
                sim.log(round=t, type="flagged", version=vid, d_hat=d_hat, d_true=v.deviation)
        if self.t_det is None and self.flagged:
            self.t_det = t
        self.detections.append(detection_scores(sim, self.flagged, t))

        # 3. correction
        if self.t_c is None or t < self.t_c:
            return set()
        covered = set()
        for vid in self.flagged:
            covered.update(sim.tree.subtree(vid))
        targets = {i for i, a in sim.agents.items()
                   if i not in self.corrected and any(v in covered for _, v in a.memory.received)}
        self.corrected |= targets
        return targets


def detection_scores(sim: Sim, flagged: Dict[str, int], t: int) -> dict:
    """Precision / recall of the flags so far against the gold labels D(c) >= theta."""
    gold = {vid for vid, v in sim.tree.versions.items() if is_false(v.facts, THETA)}
    pred = set(flagged)
    tp = len(gold & pred)
    precision = tp / len(pred) if pred else None
    recall = tp / len(gold) if gold else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall else (0.0 if pred and gold else None))
    return {"round": t, "versions": len(sim.tree), "gold_false": len(gold), "flagged": len(pred),
            "tp": tp, "precision": precision, "recall": recall, "f1": f1}
