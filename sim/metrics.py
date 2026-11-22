"""Outcomes Y(I) = [R(I), D(I), E(I)] plus detection quality, for one finished run."""
from statistics import mean
from typing import Optional

from .agents import State
from .simulate import THETA, Sim


def outcomes(sim: Sim, intervention=None) -> dict:
    agents, tree = sim.agents, sim.tree
    n = len(agents)
    last = sim.rounds[-1]

    # D(I): deviation of the latest version held by each believer at the end
    held = [tree.versions[a.memory.latest_version()].deviation
            for a in agents.values() if a.state == State.B and a.memory.latest_version()]

    # E(I): agents who verified while relevant evidence was released
    verify = [e for e in sim.events if e["type"] == "verify"]
    accessed = {e["agent"] for e in verify if e["refuted"] or e["confirmed"]}
    first_access = min((e["round"] for e in verify if e["refuted"] or e["confirmed"]), default=None)

    # detection delay for gold-false versions that were flagged
    delays = []
    if intervention is not None:
        for vid, t_flag in intervention.flagged.items():
            v = tree.versions[vid]
            if v.deviation >= THETA:
                delays.append(t_flag - v.round)

    violations = [e for e in sim.events if e["type"] == "violation"] + tree.check()
    n_events = max(len(sim.events), 1)

    out = {
        "R_reach_false": last["reach_false"],                 # R(I)
        "reach_any": last["reach"],
        "D_believers_mean": mean(held) if held else 0.0,      # D(I)
        "D_max": tree.stats()["max_deviation"],
        "E_access": len(accessed) / n,                        # E(I)
        "E_first_round": first_access,
        "believers_end": last["B"],
        "corrected_end": last["C"],
        "versions": len(tree),
        "max_depth": tree.stats()["max_depth"],
        "mean_children": tree.stats()["mean_children"],
        "rule_violation_rate": len(violations) / n_events,
    }
    if intervention is not None:
        final = intervention.detections[-1] if intervention.detections else {}
        out.update({
            "t_det": intervention.t_det,
            "t_c": intervention.t_c,
            "det_precision": final.get("precision"),
            "det_recall": final.get("recall"),
            "det_delay_mean": mean(delays) if delays else None,
        })
    return out
