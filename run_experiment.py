"""Early vs Late vs None correction, with paired seeds.

For every seed the three conditions share the same network, agents, start agent,
evidence schedule and keyed random numbers, so they are identical until the
first correction. We report per-seed differences and bootstrap 95% CIs.

    python run_experiment.py                 # template text, 60 seeds, 15 rounds
    python run_experiment.py --seeds 20      # fewer seeds
    python run_experiment.py --llm           # text from Ollama qwen2.5:7b (slow), recorded
"""
import argparse
import csv
import json
import os
import statistics as st

import numpy as np

from sim.intervention import EXPERIMENT_ROUNDS, Intervention
from sim.log import save_run
from sim.metrics import outcomes
from sim.network import build
from sim.simulate import run
from sim.textgen import OllamaText, TextGen

CONDITIONS = ("early", "late", "none")
OUT = "outputs"


def bootstrap_ci(diffs, n=10_000, seed=0):
    g = np.random.default_rng(seed)
    d = np.array(diffs, dtype=float)
    means = g.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=60)
    ap.add_argument("--rounds", type=int, default=EXPERIMENT_ROUNDS)
    ap.add_argument("--lang", default="en", choices=["en", "tr"])
    ap.add_argument("--llm", action="store_true", help="use Ollama qwen2.5:7b for version text")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    rows, curves, recall_by_round = [], {c: [] for c in CONDITIONS}, []
    for seed in range(1, args.seeds + 1):
        G, agents, start = build(seed)
        for cond in CONDITIONS:
            if args.llm:
                tg = TextGen(OllamaText(), record_path=f"{OUT}/texts_{args.lang}.json")
            else:
                tg = TextGen()
            iv = Intervention(cond)
            sim = run(G, agents, start, seed=seed, T=args.rounds, lang=args.lang,
                      intervene=iv, text_gen=tg)
            if args.llm:
                tg.save()
            o = outcomes(sim, iv)
            rows.append({"seed": seed, "condition": cond, **o})
            if iv.t_det is not None:
                curves[cond].append([r["reach_false"] for r in sim.rounds])
            if cond == "none" and iv.t_det is not None:
                recall_by_round.append({d["round"]: d["recall"] for d in iv.detections})

    # per-run table
    with open(f"{OUT}/results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # summary: means and paired differences
    by = {(r["seed"], r["condition"]): r for r in rows}
    detected = [s for s in range(1, args.seeds + 1) if by[(s, "none")]["t_det"] is not None]
    t_dets = [by[(s, "none")]["t_det"] for s in detected]
    summary = {"seeds": args.seeds, "rounds": args.rounds, "seeds_with_detection": len(detected),
               "t_det_median": st.median(t_dets) if t_dets else None,
               "t_det_range": [min(t_dets), max(t_dets)] if t_dets else None, "means": {}, "paired": {}}
    metrics = ["R_reach_false", "D_believers_mean", "E_access", "believers_end", "corrected_end"]
    for cond in CONDITIONS:
        summary["means"][cond] = {m: round(st.mean(by[(s, cond)][m] for s in detected), 3) for m in metrics}
    for a, b in (("late", "early"), ("none", "early"), ("none", "late")):
        summary["paired"][f"{a}-{b}"] = {}
        for m in ("R_reach_false", "D_believers_mean"):
            diffs = [by[(s, a)][m] - by[(s, b)][m] for s in detected]
            lo, hi = bootstrap_ci(diffs)
            summary["paired"][f"{a}-{b}"][m] = {"mean": round(st.mean(diffs), 3),
                                                "ci95": [round(lo, 3), round(hi, 3)]}
    # detection recall per round (H2), None condition
    rounds_all = sorted({t for d in recall_by_round for t in d})
    summary["recall_by_round"] = {t: round(st.mean(d[t] for d in recall_by_round if d.get(t) is not None), 3)
                                  for t in rounds_all
                                  if any(d.get(t) is not None for d in recall_by_round)}
    summary["rule_violation_rate_max"] = max(r["rule_violation_rate"] for r in rows)
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "recall_by_round"}, indent=1))
    print("recall by round:", summary["recall_by_round"])

    plot(curves, summary, args.rounds)

    # save full logs of one example seed (first seed with a detection), all three conditions
    if detected:
        ex = detected[0]
        G, agents, start = build(ex)
        for cond in CONDITIONS:
            iv = Intervention(cond)
            sim = run(G, agents, start, seed=ex, T=args.rounds, lang=args.lang, intervene=iv)
            save_run(sim, f"{OUT}/runs/seed{ex}_{cond}",
                     {"seed": ex, "condition": cond, "rounds": args.rounds, "lang": args.lang,
                      "t_det": iv.t_det, "t_c": iv.t_c})
        print(f"example logs saved to {OUT}/runs/seed{ex}_*")


def plot(curves, summary, T):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    colours = {"early": "#2E7D32", "late": "#EF6C00", "none": "#C62828"}
    for cond, cs in curves.items():
        if cs:
            m = np.mean(np.array(cs), axis=0)
            ax[0].plot(range(len(m)), m, label=cond, color=colours[cond], linewidth=2)
    ax[0].set_xlabel("round")
    ax[0].set_ylabel("false reach R(t)")
    ax[0].set_title(f"False reach, mean over {summary['seeds_with_detection']} seeds with detection")
    ax[0].legend()
    rb = summary["recall_by_round"]
    ax[1].plot(list(rb), list(rb.values()), marker="o", color="#1565C0")
    ax[1].set_xlabel("round")
    ax[1].set_ylabel("detector recall")
    ax[1].set_ylim(0, 1.05)
    ax[1].set_title("Detector recall per round (placeholder checker, None runs)")
    for a in ax:
        a.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/early_late_none.png", dpi=150)
    print(f"plot saved to {OUT}/early_late_none.png")


if __name__ == "__main__":
    main()
