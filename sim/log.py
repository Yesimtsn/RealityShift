"""Step 7: saving runs, so they can be audited and replayed.

For one run, save_run() writes to a folder:
  config.json    everything needed to repeat the run (seed, rounds, condition, model id, ...)
  events.jsonl   one line per event (state changes, posts, versions, verifies, flags, corrections)
  versions.json  the full version tree (id, parent, author, round, edit, facts, D, text, intent)
  rounds.csv     reach, false reach and state counts per round
The text record (texts.json) is saved by TextGen.save(); with it, a run can be
replayed exactly even when the text came from an LLM.
"""
import csv
import json
from pathlib import Path


def save_run(sim, folder: str, config: dict) -> None:
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(config, text_model=sim.text_gen.backend.model_id, rules=sim.rules.name)
    (out / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    with open(out / "events.jsonl", "w", encoding="utf-8") as f:
        for e in sim.events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    versions = [v.to_dict() for v in sim.tree.versions.values()]
    (out / "versions.json").write_text(json.dumps(versions, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(out / "rounds.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sim.rounds[0]))
        w.writeheader()
        w.writerows(sim.rounds)


def load_events(folder: str) -> list:
    with open(Path(folder) / "events.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]
