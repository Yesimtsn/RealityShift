# RealityShift

An adaptive AI reality lab for disinformation resilience (ENS 491/492, Sabancı University).

AI agents share and gradually change a claim on a small social network. The system detects
claim versions that have become false, corrects the agents who saw them, and an AI mentor
explains the evidence to the learner.

## Setup

Python 3.10 or newer.

```
pip install -r requirements.txt
```

## Project structure

```
sim/            simulation (agents, network, rules, claims, rounds, intervention, logs)
retrieval/      evidence retrieval (BM25), verdicts, mentor feedback with citations, retrieval evaluation
                evidence.json: the single reviewed evidence collection, released on the simulation's schedule
                sim_checker.py: RetrievalChecker, the detector's fact checker (replaces PlaceholderChecker)
tests/          verification tests (pytest)
```

## Detector with retrieval

The detector gets its statement verdicts from a checker. `run_experiment.py` uses
`RetrievalChecker`: it retrieves the released records of `retrieval/evidence.json` with BM25 and
takes the verdict from their reviewed evidence-claim relations, so a statement is judged S or R
only once its settling record has been released and retrieved. `PlaceholderChecker`, which reads
the verdict from the fact base, is kept as the reference: `tests/test_sim_checker.py` checks that
both give the same verdicts and the same experiment outcomes.

```
from sim.intervention import Intervention
from retrieval.sim_checker import RetrievalChecker
iv = Intervention("early", checker=RetrievalChecker())
```

## Run

```
python -m sim.agents      # step 1: prints the 20 agents
python -m sim.network     # step 2: network summary + picture in outputs/
python -m retrieval.rs_retrieval   # retrieval evaluation + mentor examples, writes retrieval/outputs/
python run_experiment.py  # early vs late vs none correction, paired seeds, writes outputs/
pytest                    # runs the verification tests
```
