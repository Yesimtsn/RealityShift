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
                retrieval/sim_checker.py: RetrievalChecker, drop-in for the detector's PlaceholderChecker
                (Intervention(cond, checker=RetrievalChecker()))
tests/          verification tests (pytest)
```

## Run

```
python -m sim.agents      # step 1: prints the 20 agents
python -m sim.network     # step 2: network summary + picture in outputs/
python -m retrieval.rs_retrieval   # retrieval evaluation + mentor examples, writes retrieval/outputs/
pytest                    # runs the verification tests
```
