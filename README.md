# RealityShift

An adaptive AI reality lab for disinformation resilience.

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
tests/          verification tests (pytest)
```

## Run

```
python -m sim.agents      # step 1: prints the 20 agents
pytest                    # runs the verification tests
```
