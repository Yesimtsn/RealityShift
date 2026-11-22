"""Step 3: states, transitions and action choice.

Transition model (brief, eq. 3):
    s_i(t+1) ~ P_theta( s_i(t+1) | s_i(t), N_i(t), m_i(t), a_i(t) )

Made concrete with four rules:
  1. Exposure  (deterministic): U -> X when the agent receives any version.
  2. Decision  (random, by role): X -> B / D / X with TRANSITION_X[role].
  3. Correction (random, by role): X, B or D -> C with probability GAMMA[role]
     when the agent receives a correction.
  4. C is absorbing.
Then the agent picks an action a_i(t) ~ POLICY[role][state].

All probabilities are design assumptions (not measured on people).
Every random draw uses a generator keyed by (seed, agent, round, purpose), so
the same decision gets the same random number in Early, Late and None runs.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict

import numpy as np

from .agents import Agent, Role, State


class Action(str, Enum):
    IGNORE = "ignore"
    SHARE = "share"      # repost a version unchanged
    MODIFY = "modify"    # make a new version c' = T(c, phi, xi) and post it
    VERIFY = "verify"    # look up released evidence
    CORRECT = "correct"  # post a correction


# ---------- keyed random numbers ----------
PURPOSE = {"decide": 1, "correct": 2, "action": 3, "edit": 4, "pick": 5}


def rng(seed: int, agent: int, round_: int, purpose: str) -> np.random.Generator:
    """Independent random stream for one decision. Does not depend on the
    order in which agents are processed, or on what happened in other agents."""
    return np.random.default_rng([seed, agent, round_, PURPOSE[purpose]])


def sample(dist: Dict, g: np.random.Generator):
    """Draw one key from {outcome: probability}."""
    keys = list(dist)
    p = np.array([dist[k] for k in keys], dtype=float)
    return keys[int(g.choice(len(keys), p=p / p.sum()))]


# ---------- rule tables ----------
@dataclass
class Rules:
    name: str
    transition_x: Dict[Role, Dict[State, float]]       # P(next | X, role)
    gamma: Dict[Role, float]                           # P(accept correction | role)
    policy: Dict[Role, Dict[State, Dict[Action, float]]]
    deceptive_policy: Dict[State, Dict[Action, float]] = field(default_factory=dict)


IGN = {Action.IGNORE: 1.0}

STOCHASTIC = Rules(
    name="stochastic",
    transition_x={
        Role.SHARER:      {State.B: 0.7, State.D: 0.2, State.X: 0.1},
        Role.INTERPRETER: {State.B: 0.6, State.D: 0.3, State.X: 0.1},  # from the design guide
        Role.VERIFIER:    {State.B: 0.2, State.D: 0.6, State.X: 0.2},
        Role.OBSERVER:    {State.B: 0.4, State.D: 0.4, State.X: 0.2},
    },
    gamma={Role.SHARER: 0.6, Role.INTERPRETER: 0.5, Role.VERIFIER: 0.95, Role.OBSERVER: 0.7},
    policy={
        Role.SHARER: {
            State.U: IGN,
            State.X: {Action.SHARE: 0.3, Action.IGNORE: 0.7},
            State.B: {Action.SHARE: 0.85, Action.IGNORE: 0.15},
            State.D: {Action.VERIFY: 0.2, Action.IGNORE: 0.8},
            State.C: IGN,
        },
        Role.INTERPRETER: {
            State.U: IGN,
            State.X: {Action.MODIFY: 0.2, Action.IGNORE: 0.8},
            State.B: {Action.MODIFY: 0.6, Action.SHARE: 0.3, Action.IGNORE: 0.1},
            State.D: {Action.VERIFY: 0.3, Action.IGNORE: 0.7},
            State.C: IGN,
        },
        Role.VERIFIER: {
            State.U: IGN,
            State.X: {Action.VERIFY: 0.8, Action.IGNORE: 0.2},
            State.B: {Action.VERIFY: 0.7, Action.SHARE: 0.2, Action.IGNORE: 0.1},
            State.D: {Action.VERIFY: 0.9, Action.IGNORE: 0.1},
            State.C: {Action.CORRECT: 0.7, Action.IGNORE: 0.3},
        },
        Role.OBSERVER: {
            State.U: IGN,
            State.X: IGN,
            State.B: {Action.SHARE: 0.05, Action.IGNORE: 0.95},
            State.D: IGN,
            State.C: IGN,
        },
    },
    # the deceptive agent edits the claim on purpose, whatever it believes
    deceptive_policy={
        State.X: {Action.MODIFY: 0.8, Action.IGNORE: 0.2},
        State.B: {Action.MODIFY: 0.8, Action.IGNORE: 0.2},
        State.D: {Action.MODIFY: 0.8, Action.IGNORE: 0.2},
    },
)

# Deterministic rules for the hand-calculated test (design guide, page 4):
# everyone who sees the claim believes it and shares it next round,
# observers never share, a correction always works.
_share_all = {State.U: IGN, State.X: {Action.SHARE: 1.0}, State.B: {Action.SHARE: 1.0},
              State.D: IGN, State.C: IGN}
REFERENCE = Rules(
    name="reference",
    transition_x={r: {State.B: 1.0} for r in Role},
    gamma={r: 1.0 for r in Role},
    policy={Role.SHARER: _share_all, Role.INTERPRETER: _share_all, Role.VERIFIER: _share_all,
            Role.OBSERVER: {s: IGN for s in State}},
)


# ---------- the rules applied to one agent ----------
def next_state(agent: Agent, got_correction: bool, rules: Rules, seed: int, t: int) -> State:
    """Rules 2-4 for agent i in round t (rule 1, exposure, happens on delivery)."""
    s = agent.state
    if s == State.C or s == State.U:
        return s                                   # absorbing / nothing to decide
    if got_correction:
        g = rng(seed, agent.id, t, "correct")
        if g.random() < rules.gamma[agent.role]:
            return State.C
        # correction rejected: keep the current state
    if s == State.X:
        return sample(rules.transition_x[agent.role], rng(seed, agent.id, t, "decide"))
    return s


def choose_action(agent: Agent, rules: Rules, seed: int, t: int) -> Action:
    if agent.deceptive and agent.state in rules.deceptive_policy:
        dist = rules.deceptive_policy[agent.state]
    else:
        dist = rules.policy[agent.role][agent.state]
    return sample(dist, rng(seed, agent.id, t, "action"))


def check_tables(rules: Rules) -> None:
    """Every distribution sums to 1, and a corrected agent can never share or modify."""
    dists = list(rules.transition_x.values()) + [d for r in rules.policy.values() for d in r.values()]
    dists += list(rules.deceptive_policy.values())
    for d in dists:
        assert abs(sum(d.values()) - 1.0) < 1e-9, d
    for role in Role:
        c = rules.policy[role][State.C]
        assert c.get(Action.SHARE, 0) == 0 and c.get(Action.MODIFY, 0) == 0, role


if __name__ == "__main__":
    check_tables(STOCHASTIC)
    check_tables(REFERENCE)
    from .agents import Agent as A
    # empirical check: 10,000 exposed interpreters
    counts = {State.B: 0, State.D: 0, State.X: 0}
    for i in range(10_000):
        a = A(id=i, role=Role.INTERPRETER, community=0, state=State.X)
        counts[next_state(a, False, STOCHASTIC, seed=1, t=1)] += 1
    print("interpreter X ->", {k.value: v / 10_000 for k, v in counts.items()},
          "(table: B 0.6, D 0.3, X 0.1)")
