"""

An agent i has: a role rho_i, a community, an intent flag (deceptive),
a state s_i(t) in {U, X, B, D, C}, and a memory m_i(t).
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class State(str, Enum):
    U = "U"  # unaware: has never seen the claim
    X = "X"  # exposed: has seen it, not decided yet
    B = "B"  # believes it
    D = "D"  # doubts it
    C = "C"  # corrected (absorbing: never leaves C)


class Role(str, Enum):
    SHARER = "sharer"            # shares fast, adds urgency
    INTERPRETER = "interpreter"  # adds opinion and detail, changes the claim most
    VERIFIER = "verifier"        # checks evidence first, posts corrections
    OBSERVER = "observer"        # reads, rarely posts


@dataclass
class Memory:
    """m_i(t): what the agent has seen and done."""
    received: List[Tuple[int, str]] = field(default_factory=list)  # (round, version_id)
    own_posts: List[str] = field(default_factory=list)              # post ids
    correction_round: Optional[int] = None                          # when it saw a correction
    first_exposure_round: Optional[int] = None                      # when it first saw the claim

    def has_version(self, version_id: str) -> bool:
        return any(v == version_id for _, v in self.received)

    def latest_version(self) -> Optional[str]:
        return self.received[-1][1] if self.received else None


@dataclass
class Agent:
    id: int
    role: Role
    community: int
    deceptive: bool = False      # intent flag: edits by this agent are intentional
    persona: str = ""            # optional text for the LLM later; not used by the rules
    state: State = State.U
    memory: Memory = field(default_factory=Memory)


# Role mix per community: 2 sharers, 2 interpreters, 1 verifier, rest observers.
# Communities of size 7, 7, 6 -> 6 sharers, 6 interpreters, 3 verifiers, 5 observers.
def roles_for_community(size: int) -> List[Role]:
    base = [Role.SHARER, Role.SHARER, Role.INTERPRETER, Role.INTERPRETER, Role.VERIFIER]
    return base + [Role.OBSERVER] * (size - len(base))


def make_agents(sizes=(7, 7, 6), deceptive_community: Optional[int] = 1) -> Dict[int, Agent]:
    """Creates the agents, numbered 0..n-1, community by community.
    One interpreter in `deceptive_community` is deceptive (None = no deceptive agent)."""
    agents: Dict[int, Agent] = {}
    next_id = 0
    for community, size in enumerate(sizes):
        for role in roles_for_community(size):
            agents[next_id] = Agent(id=next_id, role=role, community=community)
            next_id += 1
    if deceptive_community is not None:
        first_interpreter = next(a for a in agents.values()
                                 if a.community == deceptive_community and a.role == Role.INTERPRETER)
        first_interpreter.deceptive = True
    return agents


if __name__ == "__main__":
    agents = make_agents()
    for a in agents.values():
        flag = "  (deceptive)" if a.deceptive else ""
        print(f"agent {a.id:2d}  community {a.community}  {a.role.value:11s}  state {a.state.value}{flag}")
    counts = {r.value: sum(a.role == r for a in agents.values()) for r in Role}
    print("\nn =", len(agents), counts)
