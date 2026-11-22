"""Step 5: one simulation round, and a full run.

Round 0: the start agent believes the original claim c0 (it posts it in round 1).
Round t >= 1, in this order:
  1. corrections scheduled for round t are delivered (intervene hook, step 6)
  2. state update for every agent, using what it received in round t-1
  3. agents that received something in round t-1 choose and perform an action
  4. posts of round t are delivered to neighbours; unaware receivers become exposed
So an agent exposed in round t acts in round t+1, and its post is seen in round t+1.
"""
import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

import networkx as nx

from .agents import Agent, State
from .claims import FACTS, ROLE_EDITS, Verdict, VersionTree, make_child, seed_version
from .rules import STOCHASTIC, Action, Rules, choose_action, next_state, rng, sample
from .textgen import TextGen

THETA = 0.5  # a version is false if D >= THETA

# Evidence released at the start (scenario records). Step 6 can release more.
DEFAULT_EVIDENCE = {"ev_report"}

# An agent stays active for ATTENTION rounds after it last received something,
# until it acts (share, modify, verify or correct). "Ignore" means: not this round.
ATTENTION = 3


@dataclass
class Post:
    id: str
    author: int
    round: int
    kind: str                 # "claim" or "correction"
    version: Optional[str]    # claim posts point to a version


@dataclass
class Sim:
    """Everything about one run that changes over time."""
    G: nx.Graph
    agents: Dict[int, Agent]
    rules: Rules
    seed: int
    tree: VersionTree
    text_gen: TextGen
    evidence_open: Set[str] = field(default_factory=lambda: set(DEFAULT_EVIDENCE))
    inbox: Dict[int, List[Post]] = field(default_factory=dict)      # posts received last round
    events: List[dict] = field(default_factory=list)
    rounds: List[dict] = field(default_factory=list)
    n_posts: int = 0
    last_received: Dict[int, int] = field(default_factory=dict)  # round of last post received
    last_acted: Dict[int, int] = field(default_factory=dict)     # round of last non-ignore action

    def log(self, **e) -> None:
        self.events.append(e)


Intervene = Callable[[int, "Sim"], Set[int]]   # (round, sim) -> agents to correct this round


def no_intervention(t: int, sim: Sim) -> Set[int]:
    return set()


def run(G: nx.Graph, agents: Dict[int, Agent], start: int, rules: Rules = STOCHASTIC,
        seed: int = 0, T: int = 10, lang: str = "en", intervene: Intervene = no_intervention,
        text_gen: Optional[TextGen] = None) -> Sim:
    agents = copy.deepcopy(agents)                    # every run starts from the same agents
    sim = Sim(G=G, agents=agents, rules=rules, seed=seed, tree=VersionTree(),
              text_gen=text_gen or TextGen())
    sim.inbox = {i: [] for i in agents}

    # round 0: the start agent has the original claim
    c0 = seed_version(author=start, lang=lang, text_gen=sim.text_gen)
    sim.tree.add(c0)
    a0 = agents[start]
    a0.state = State.B
    a0.memory.received.append((0, c0.id))
    a0.memory.first_exposure_round = 0
    sim.inbox[start] = [Post("p0", start, 0, "claim", c0.id)]
    sim.last_received[start] = 0
    sim.log(round=0, type="seed", agent=start, version=c0.id)
    record_round(sim, 0, new_versions=1)

    for t in range(1, T + 1):
        step(sim, t, intervene)
    return sim


def step(sim: Sim, t: int, intervene: Intervene) -> None:
    agents, rules, seed = sim.agents, sim.rules, sim.seed

    # 1. corrections scheduled for this round
    corrected_now = set(intervene(t, sim))
    for i in corrected_now:
        agents[i].memory.correction_round = t
        sim.log(round=t, type="correction_delivered", agent=i, source="official")
    got_correction = {i for i in agents
                      if i in corrected_now or any(p.kind == "correction" for p in sim.inbox[i])}

    # 2. state update (all agents at once)
    new_states = {i: next_state(a, i in got_correction, rules, seed, t) for i, a in agents.items()}
    for i, s in new_states.items():
        if s != agents[i].state:
            sim.log(round=t, type="state", agent=i, before=agents[i].state.value, after=s.value)
            agents[i].state = s

    # 3. actions by active agents: received something in the last ATTENTION rounds
    #    and has not acted since, or was corrected this round
    def is_active(i: int) -> bool:
        r = sim.last_received.get(i)
        recent = r is not None and r >= t - ATTENTION and sim.last_acted.get(i, -1) < r
        return recent or i in corrected_now
    active = sorted(i for i in agents if is_active(i))
    outbox: List[Post] = []
    versions_before = len(sim.tree)
    for i in active:
        a = agents[i]
        act = choose_action(a, rules, seed, t)
        if act == Action.IGNORE:
            continue
        sim.last_acted[i] = t
        latest = a.memory.latest_version()
        if act in (Action.SHARE, Action.MODIFY):
            if latest is None or a.state == State.C:          # cannot happen with valid tables
                sim.log(round=t, type="violation", agent=i, rule="share without version or while corrected")
                continue
            if act == Action.SHARE:
                vid = latest
            else:
                phi = sample(ROLE_EDITS[a.role.value], rng(seed, i, t, "edit"))
                child = make_child(sim.tree, latest, author=i, t=t, phi=phi,
                                   intentional=a.deceptive, g=rng(seed, i, t, "pick"),
                                   text_gen=sim.text_gen)
                vid = child.id
                sim.log(round=t, type="version", agent=i, version=vid, parent=latest,
                        edit=phi, deviation=child.deviation, intentional=a.deceptive)
            outbox.append(new_post(sim, i, t, "claim", vid))
            sim.log(round=t, type="post", agent=i, kind="claim", version=vid, action=act.value)
        elif act == Action.VERIFY:
            verify(sim, a, t)
        elif act == Action.CORRECT:
            outbox.append(new_post(sim, i, t, "correction", None))
            sim.log(round=t, type="post", agent=i, kind="correction", action=act.value)

    # 4. delivery
    sim.inbox = {i: [] for i in agents}
    for p in outbox:
        for j in sim.G.neighbors(p.author):
            sim.inbox[j].append(p)
            sim.last_received[j] = t
            if p.kind == "claim":
                m = agents[j].memory
                m.received.append((t, p.version))
                if agents[j].state == State.U:
                    agents[j].state = State.X
                    m.first_exposure_round = t
                    sim.log(round=t, type="state", agent=j, before="U", after="X", version=p.version)
    record_round(sim, t, new_versions=len(sim.tree) - versions_before)


def new_post(sim: Sim, author: int, t: int, kind: str, version: Optional[str]) -> Post:
    sim.n_posts += 1
    p = Post(f"p{sim.n_posts}", author, t, kind, version)
    sim.agents[author].memory.own_posts.append(p.id)
    return p


def verify(sim: Sim, a: Agent, t: int) -> None:
    """The agent checks the latest version it holds against released evidence:
      a refuted fact whose evidence is released     -> C (corrected)
      every fact supported by released evidence      -> B (confirmed, it is true)
      otherwise (missing or unsupported evidence)    -> D (doubts)"""
    vid = a.memory.latest_version()
    v = sim.tree.versions[vid]
    released = lambda f: FACTS[f].evidence in sim.evidence_open
    refutable = [f for f in v.facts if FACTS[f].verdict == Verdict.R and released(f)]
    confirmed = all(FACTS[f].verdict == Verdict.S and released(f) for f in v.facts)
    before = a.state
    a.state = State.C if refutable else (State.B if confirmed else State.D)
    sim.log(round=t, type="verify", agent=a.id, version=vid, refuted=bool(refutable), confirmed=confirmed,
            evidence=[FACTS[f].evidence for f in refutable])
    if a.state != before:
        sim.log(round=t, type="state", agent=a.id, before=before.value, after=a.state.value)


def record_round(sim: Sim, t: int, new_versions: int) -> None:
    agents, tree = sim.agents, sim.tree
    n = len(agents)
    exposed = [a for a in agents.values() if a.memory.first_exposure_round is not None]
    saw_false = [a for a in agents.values()
                 if any(tree.versions[v].deviation >= THETA for _, v in a.memory.received)]
    counts = {s.value: sum(a.state == s for a in agents.values()) for s in State}
    sim.rounds.append({
        "round": t,
        "reach": len(exposed) / n,               # R(t): saw any version
        "reach_false": len(saw_false) / n,       # saw a false version (D >= theta)
        **counts,
        "versions": len(tree),
        "new_versions": new_versions,
    })


if __name__ == "__main__":
    from .network import build
    G, agents, start = build(seed=1)
    sim = run(G, agents, start, seed=1, T=10)
    print("round  reach  reach_false   U  X  B  D  C  versions")
    for r in sim.rounds:
        print(f"{r['round']:5d}  {r['reach']:.2f}   {r['reach_false']:.2f}       "
              f"{r['U']:2d} {r['X']:2d} {r['B']:2d} {r['D']:2d} {r['C']:2d}  {r['versions']}")
    print("\ntree:", sim.tree.stats(), "| rule check:", sim.tree.check() or "ok")
    for v in sim.tree.versions.values():
        print(f"  {v.id:4s} <- {str(v.parent):4s} {v.edit:13s} D={v.deviation:.2f} agent {v.author}: {v.text}")
