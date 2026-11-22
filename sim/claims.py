"""Step 4: claims, edits, deviation and the version tree.

Fact list F: small statements about the scenario, each with a verdict
    v(a) in {S, R, N}   S = supported by F, R = refuted by F, N = not in F
Deviation of a claim version c with facts A(c):
    D(c) = (#R + 0.5 * #N) / |A(c)|
A new version comes from its parent by one edit:
    c_new = T(c_parent, phi, xi),  phi in {exaggerate, add_detail, change_source, correct}
Each edit has a fixed *fact edit*, so the true D of every version is known.
"""
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

LAMBDA = 0.5  # weight of an unsupported (N) fact


class Verdict(str, Enum):
    S = "S"
    R = "R"
    N = "N"


@dataclass(frozen=True)
class Fact:
    id: str
    en: str
    tr: str
    verdict: Verdict
    evidence: Optional[str] = None  # id of the scenario record that settles it


# Scenario: fictional city Northtown / Kuzeykent. No real people or places.
FACTS: Dict[str, Fact] = {f.id: f for f in [
    Fact("drop12", "the reservoir level fell 12% this month",
         "baraj doluluk oranı bu ay %12 düştü", Verdict.S, "ev_report"),
    Fact("official_source", "according to the Northtown Water Authority report",
         "Kuzeykent Su İdaresi raporuna göre", Verdict.S, "ev_report"),
    Fact("seasonal", "drops like this are normal in the dry season",
         "bu tür düşüşler kurak mevsimde olağan", Verdict.S, "ev_capacity"),
    Fact("nearly_empty", "the reservoir is nearly empty",
         "baraj neredeyse boş", Verdict.R, "ev_capacity"),
    Fact("cuts_next_week", "water cuts start next week",
         "su kesintileri gelecek hafta başlıyor", Verdict.R, "ev_no_cuts"),
    Fact("hide_numbers", "officials are hiding the real numbers",
         "yetkililer gerçek rakamları saklıyor", Verdict.R, "ev_open_data"),
    Fact("cause_hotels", "the drop is caused by the new hotel district",
         "düşüşün nedeni yeni otel bölgesi", Verdict.N),
    Fact("mayor_resigned", "the mayor resigned",            # used in the guide's hand example
         "belediye başkanı istifa etti", Verdict.N),
]}

SEED_FACTS: Tuple[str, ...] = ("drop12", "official_source")
EDITS = ("exaggerate", "add_detail", "change_source", "correct")

# Which facts each edit is allowed to introduce (used to detect rule violations)
ALLOWED_NEW = {
    "exaggerate": {"nearly_empty"},
    "add_detail": {"cuts_next_week", "cause_hotels"},
    "change_source": {"hide_numbers"},
    "correct": {"drop12", "official_source", "seasonal"},
}

# Which edit a role makes when it modifies: P(phi | role)
ROLE_EDITS = {
    "sharer": {"exaggerate": 1.0},
    "interpreter": {"add_detail": 0.5, "change_source": 0.5},
    "verifier": {"correct": 1.0},
    "observer": {"exaggerate": 1.0},
}


def deviation(facts) -> float:
    facts = list(facts)
    if not facts:
        return 0.0
    r = sum(FACTS[f].verdict == Verdict.R for f in facts)
    n = sum(FACTS[f].verdict == Verdict.N for f in facts)
    return (r + LAMBDA * n) / len(facts)


def apply_edit(phi: str, facts: Tuple[str, ...], style: Tuple[str, ...],
               g: np.random.Generator) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """The fact edit of T. Returns the child's (facts, style)."""
    f, s = list(facts), list(style)
    if phi == "exaggerate":
        if "drop12" in f:
            f[f.index("drop12")] = "nearly_empty"
        if "urgent" not in s:
            s.append("urgent")
    elif phi == "add_detail":
        options = [x for x in ("cuts_next_week", "cause_hotels") if x not in f]
        if options:
            f.append(options[int(g.integers(len(options)))])
    elif phi == "change_source":
        if "official_source" in f:
            f[f.index("official_source")] = "hide_numbers"
        elif "hide_numbers" not in f:
            f.append("hide_numbers")
        if "alarm" not in s:
            s.append("alarm")
    elif phi == "correct":
        f = [x for x in f if FACTS[x].verdict == Verdict.S]   # drop false and unsupported facts
        for core in SEED_FACTS + ("seasonal",):
            if core not in f:
                f.append(core)
        s = []
    else:
        raise ValueError(f"unknown edit {phi}")
    return tuple(f), tuple(s)


# ---------- versions and the version tree ----------
@dataclass
class Version:
    id: str
    parent: Optional[str]
    root: str
    author: int
    round: int
    edit: str                  # phi, or "seed" for c0
    facts: Tuple[str, ...]
    style: Tuple[str, ...]
    lang: str
    text: str
    deviation: float
    intentional: bool          # intent flag of the edit that made this version
    synthetic: bool = True     # always: simulated content is never evidence

    def to_dict(self) -> dict:
        return asdict(self)


class VersionTree:
    """All claim versions of one run. add() refuses anything that breaks the rules:
      1. exactly one root; every other version has exactly one existing parent
      2. a parent is created in an earlier round than its child
      3. root id is inherited from the parent
      4. stored deviation = D(facts)
      5. every version is labelled synthetic
    (rule 'an agent can only edit a version in its memory' is checked in the round loop)"""

    def __init__(self) -> None:
        self.versions: Dict[str, Version] = {}
        self.children: Dict[str, List[str]] = {}

    def __len__(self) -> int:
        return len(self.versions)

    def add(self, v: Version) -> None:
        if v.id in self.versions:
            raise ValueError(f"duplicate id {v.id}")
        if v.parent is None:
            if self.versions:
                raise ValueError("only one root allowed")
            if v.root != v.id:
                raise ValueError("root must be its own root")
        else:
            p = self.versions.get(v.parent)
            if p is None:
                raise ValueError(f"parent {v.parent} does not exist")
            if not p.round < v.round:
                raise ValueError("parent must be from an earlier round")
            if v.root != p.root:
                raise ValueError("root id must match the parent's")
        if abs(v.deviation - deviation(v.facts)) > 1e-12:
            raise ValueError("stored deviation does not match the facts")
        if not v.synthetic:
            raise ValueError("versions must be labelled synthetic")
        self.versions[v.id] = v
        self.children[v.id] = []
        if v.parent is not None:
            self.children[v.parent].append(v.id)

    def path_to_root(self, vid: str) -> List[str]:
        path = [vid]
        while self.versions[path[-1]].parent is not None:
            path.append(self.versions[path[-1]].parent)
        return path

    def intentional_path(self, vid: str) -> bool:
        """iota(c): true if any edit between c0 and c was intentional."""
        return any(self.versions[x].intentional for x in self.path_to_root(vid))

    def subtree(self, vid: str) -> List[str]:
        """c and every version derived from it (used to pick who gets a correction)."""
        out, stack = [], [vid]
        while stack:
            x = stack.pop()
            out.append(x)
            stack.extend(self.children[x])
        return out

    def check(self) -> List[str]:
        """Re-checks every rule on the finished tree. Empty list = valid."""
        errors = []
        roots = [v for v in self.versions.values() if v.parent is None]
        if len(roots) != 1:
            return [f"expected 1 root, found {len(roots)}"]
        r = roots[0].id
        for v in self.versions.values():
            if v.parent is not None and self.versions[v.parent].round >= v.round:
                errors.append(f"{v.id}: parent not earlier")
            if v.root != r:
                errors.append(f"{v.id}: wrong root")
            if self.path_to_root(v.id)[-1] != r:
                errors.append(f"{v.id}: does not reach the root")
            if abs(v.deviation - deviation(v.facts)) > 1e-12:
                errors.append(f"{v.id}: wrong deviation")
            if v.parent is not None:
                new = set(v.facts) - set(self.versions[v.parent].facts)
                if not new <= ALLOWED_NEW.get(v.edit, set()):
                    errors.append(f"{v.id}: edit {v.edit} added facts it may not add: {new}")
        return errors

    def stats(self) -> dict:
        depth = {vid: len(self.path_to_root(vid)) - 1 for vid in self.versions}
        with_children = [k for k, c in self.children.items() if c]
        return {
            "versions": len(self.versions),
            "max_depth": max(depth.values()) if depth else 0,
            "mean_children": (sum(len(self.children[k]) for k in with_children) / len(with_children)
                              if with_children else 0.0),
            "max_deviation": max(v.deviation for v in self.versions.values()) if self.versions else 0.0,
        }


def seed_version(author: int, lang: str, text_gen) -> Version:
    facts, style = SEED_FACTS, ()
    text = text_gen.write(None, "seed", facts, style, lang, call_key="c0")
    return Version(id="c0", parent=None, root="c0", author=author, round=0, edit="seed",
                   facts=facts, style=style, lang=lang, text=text, deviation=deviation(facts),
                   intentional=False)


def make_child(tree: VersionTree, parent_id: str, author: int, t: int, phi: str,
               intentional: bool, g: np.random.Generator, text_gen) -> Version:
    p = tree.versions[parent_id]
    facts, style = apply_edit(phi, p.facts, p.style, g)
    vid = f"c{len(tree)}"
    text = text_gen.write(p.text, phi, facts, style, p.lang, call_key=f"{vid}|{parent_id}|{author}|{t}")
    v = Version(id=vid, parent=parent_id, root=p.root, author=author, round=t, edit=phi,
                facts=facts, style=style, lang=p.lang, text=text, deviation=deviation(facts),
                intentional=intentional)
    tree.add(v)
    return v
