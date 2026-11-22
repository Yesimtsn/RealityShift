"""Tests for step 4 (claims, deviation, version tree)."""
import numpy as np
import pytest

from sim.claims import (Version, VersionTree, apply_edit, deviation, make_child,
                        seed_version)
from sim.textgen import TextGen


def g():
    return np.random.default_rng(0)


def test_deviation_examples_from_the_presentation():
    assert deviation(("drop12", "official_source")) == 0                     # C0
    assert deviation(("nearly_empty", "official_source")) == 0.5             # C1
    assert deviation(("nearly_empty", "official_source", "cuts_next_week")) == pytest.approx(2 / 3)  # C2a
    assert deviation(("nearly_empty", "hide_numbers")) == 1.0                # C2b


def test_deviation_hand_example_from_the_guide():
    # "The reservoir fell 12% and the mayor resigned." -> S + N -> 0.25
    assert deviation(("drop12", "mayor_resigned")) == 0.25


def test_edits_change_the_right_facts():
    f, s = apply_edit("exaggerate", ("drop12", "official_source"), (), g())
    assert f == ("nearly_empty", "official_source") and "urgent" in s
    f, _ = apply_edit("change_source", f, s, g())
    assert f == ("nearly_empty", "hide_numbers")
    f, s = apply_edit("correct", f, (), g())
    assert deviation(f) == 0 and "drop12" in f


def test_version_tree_rules_and_tracing():
    tg = TextGen()
    tree = VersionTree()
    tree.add(seed_version(author=0, lang="en", text_gen=tg))
    c1 = make_child(tree, "c0", author=1, t=1, phi="exaggerate", intentional=False, g=g(), text_gen=tg)
    c2 = make_child(tree, c1.id, author=9, t=3, phi="change_source", intentional=True, g=g(), text_gen=tg)
    c3 = make_child(tree, c1.id, author=2, t=3, phi="add_detail", intentional=False, g=g(), text_gen=tg)
    assert tree.check() == []
    assert tree.path_to_root(c2.id) == [c2.id, c1.id, "c0"]
    assert tree.intentional_path(c2.id) and not tree.intentional_path(c3.id)
    assert set(tree.subtree(c1.id)) == {c1.id, c2.id, c3.id}
    assert c1.text.startswith("BREAKING:")


def test_tree_refuses_bad_versions():
    tg = TextGen()
    tree = VersionTree()
    tree.add(seed_version(author=0, lang="en", text_gen=tg))
    bad_round = Version("x", "c0", "c0", 1, 0, "exaggerate", ("nearly_empty",), (), "en", "t", 1.0, False)
    with pytest.raises(ValueError):
        tree.add(bad_round)                      # parent not earlier
    missing = Version("y", "zz", "c0", 1, 2, "exaggerate", ("nearly_empty",), (), "en", "t", 1.0, False)
    with pytest.raises(ValueError):
        tree.add(missing)                        # parent does not exist
    wrong_d = Version("z", "c0", "c0", 1, 2, "exaggerate", ("nearly_empty",), (), "en", "t", 0.3, False)
    with pytest.raises(ValueError):
        tree.add(wrong_d)                        # stored deviation is wrong


def test_turkish_text():
    tg = TextGen()
    tree = VersionTree()
    tree.add(seed_version(author=0, lang="tr", text_gen=tg))
    c1 = make_child(tree, "c0", 1, 1, "exaggerate", False, g(), tg)
    assert c1.text.startswith("SON DAKİKA:")
