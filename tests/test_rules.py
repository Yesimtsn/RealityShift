"""Tests for step 3 (states, transitions, actions)."""
from sim.agents import Agent, Role, State
from sim.rules import (REFERENCE, STOCHASTIC, Action, check_tables, choose_action,
                       next_state, rng)


def test_tables_are_valid():
    check_tables(STOCHASTIC)
    check_tables(REFERENCE)


def test_keyed_random_is_reproducible():
    assert rng(1, 5, 3, "decide").random() == rng(1, 5, 3, "decide").random()
    assert rng(1, 5, 3, "decide").random() != rng(1, 5, 4, "decide").random()


def test_corrected_is_absorbing_and_never_shares():
    for role in Role:
        a = Agent(id=0, role=role, community=0, state=State.C)
        for t in range(50):
            assert next_state(a, True, STOCHASTIC, seed=1, t=t) == State.C
            assert choose_action(a, STOCHASTIC, seed=1, t=t) in (Action.IGNORE, Action.CORRECT)


def test_unaware_does_not_change_without_exposure():
    a = Agent(id=0, role=Role.SHARER, community=0)
    assert next_state(a, True, STOCHASTIC, seed=1, t=1) == State.U


def test_interpreter_frequencies_match_table():
    n = 20_000
    b = sum(next_state(Agent(id=i, role=Role.INTERPRETER, community=0, state=State.X),
                       False, STOCHASTIC, seed=3, t=1) == State.B for i in range(n))
    assert abs(b / n - 0.6) < 0.02


def test_reference_rules_are_deterministic():
    a = Agent(id=1, role=Role.SHARER, community=0, state=State.X)
    assert next_state(a, False, REFERENCE, seed=0, t=1) == State.B
    a.state = State.B
    assert choose_action(a, REFERENCE, seed=0, t=1) == Action.SHARE
    o = Agent(id=2, role=Role.OBSERVER, community=0, state=State.B)
    assert choose_action(o, REFERENCE, seed=0, t=1) == Action.IGNORE
