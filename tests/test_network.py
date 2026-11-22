"""Tests for step 2 (network)."""
import networkx as nx

from sim.network import build, distances_from, guide_network, sbm_network


def test_same_seed_same_network():
    assert set(sbm_network(1).edges()) == set(sbm_network(1).edges())


def test_different_seeds_differ():
    assert set(sbm_network(1).edges()) != set(sbm_network(2).edges())


def test_network_is_connected_and_matches_agents():
    for seed in range(20):
        G, agents, start = build(seed)
        assert nx.is_connected(G)
        assert set(G.nodes()) == set(agents)
        assert start in agents


def test_guide_network_distances():
    # distances from agent 1 in the guide example: these are the earliest
    # possible exposure rounds, and match the hand-calculated table
    G, agents, start = guide_network()
    assert distances_from(G, start) == {1: 0, 2: 1, 3: 1, 4: 2, 5: 3, 6: 3}
