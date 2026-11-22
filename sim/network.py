"""Step 2: the network G = (V, E).

Undirected: (i, j) in E means i and j see each other's posts.
N(i) = {j : (i, j) in E} are the neighbours of agent i.

The simulation network is a stochastic block model (SBM): each pair of agents
in the same community is linked with probability p_in, each pair in different
communities with probability p_out. Same seed -> same network.
"""
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx

from .agents import Agent, Role, make_agents

P_IN = 0.5    # link probability inside a community
P_OUT = 0.04  # link probability between communities (few bridges)


def sbm_network(seed: int, sizes: Sequence[int] = (7, 7, 6),
                p_in: float = P_IN, p_out: float = P_OUT) -> nx.Graph:
    """Random network with communities. Nodes 0..n-1 are in the same order as
    make_agents(), so node i is agent i. If the drawn graph is not connected,
    it is redrawn with seed + 1000, seed + 2000, ... (still deterministic)."""
    k = len(sizes)
    probs = [[p_in if a == b else p_out for b in range(k)] for a in range(k)]
    attempt = 0
    while True:
        G = nx.stochastic_block_model(list(sizes), probs, seed=seed + 1000 * attempt)
        if nx.is_connected(G):
            G.graph["seed"] = seed
            G.graph["attempt"] = attempt
            return nx.Graph(G)  # plain graph (drop block metadata)
        attempt += 1


def build(seed: int, sizes: Sequence[int] = (7, 7, 6),
          deceptive_community: Optional[int] = 1) -> Tuple[nx.Graph, Dict[int, Agent], int]:
    """Agents + network + the agent who starts with the claim.
    The first sharer in community 0 posts the original claim c0."""
    agents = make_agents(sizes, deceptive_community)
    G = sbm_network(seed, sizes)
    start = next(a.id for a in agents.values() if a.community == 0 and a.role == Role.SHARER)
    return G, agents, start


def guide_network() -> Tuple[nx.Graph, Dict[int, Agent], int]:
    """Fixed 6-agent network from 'How to formulate your design' (page 4).
    Agent 1 starts with the claim; agent 2 is an observer. Used for the
    hand-calculated reach test."""
    G = nx.Graph()
    G.add_nodes_from(range(1, 7))
    G.add_edges_from([(1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (4, 6), (5, 6)])
    agents = {i: Agent(id=i, role=Role.SHARER, community=0) for i in range(1, 7)}
    agents[2].role = Role.OBSERVER
    return G, agents, 1


# ---------- helpers used later (reach, bridges, report) ----------

def neighbours(G: nx.Graph, i: int) -> List[int]:
    return sorted(G.neighbors(i))


def bridges(G: nx.Graph, agents: Dict[int, Agent]) -> List[Tuple[int, int]]:
    """Links between different communities."""
    return sorted((min(i, j), max(i, j)) for i, j in G.edges()
                  if agents[i].community != agents[j].community)


def distances_from(G: nx.Graph, start: int) -> Dict[int, int]:
    """Graph distance d(start, j). An agent at distance d cannot be exposed
    before round d (each hop takes at least one round)."""
    return dict(nx.single_source_shortest_path_length(G, start))


def summary(G: nx.Graph, agents: Dict[int, Agent], start: int) -> str:
    d = distances_from(G, start)
    lines = [
        f"agents n = {G.number_of_nodes()}, links |E| = {G.number_of_edges()}",
        f"mean degree = {2 * G.number_of_edges() / G.number_of_nodes():.2f}",
        f"bridges between communities = {bridges(G, agents)}",
        f"start agent = {start} (community {agents[start].community}, {agents[start].role.value})",
        f"max distance from start = {max(d.values())}",
    ]
    return "\n".join(lines)


def draw(G: nx.Graph, agents: Dict[int, Agent], path: str, start: Optional[int] = None) -> None:
    """Saves a picture of the network: colour = community, label = id + role letter."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
    pos = nx.spring_layout(G, seed=7)
    labels = {i: f"{i}{agents[i].role.value[0].upper()}" for i in G.nodes()}
    node_colours = [colours[agents[i].community % len(colours)] for i in G.nodes()]
    edge_colours = ["#D62728" if agents[i].community != agents[j].community else "#BBBBBB"
                    for i, j in G.edges()]
    plt.figure(figsize=(7, 5))
    nx.draw_networkx_edges(G, pos, edge_color=edge_colours, width=1.5)
    nx.draw_networkx_nodes(G, pos, node_color=node_colours, node_size=520,
                           edgecolors=["black" if i == start else "white" for i in G.nodes()],
                           linewidths=2.5)
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color="white")
    plt.title("Network: colour = community, red links = bridges, black ring = start agent\n"
              "S sharer, I interpreter, V verifier, O observer", fontsize=9)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


if __name__ == "__main__":
    import os
    G, agents, start = build(seed=1)
    print(summary(G, agents, start))
    os.makedirs("outputs", exist_ok=True)
    draw(G, agents, "outputs/network_seed1.png", start)
    print("picture saved to outputs/network_seed1.png")
