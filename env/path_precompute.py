import networkx as nx
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.hyperparams import K_PATHS

def build_graph():
    G = nx.Graph()
    G.add_edge(0, 2, bw=10, delay=2,  capacity_bps=10e6)
    G.add_edge(0, 3, bw=7,  delay=4,  capacity_bps=7e6)
    G.add_edge(1, 3, bw=7,  delay=4,  capacity_bps=7e6)
    G.add_edge(1, 4, bw=10, delay=2,  capacity_bps=10e6)
    G.add_edge(2, 3, bw=5,  delay=5,  capacity_bps=5e6)
    G.add_edge(3, 4, bw=5,  delay=5,  capacity_bps=5e6)
    G.add_edge(2, 5, bw=8,  delay=8,  capacity_bps=8e6)
    G.add_edge(3, 5, bw=6,  delay=6,  capacity_bps=6e6)
    G.add_edge(4, 5, bw=4,  delay=10, capacity_bps=4e6)
    G.add_edge(2, 4, bw=3,  delay=12, capacity_bps=3e6)
    return G

def get_k_paths(G, source, target, k=K_PATHS):
    path_generator = nx.shortest_simple_paths(G, source, target, weight='delay')
    paths = []
    try:
        for _ in range(k):
            paths.append(next(path_generator))
    except StopIteration:
        pass
    return paths

G = build_graph()

PATHS = {
    (0, 5): get_k_paths(G, 0, 5),   # S1 -> D1
    (1, 5): get_k_paths(G, 1, 5),   # S2 -> D1
}

SD_PAIRS = {
    0: (0, 5),   # index 0 = S1->D1
    1: (1, 5),   # index 1 = S2->D1
}

def get_path(sd_idx, action):
    return PATHS[SD_PAIRS[sd_idx]][action]

def get_link_indices_on_path(path):
    LINK_ENDPOINTS = [
        (0,2),(0,3),(1,3),(1,4),(2,3),(3,4),(2,5),(3,5),(4,5),(2,4)
    ]
    indices = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        for idx, (eu, ev) in enumerate(LINK_ENDPOINTS):
            if (eu == u and ev == v) or (eu == v and ev == u):
                indices.append(idx)
                break
    return indices

if __name__ == "__main__":
    for sd, paths in PATHS.items():
        print(f"SD {sd} S{sd[0]+1}->D{sd[1]-4}:")
        for i, p in enumerate(paths):
            print(f"  Action {i}: {p}")