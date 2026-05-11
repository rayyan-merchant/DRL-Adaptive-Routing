import networkx as nx


def build_graph():
    G = nx.Graph()
    G.add_edges_from([
        (0, 2, {"bw": 10, "delay": 2}), (0, 3, {"bw": 7, "delay": 4}),
        (1, 3, {"bw": 7, "delay": 4}), (1, 4, {"bw": 10, "delay": 2}),
        (2, 3, {"bw": 5, "delay": 5}), (3, 4, {"bw": 5, "delay": 5}),
        (2, 5, {"bw": 8, "delay": 8}), (3, 5, {"bw": 6, "delay": 6}),
        (4, 5, {"bw": 4, "delay": 10}), (2, 4, {"bw": 3, "delay": 12}),
    ])
    return G


def k_paths(G, src, dst, k=3):
    gen = nx.shortest_simple_paths(G, src, dst, weight="delay")
    return [next(gen) for _ in range(k)]


G = build_graph()
PATHS = {
    (0, 5): k_paths(G, 0, 5),
    (1, 5): k_paths(G, 1, 5),
}
