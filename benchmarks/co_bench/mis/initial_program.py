# CO-Bench :: Maximal independent set  (category: Graph & set)
#
# The Maximum Independent Set (MIS) problem is a fundamental NP-hard optimization problem in graph theory. Given an undirected graph G = (V, E), where V is a set of vertices and E is a set of edges, the goal is to find the largest subset S ⊆ V such that no two vertices in S are adjacent (i.e., connected by an edge).
#
# # Implement in Solve Function
#
# def solve(**kwargs):
#     """
#     Solve the Maximum Independent Set problem for a given test case.
#
#    Input:
#         kwargs (dict): A dictionary with the following keys:
#             - graph (networkx.Graph): The graph to solve
#
#     Returns:
#         dict: A solution dictionary containing:
#             - mis_nodes (list): List of node indices in the maximum independent set
#     """
#     # TODO: Implement your MIS solving algorithm here. Below is a placeholder.
#     solution = {
#         'mis_nodes': [0, 1, ...],
#     }
#     return solution

# EVOLVE-BLOCK-START
def solve(**kwargs):
    import heapq
    G = kwargs["graph"]
    adj = {v: set(G.neighbors(v)) for v in G.nodes()}
    deg = {v: len(adj[v]) for v in adj}
    alive = set(adj.keys())
    heap = [(deg[v], v) for v in adj]
    heapq.heapify(heap)
    chosen = []
    # greedy minimum-degree independent set
    while heap:
        d, v = heapq.heappop(heap)
        if v not in alive or d != deg[v]:
            continue
        chosen.append(v)
        alive.discard(v)
        for u in list(adj[v]):
            if u in alive:
                alive.discard(u)          # neighbours can't join the set
                for w in adj[u]:
                    if w in alive:
                        deg[w] -= 1
                        heapq.heappush(heap, (deg[w], w))
    return {"mis_nodes": chosen}
# EVOLVE-BLOCK-END
