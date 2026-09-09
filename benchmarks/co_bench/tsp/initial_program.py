# CO-Bench :: Travelling salesman problem  (category: Routing)
#
# The Traveling Salesman Problem (TSP) is a classic combinatorial optimization problem where, given a set of cities with known pairwise distances, the objective is to find the shortest possible tour that visits each city exactly once and returns to the starting city. More formally, given a complete graph G = (V, E) with vertices V representing cities and edges E with weights representing distances, we seek to find a Hamiltonian cycle (a closed path visiting each vertex exactly once) of minimum total weight.
#
# # Implement in Solve Function
#
# def solve(**kwargs):
#     """
#     Solve a TSP instance.
#
#     Args:
#         - nodes (list): List of (x, y) coordinates representing cities in the TSP problem
#                      Format: [(x1, y1), (x2, y2), ..., (xn, yn)]
#
#     Returns:
#         dict: Solution information with:
#             - 'tour' (list): List of node indices representing the solution path
#                             Format: [0, 3, 1, ...] where numbers are indices into the nodes list
#     """
#
#     return {
#         'tour': [],
#     }

# EVOLVE-BLOCK-START
def solve(**kwargs):
    nodes = kwargs["nodes"]
    n = len(nodes)
    if n <= 1:
        return {"tour": list(range(n))}
    # Baseline: nearest-neighbour tour from city 0.
    unvisited = set(range(1, n))
    tour = [0]
    cur = 0
    while unvisited:
        nxt = min(unvisited, key=lambda j:
                  (nodes[cur][0] - nodes[j][0]) ** 2 + (nodes[cur][1] - nodes[j][1]) ** 2)
        tour.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    return {"tour": tour}
# EVOLVE-BLOCK-END
