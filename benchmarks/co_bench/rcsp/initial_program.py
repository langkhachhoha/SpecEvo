# CO-Bench :: Resource constrained shortest path  (category: Routing)
#
# This problem involves finding the shortest path from vertex 1 to vertex n in a directed graph while satisfying resource constraints. Specifically, each vertex and arc has associated resource consumptions, and the cumulative consumption for each resource must fall within the provided lower_bounds and upper_bounds. The input includes the number of vertices (n), arcs (m), resource types (K), resource consumption at each vertex, and a graph represented as a mapping from vertices to lists of arcs (each arc being a tuple of end vertex, cost, and arc resource consumptions). The optimization objective is to minimize the total arc cost of the path, with the condition that the path is valid—meaning it starts at vertex 1, ends at vertex n, follows defined transitions in the graph, and respects all resource bounds; if any of these constraints are not met, the solution receives no score.
#
# # Implement in Solve Function
#
# def solve(**kwargs):
#     """
#     Solve the Resource Constrained Shortest Path problem.
#
#     Input kwargs should include:
#       - n (int): number of vertices,
#       - m (int): number of arcs,
#       - K (int): number of resources,
#       - lower_bounds (list of float): list of lower limits for each resource,
#       - upper_bounds (list of float): list of upper limits for each resource,
#       - vertex_resources (list of list of float): list (of length n) of lists (of length K) with the resource consumption at each vertex,
#       - graph (dict): dictionary mapping each vertex (1-indexed) to a list of arcs, where each arc is a tuple
#                       (end_vertex (int), cost (float), [arc resource consumptions] (list of float)).
#
#
#     Evaluation Metric:
#       If the computed path is valid (i.e. it starts at vertex 1, ends at vertex n, every transition is
#       defined in the graph, and the total resource consumption from both vertices and arcs is within the
#       specified bounds for each resource), then the score equals the total arc cost along the path.
#       Otherwise, the solution is invalid and receives no score.
#
#     Returns:
#       A dictionary with keys:
#          "total_cost": total cost (a float) of the computed path,
#          "path": a list of vertex indices (integers) defining the path.
#
#     (Placeholder implementation)
#     """
#     # Placeholder implementation.
#     n = kwargs.get("n", 1)
#     # Return a trivial solution: just go directly from vertex 1 to vertex n.
#     return {"total_cost": 0.0, "path": [1, n]}

# EVOLVE-BLOCK-START
def solve(**kwargs):
    import heapq
    n = kwargs["n"]
    K = kwargs["K"]
    lb = kwargs["lower_bounds"]
    ub = kwargs["upper_bounds"]
    vres = kwargs["vertex_resources"]
    graph = kwargs["graph"]

    start_res = tuple(vres[0][k] for k in range(K))
    if any(start_res[k] > ub[k] + 1e-6 for k in range(K)):
        return {"total_cost": 0.0, "path": [1, n]}

    use_dom = all(lb[k] <= 1e-9 for k in range(K))  # dominance is only sound w/o >= bounds
    labels = [(0.0, start_res, 1, -1)]  # (cost, res, vertex, parent)
    node_labels = {i: [] for i in range(1, n + 1)}
    node_labels[1].append(0)
    pq = [(0.0, 0)]
    CAP = 200000
    best = None

    def dominated(v, cost, res):
        for li in node_labels[v]:
            lc, lr, _, _ = labels[li]
            if lc <= cost + 1e-9 and all(lr[k] <= res[k] + 1e-9 for k in range(K)):
                return True
        return False

    while pq and len(labels) < CAP:
        c, li = heapq.heappop(pq)
        cost, res, u, par = labels[li]
        if c > cost + 1e-9:
            continue
        if u == n and all(res[k] >= lb[k] - 1e-6 for k in range(K)):
            best = li
            break
        for (v, ac, ar) in graph.get(u, []):
            nres = tuple(res[k] + ar[k] + vres[v - 1][k] for k in range(K))
            if any(nres[k] > ub[k] + 1e-6 for k in range(K)):
                continue
            if use_dom and dominated(v, cost + ac, nres):
                continue
            idx = len(labels)
            labels.append((cost + ac, nres, v, li))
            node_labels[v].append(idx)
            heapq.heappush(pq, (cost + ac, idx))
            if len(labels) >= CAP:
                break

    if best is None:
        return {"total_cost": 0.0, "path": [1, n]}
    path = []
    li = best
    while li != -1:
        path.append(labels[li][2])
        li = labels[li][3]
    path.reverse()
    return {"total_cost": labels[best][0], "path": path}
# EVOLVE-BLOCK-END
