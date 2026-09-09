# CO-Bench :: Assignment problem  (category: Assignment)
#
# The Assignment Problem involves optimally assigning  n  items to  n  agents based on a provided  n 	imes n  cost matrix, where each entry \( 	ext{cost\matrix}[i][j] \) denotes the cost of assigning item  i+1  to agent  j+1 . The goal is to identify a permutation—each item assigned exactly one agent—that minimizes the total assignment cost. Formally, this is an optimization problem to find a permutation \pi of agents such that the total cost \sum{i=1}^{n} 	ext{cost\_matrix}[i-1][\pi(i)-1] is minimized. The solution returned includes both the minimal total cost and the corresponding optimal assignments.
#
# # Implement in Solve Function
#
# def solve(**kwargs):
#     """
#     Solves an instance of the Assignment Problem.
#
#     Given n items and an n×n cost matrix (where cost_matrix[i][j] is the cost of assigning
#     item (i+1) to agent (j+1)), the goal is to determine a permutation (a one-to-one assignment
#     between items and agents) that minimizes the total cost. The returned solution is a
#     dictionary with:
#
#       - "total_cost": The sum of the costs of the chosen assignments.
#       - "assignment": A list of n tuples (i, j), where i is the item number (1-indexed)
#                       and j is the assigned agent number (1-indexed).
#
#     Input kwargs:
#       - n: int, the number of items/agents.
#       - cost_matrix: numpy.ndarray, a 2D array with shape (n, n) containing the costs.
#
#     Returns:
#       A dictionary with keys "total_cost" and "assignment" representing the optimal solution.
#     """
#     # Your algorithm implementation goes here.
#     # For example, you may use the Hungarian algorithm.
#     return {"total_cost": None, "assignment": None}

# EVOLVE-BLOCK-START
def solve(**kwargs):
    import numpy as np
    n = kwargs["n"]
    C = np.asarray(kwargs["cost_matrix"], dtype=float)
    if np.isinf(C).any():  # replace inf so the solver stays finite
        finite = C[np.isfinite(C)]
        big = (finite.max() * 1e6) if finite.size else 1e18
        C = np.where(np.isinf(C), big, C)
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(C)
        assignment = [(int(i) + 1, int(j) + 1) for i, j in zip(r, c)]
        total = float(C[r, c].sum())
        return {"total_cost": total, "assignment": assignment}
    except Exception:
        # greedy fallback: cheapest available agent per item (row order)
        used = np.zeros(n, dtype=bool)
        assignment = []
        total = 0.0
        for i in range(n):
            row = C[i].copy()
            row[used] = np.inf
            j = int(np.argmin(row))
            used[j] = True
            assignment.append((i + 1, j + 1))
            total += float(C[i, j])
        return {"total_cost": total, "assignment": assignment}
# EVOLVE-BLOCK-END
