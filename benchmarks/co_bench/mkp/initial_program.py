# CO-Bench :: Multidimensional knapsack problem  (category: Packing)
#
# This problem is a multidimensional knapsack optimization where the objective is to maximize the total profit by selecting decision variables, each associated with a profit and resource consumption across multiple constraints. The decision variables must be chosen such that the sum of resource usage for each constraint does not exceed its corresponding capacity. Importantly, if any constraint is violated—that is, if the resource consumption for any constraint exceeds its allowed capacity—the solution is deemed infeasible and earns no score. The challenge lies in identifying the optimal combination of items that yields the highest total profit while strictly satisfying all resource constraints.
#
# # Implement in Solve Function
#
# def solve(**kwargs):
#     """
#     Solves a multidimensional knapsack problem instance.
#
#     Input kwargs (for one test case):
#       - n: int, number of decision variables.
#       - m: int, number of constraints.
#       - p: list of floats, profit coefficients (length n).
#       - r: list of m lists, each of length n, representing the resource consumption per constraint.
#       - b: list of floats, right-hand side values for each constraint (length m).
#
#     Evaluation metric:
#     The score is computed as:
#         score = sum(p[j] * x[j] for j in range(n))
#     if and only if all constraints are satisfied—that is, for every constraint i, the total resource consumption
#         sum(r[i][j] * x[j] for j in range(n))
#     does not exceed b[i].
#     If any constraint is violated, the solution receives no score. A higher score is better.
#
#     Returns:
#       A dict with key 'x' whose value is a list of n binary decisions (0 or 1).
#     """
#     # Placeholder implementation: a dummy solution that selects no items.
#     x = [0] * kwargs['n']
#     return {'x': x}

# EVOLVE-BLOCK-START
def solve(**kwargs):
    n = kwargs["n"]
    m = kwargs["m"]
    p = kwargs["p"]
    r = kwargs["r"]
    b = kwargs["b"]
    x = [0] * n
    used = [0.0] * m

    def ratio(j):
        denom = 0.0
        for i in range(m):
            if b[i] > 0:
                denom += r[i][j] / b[i]
        return (p[j] / denom) if denom > 1e-12 else p[j]

    order = sorted(range(n), key=lambda j: -ratio(j))
    for j in order:
        ok = True
        for i in range(m):
            if used[i] + r[i][j] > b[i] + 1e-9:
                ok = False
                break
        if ok:
            x[j] = 1
            for i in range(m):
                used[i] += r[i][j]
    return {"x": x}
# EVOLVE-BLOCK-END
