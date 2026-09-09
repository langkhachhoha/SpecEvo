# CO-Bench :: Multi-Demand Multidimensional Knapsack problem  (category: Packing)
#
# The Multi-Demand Multidimensional Knapsack Problem (MDMKP) is a binary optimization problem that extends the classical MKP by incorporating both upper-bound (≤) and lower-bound (≥) constraints. Formally, given n decision variables x_j \in \{0,1\}, the goal is to maximize \sum_{j=1}^n c_j x_j subject to \sum_{j=1}^n a_{ij} x_j \le b_i for i=1,\dots,m and \sum_{j=1}^n a_{ij} x_j \ge b_i for i=m+1,\dots,m+q. Instances are generated from standard MKP problems by varying the number of ≥ constraints (with q taking values 1, m/2, or m) and by using two types of cost coefficients (positive and mixed), thereby producing six distinct variants per base instance. This formulation enables rigorous evaluation of algorithms in contexts where both resource limits and demand fulfillment must be simultaneously addressed.
#
# # Implement in Solve Function
#
# def solve(**kwargs):
#     """
#     Solve a given MDMKP test instance.
#
#     Input (via kwargs):
#       - n: int
#           Number of decision variables.
#       - m: int
#           Number of <= constraints.
#       - q: int
#           Number of active >= constraints (subset of the full set).
#       - A_leq: list of lists of int
#           Coefficient matrix for <= constraints (dimensions: m x n).
#       - b_leq: list of int
#           Right-hand side for <= constraints (length m).
#       - A_geq: list of lists of int
#           Coefficient matrix for >= constraints (dimensions: q x n).
#       - b_geq: list of int
#           Right-hand side for >= constraints (length q).
#       - cost_vector: list of int
#           Objective function coefficients (length n).
#       - cost_type: str
#           Type of cost coefficients ("positive" or "mixed").
#
#     Output:
#       A dictionary with the following keys:
#         - 'optimal_value': int/float
#              The optimal objective function value (if found).
#         - 'x': list of int
#              Binary vector (0 or 1) representing the decision variable assignment.
#
#     TODO: Implement the actual solution algorithm for the MDMKP instance.
#     """
#     # TODO: Define your model variables, constraints, and objective function.
#     # For example, you might use an integer programming solver (e.g., PuLP, Gurobi, or another solver)
#     # to model and solve the instance.
#
#     # Placeholder solution:
#     solution = {
#         'optimal_value': None,  # Replace with the computed objective value.
#         'x': [0] * kwargs.get('n', 0),  # Replace with the computed decision vector.
#     }
#     return solution

# EVOLVE-BLOCK-START
def solve(**kwargs):
    n = kwargs["n"]
    m = kwargs["m"]
    q = kwargs["q"]
    A_leq = kwargs["A_leq"]
    b_leq = kwargs["b_leq"]
    A_geq = kwargs["A_geq"]
    b_geq = kwargs["b_geq"]
    c = kwargs["cost_vector"]

    x = [0] * n
    leq = [0] * m
    geq = [0] * q

    def can_add(j):
        for i in range(m):
            if leq[i] + A_leq[i][j] > b_leq[i]:
                return False
        return True

    # Phase 1: greedily satisfy the >= (demand) constraints while respecting <=.
    for _ in range(n):
        deficits = [max(0, b_geq[i] - geq[i]) for i in range(q)]
        if sum(deficits) <= 0:
            break
        best, best_val = None, 0
        for j in range(n):
            if x[j] == 1 or not can_add(j):
                continue
            contrib = 0
            for i in range(q):
                if deficits[i] > 0:
                    contrib += min(A_geq[i][j], deficits[i])
            if contrib > best_val:
                best_val, best = contrib, j
        if best is None:
            break
        x[best] = 1
        for i in range(m):
            leq[i] += A_leq[i][best]
        for i in range(q):
            geq[i] += A_geq[i][best]

    feasible = all(geq[i] >= b_geq[i] for i in range(q))
    if feasible:
        # Phase 2: add profitable items that keep <= feasible.
        for j in sorted(range(n), key=lambda k: -c[k]):
            if x[j] == 1 or c[j] <= 0:
                continue
            if can_add(j):
                x[j] = 1
                for i in range(m):
                    leq[i] += A_leq[i][j]
    return {"x": x}
# EVOLVE-BLOCK-END
