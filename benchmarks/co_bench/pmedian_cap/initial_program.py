# CO-Bench :: p-median - capacitated  (category: Facility location)
#
# The Capacitated P-Median Problem is a facility location optimization problem where the objective is to select exactly  p  customers as medians (facility locations) and assign each customer to one of these medians to minimize the total cost, defined as the sum of the Euclidean distances (rounded down to the nearest integer) between customers and their assigned medians. Each median has a capacity constraint  Q , meaning the total demand of the customers assigned to it cannot exceed  Q . A feasible solution must respect this capacity constraint for all medians; otherwise, it receives a score of zero. The solution is evaluated by the ratio  	ext{score} =
# rac{	ext{best\_known}}{	ext{computed\_total\_cost}} , where computed_total_cost is the total assignment cost if all constraints are satisfied; otherwise, the score is zero. The output consists of the total cost (if feasible), the selected medians, and the customer assignments.
#
# # Implement in Solve Function
#
# def solve(**kwargs):
#     """
#     Solve the Capacitated P-Median Problem.
#
#     This function receives the data for one problem instance via keyword arguments:
#       - best_known (float): Best known solution value for reference.
#       - n (int): Number of customers.
#       - p (int): Number of medians to choose.
#       - Q (float): Capacity limit for each median.
#       - customers (list of tuples): Each tuple is (customer_id, x, y, demand).
#
#     The goal is to select p medians (from the customers) and assign every customer to one
#     of these medians so that the total cost is minimized. The cost for a customer is the
#     Euclidean distance (rounded down to the nearest integer) to its assigned median, and the
#     total demand assigned to each median must not exceed Q.
#
#     Evaluation Metric:
#       The solution is evaluated by computing the ratio:
#           score = best_known / computed_total_cost,
#       where computed_total_cost is the sum over all customers of the (floored) Euclidean distance
#       to its assigned median.
#
#     Note: This is a placeholder function. Replace the placeholder with an actual algorithm.
#
#     Returns:
#       A dictionary with the following keys:
#         - 'objective': (numeric) the total cost (objective value) computed by the algorithm.
#         - 'medians': (list of int) exactly p customer IDs chosen as medians.
#         - 'assignments': (list of int) a list of n integers, where the i-th integer is the customer
#                          ID (from the chosen medians) assigned to customer i.
#     """
#     # Placeholder: Replace this with your actual implementation.
#     # For now, we return an empty solution structure.
#     return {
#         "objective": 0,  # total cost (to be computed)
#         "medians": [],  # list of p medians (customer IDs)
#         "assignments": []  # list of n assignments (each is one of the medians)
#     }

# EVOLVE-BLOCK-START
def solve(**kwargs):
    import math
    n = kwargs["n"]
    p = kwargs["p"]
    Q = kwargs["Q"]
    customers = kwargs["customers"]
    ids = [c[0] for c in customers]
    xs = [c[1] for c in customers]
    ys = [c[2] for c in customers]
    dem = [c[3] for c in customers]

    def dist(i, j):
        return math.hypot(xs[i] - xs[j], ys[i] - ys[j])

    # farthest-first: spread p medians, seeded at the highest-demand customer
    start = max(range(n), key=lambda i: dem[i])
    med = [start]
    mind = [dist(i, start) for i in range(n)]
    while len(med) < min(p, n):
        nxt = max((i for i in range(n) if i not in set(med)), key=lambda i: mind[i])
        med.append(nxt)
        for i in range(n):
            d = dist(i, nxt)
            if d < mind[i]:
                mind[i] = d

    cap = {mi: Q for mi in med}
    assign = [None] * n
    # assign heaviest customers first to the nearest median with spare capacity
    for i in sorted(range(n), key=lambda i: -dem[i]):
        best, bestd = None, None
        for mi in med:
            if cap[mi] + 1e-9 >= dem[i]:
                d = dist(i, mi)
                if bestd is None or d < bestd:
                    bestd, best = d, mi
        if best is None:  # no spare capacity anywhere -> nearest (may be infeasible)
            best = min(med, key=lambda mi: dist(i, mi))
        assign[i] = best
        cap[best] -= dem[i]

    obj = sum(math.floor(dist(i, assign[i])) for i in range(n))
    return {
        "objective": obj,
        "medians": [ids[mi] for mi in med],
        "assignments": [ids[assign[i]] for i in range(n)],
    }
# EVOLVE-BLOCK-END
