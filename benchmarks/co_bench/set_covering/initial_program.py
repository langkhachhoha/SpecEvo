# CO-Bench :: Set covering  (category: Graph & set)
#
# Set Covering Problem. The goal is to select a subset of columns, each with an associated cost, such that every row is covered by at least one chosen column. For each row, the available covering columns are provided (as 1-indexed numbers). The objective is to minimize the total cost of the selected columns, and if even one row is left uncovered, then no score is awarded.
#
# # Implement in Solve Function
#
# def solve(**kwargs):
#     """
#     Solves the set covering optimization problem.
#
#     Problem Description:
#       Given m rows (constraints) and n columns (covering sets) with associated costs,
#       choose a subset of columns such that every row is covered (i.e. for every row,
#       at least one chosen column appears in that row's coverage list) while minimizing
#       the total cost (the sum of the costs of the chosen columns).
#
#     Input kwargs:
#       - m: (int) number of rows.
#       - n: (int) number of columns.
#       - costs: (list of int) where costs[j] is the cost for column j+1.
#       - row_cover: (list of list of int) where row_cover[i] contains the 1-indexed column
#                    numbers that cover row i+1.
#
#     Evaluation Metric:
#       The score is computed as the sum of the costs for the chosen columns.
#       However, if any row is left uncovered by the chosen columns, the solution is invalid and receives no score.
#       Otherwise, the score is simply the total cost of the selected columns.
#
#     Returns:
#       A dictionary with one key:
#          - "selected_columns": a list of 1-indexed column numbers representing the chosen covering set.
#     """
#     ## placeholder. You do not need to write anything here.
#     return {"selected_columns": []}

# EVOLVE-BLOCK-START
def solve(**kwargs):
    m = kwargs["m"]
    n = kwargs["n"]
    costs = kwargs["costs"]
    row_cover = kwargs["row_cover"]
    col_rows = [set() for _ in range(n + 1)]
    for i in range(m):
        for c in row_cover[i]:
            col_rows[c].add(i)
    uncovered = set(range(m))
    selected = []
    # greedy: repeatedly take the column with the best cost / newly-covered ratio
    while uncovered:
        best, best_key = None, None
        for c in range(1, n + 1):
            new = len(col_rows[c] & uncovered)
            if new <= 0:
                continue
            key = costs[c - 1] / new
            if best_key is None or key < best_key:
                best_key, best = key, c
        if best is None:
            break
        selected.append(best)
        uncovered -= col_rows[best]
    return {"selected_columns": selected}
# EVOLVE-BLOCK-END
