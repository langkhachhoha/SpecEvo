# CO-Bench :: Set partitioning  (category: Graph & set)
#
# This problem involves solving a set partitioning instance where the goal is to choose a subset of columns such that each row is covered exactly once while minimizing the total cost. Each column is associated with a cost and covers a specific set of rows. The optimization problem is defined by selecting columns from a given set so that every row is covered precisely once, and the sum of the selected columns’ costs is minimized. If the solution fails to cover every row exactly once, then no score is awarded.
#
# # Implement in Solve Function
#
# def solve(**kwargs):
#     """
#     Solve a set partitioning problem instance.
#
#     The problem: Given a set of rows and a set of columns (each with an associated cost and a set
#     of rows it covers), select a subset of columns so that each row is covered exactly once and the
#     total cost is minimized.
#
#     Input kwargs:
#   - num_rows (int): Total number of rows. (int)
#   - num_columns (int): Total number of columns. (int)
#   - columns_info (dict): Dictionary mapping 1-indexed column indices (int) to a tuple:
#                          (cost (int), set of row indices (set[int]) covered by that column).
#
#
#     Evaluation metric:
#       The objective score equals the sum of the costs of the selected columns if the solution is feasible,
#       i.e., if every row is covered exactly once. Otherwise, the solution is invalid and receives no score.
#
#     Returns:
#       A dictionary with key "selected_columns" containing a list of chosen column indices in strictly increasing order.
#       (This is a placeholder implementation.)
#     """
#     # Placeholder implementation.
#     # You must replace the following line with your actual solution logic.
#     return {"selected_columns": []}

# EVOLVE-BLOCK-START
def solve(**kwargs):
    R = kwargs["num_rows"]
    cols = kwargs["columns_info"]
    covered = set()
    remaining = set(range(1, R + 1))
    selected = []
    # greedy exact cover: pick non-overlapping columns by best cost / new-rows ratio
    while remaining:
        best, best_key, best_rows = None, None, None
        for c, (cost, rows) in cols.items():
            if rows & covered:
                continue                    # would cover a row twice
            new = len(rows & remaining)
            if new <= 0:
                continue
            key = cost / new
            if best_key is None or key < best_key:
                best_key, best, best_rows = key, c, rows
        if best is None:
            break                           # stuck -> infeasible (scores 0)
        selected.append(best)
        covered |= best_rows
        remaining -= best_rows
    return {"selected_columns": sorted(selected)}
# EVOLVE-BLOCK-END
