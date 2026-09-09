# CO-Bench :: Bin packing - one-dimensional  (category: Packing)
#
# The **one-dimensional bin packing problem** seeks to minimize the number of bins required to pack a given set of items while ensuring that the sum of item sizes within each bin does not exceed the specified bin capacity. Given a test case with an identifier (`id`), a fixed `bin_capacity`, and a list of `num_items` with their respective sizes (`items`), the objective is to find a packing arrangement that uses the least number of bins. The solution is evaluated based on the total `num_bins` used, with invalid solutions (e.g., missing or duplicated items, or bins exceeding capacity) incurring a inf heavy penalty. The output must include the number of bins used and a valid assignment of item indices to bins.
#
# # Implement in Solve Function
#
# def solve(**kwargs):
#     """
#     Solve the one-dimensional bin packing problem for a single test case.
#
#     Input kwargs (for a single test case):
#       - id:           The problem identifier (string)
#       - bin_capacity: The capacity of each bin (int)
#       - num_items:    The number of items (int)
#       - items:        A list of item sizes (list of ints)
#       - **kwargs:     Other unused keyword arguments
#
#     Evaluation metric:
#       - The solution is scored by the total number of bins used.
#       - If the solution is invalid (e.g., items are missing or duplicated, or bin capacity is exceeded),
#         a penalty of 1,000,000 is added.
#
#     Returns:
#       A dictionary with:
#         - 'num_bins': An integer, the number of bins used.
#         - 'bins': A list of lists, where each inner list contains the 1-based indices of items assigned to that bin.
#
#     Note: This is a placeholder implementation.
#     """
#     # Placeholder: Replace with your bin packing solution.
#     return {
#         'num_bins': 0,
#         'bins': []
#     }

# EVOLVE-BLOCK-START
def solve(**kwargs):
    bin_capacity = kwargs["bin_capacity"]
    items = kwargs["items"]
    order = sorted(range(len(items)), key=lambda i: -items[i])
    bins, loads = [], []
    for i in order:
        placed = False
        for b in range(len(bins)):
            if loads[b] + items[i] <= bin_capacity + 1e-9:
                bins[b].append(i + 1)
                loads[b] += items[i]
                placed = True
                break
        if not placed:
            bins.append([i + 1])
            loads.append(items[i])
    return {"num_bins": len(bins), "bins": bins}
# EVOLVE-BLOCK-END
