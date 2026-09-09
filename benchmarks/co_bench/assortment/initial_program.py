# CO-Bench :: Assortment problem  (category: Cutting)
#
# This optimization problem involves arranging a set of rectangular pieces within available stock rectangles to minimize the overall waste area percentage. Each stock rectangle has a defined area, and each piece—which may be rotated by 90°—must be fully contained within a stock without overlapping with other pieces. Additionally, each piece type has specific total minimum and maximum placement limits. You have access to an unlimited number of stocks for each type, but you may use at most two stock types. The objective is to achieve the lowest possible waste area percentage, defined as the ratio of unused area to the total stock area. Solutions must ensure efficient resource utilization while satisfying all geometric and quantity constraints. Any violation of these constraints results in no score.
#
# # Implement in Solve Function
#
# def solve(**kwargs):
#     """
#     Solves the rectangular piece arrangement optimization problem to minimize the overall waste area percentage.
#
#     Given:
#       - m (int): Number of piece types.
#       - stocks (list of dict): Each dict represents a stock type with keys:
#             'length' (float), 'width' (float), 'fixed_cost' (float).
#       - pieces (list of dict): Each dict represents a piece type with keys:
#             'length' (float), 'width' (float), 'min' (int), 'max' (int), 'value' (float).
#
#     Objective:
#       Arrange rectangular pieces (which may be rotated by 90°) into stock rectangles such that the overall waste area percentage is minimized.
#       The waste area percentage is computed as:
#
#              Waste Percentage = (Total Stock Area - Total Used Area) / (Total Stock Area)
#
#     Constraints:
#       • Each piece must lie entirely within its assigned stock rectangle.
#       • Pieces must not overlap within the same stock rectangle.
#       • The number of pieces placed for each piece type must lie within its specified minimum and maximum bounds.
#       • You may use unlimited many instances of each selected stock type, but the solution can include at most 2 distinct stock types.
#
#     Output:
#       Returns a dictionary with two keys (exactly follow this format):
#         - "objective": The overall waste area percentage (float) as computed by the evaluation function.
#         - "placements": A dictionary mapping stock instance ids (1-indexed) to their placement details.
#           Each stock instance is represented by a dictionary with the following keys:
#               'stock_type': (the 1-indexed id of the stock type used for this instance),
#               'placements': a list of placements for pieces within that stock instance.
#                   Each placement is a dict with keys:
#                       'piece'       (piece type, 1-indexed, 1 <= piece type <= m),
#                       'x'           (x-coordinate of the bottom-left corner),
#                       'y'           (y-coordinate of the bottom-left corner),
#                       'orientation' (0 for normal, 1 for rotated 90°).
#
#     NOTE: The returned data should adhere to the output format required for evaluation.
#     """
#     # ----- INSERT YOUR SOLUTION ALGORITHM HERE -----
#     # For demonstration purposes, we provide a dummy solution that does not place any pieces.
#     # In a real solution, you would compute placements that respect all constraints.
#
#     # Dummy solution: Create a single stock instance of the first stock type, with no pieces placed.
#     solution = {
#         "objective": 0.0,  # With no placements, the evaluation function would compute a waste area percentage of 0.0.
#         "placements": {
#             1: {
#                 "stock_type": 1,
#                 "placements": []
#             }
#         }
#     }
#     return solution

# EVOLVE-BLOCK-START
def solve(**kwargs):
    m = kwargs["m"]
    stocks = kwargs["stocks"]
    pieces = kwargs["pieces"]
    need = [p["min"] for p in pieces]
    maxc = [p["max"] for p in pieces]

    def piece_fits(p, s):
        L, W = p["length"], p["width"]
        SL, SW = s["length"], s["width"]
        return (L <= SL + 1e-9 and W <= SW + 1e-9) or (W <= SL + 1e-9 and L <= SW + 1e-9)

    # pick smallest-area stock type that can hold every piece type (with rotation)
    cand = [(s["length"] * s["width"], si) for si, s in enumerate(stocks)
            if all(piece_fits(p, s) for p in pieces)]
    if not cand:
        cand = [(-(s["length"] * s["width"]), si) for si, s in enumerate(stocks)]
    cand.sort()
    sidx = cand[0][1]
    S = stocks[sidx]
    SL, SW = S["length"], S["width"]

    instances = []  # {'pl':[...], 'x','y','h'}

    def try_place(inst, p, ti):
        L, W = p["length"], p["width"]
        for orient, pw, ph in ((0, L, W), (1, W, L)):
            if pw > SL + 1e-9 or ph > SW + 1e-9:
                continue
            if inst["x"] + pw <= SL + 1e-9 and inst["y"] + ph <= SW + 1e-9:
                inst["pl"].append({"piece": ti + 1, "x": inst["x"], "y": inst["y"], "orientation": orient})
                inst["x"] += pw
                inst["h"] = max(inst["h"], ph)
                return True
            ny = inst["y"] + inst["h"]
            if pw <= SL + 1e-9 and ny + ph <= SW + 1e-9:
                inst["y"] = ny
                inst["x"] = pw
                inst["h"] = ph
                inst["pl"].append({"piece": ti + 1, "x": 0, "y": ny, "orientation": orient})
                return True
        return False

    counts = [0] * m

    def place(ti):
        p = pieces[ti]
        for inst in instances:
            if try_place(inst, p, ti):
                counts[ti] += 1
                return True
        inst = {"pl": [], "x": 0, "y": 0, "h": 0}
        instances.append(inst)
        if try_place(inst, p, ti):
            counts[ti] += 1
            return True
        instances.pop()  # piece cannot fit even an empty stock
        return False

    for ti in range(m):  # required minimum counts
        for _ in range(need[ti]):
            place(ti)

    changed = True  # top up open stocks to cut waste (never exceed max)
    while changed:
        changed = False
        for ti in range(m):
            if counts[ti] >= maxc[ti]:
                continue
            p = pieces[ti]
            for inst in instances:
                if try_place(inst, p, ti):
                    counts[ti] += 1
                    changed = True
                    break

    placements = {}
    for i, inst in enumerate(instances, 1):
        placements[i] = {"stock_type": sidx + 1, "placements": inst["pl"]}
    if not placements:
        placements = {1: {"stock_type": sidx + 1, "placements": []}}
    return {"objective": 0.0, "placements": placements}
# EVOLVE-BLOCK-END
