# CO-Bench :: Packing unequal rectangles and squares area  (category: Packing)
#
# We consider the problem of selecting and placing a subset of  n  unequal rectangles (or squares) into a fixed‐size circular container of radius  R  so as to maximize the total area of the packed items. Each item  i  has given dimensions  L_i  and  W_i  (with  L_i = W_i  for squares) and an associated area  L_iW_i . The decision variables include a binary indicator lpha_i for whether item  i  is packed and continuous variables (x_i, y_i) for the placement of its center, along with a rotation angle  	heta_i  when 90^\circ rotations are allowed. The formulation enforces that for every packed item, all four of its rotated corners must lie within the circle, and that no two packed items overlap; if an item is not packed, it is fixed at a dummy position.
#
# # Implement in Solve Function
#
# def solve(**kwargs):
#     """
#     Solves the problem of packing a subset of unequal rectangles and squares into a fixed‐size circular container
#     with the objective of maximizing the total area of the items placed inside the container.
#
#     Input kwargs:
#       - n         : int, the number of items (rectangles or squares)
#       - cx, cy    : floats, the coordinates of the container center
#       - R         : float, the radius of the container
#       - items     : list of tuples, where each tuple (L, W) gives the dimensions of an item
#                     (for a square, L == W)
#       - shape     : string, either "rectangle" or "square"
#       - rotation  : bool, whether 90° rotation is allowed (True or False)
#
#     Objective:
#       - Select and place a subset of the given items so that each packed item lies completely inside the circular container,
#         no two packed items overlap, and the sum of the areas of the packed items is maximized.
#       - An item that is not packed contributes zero area.
#
#     Returns:
#       A dictionary with the key 'placements' containing a list of exactly n tuples.
#       Each tuple is (x-coordinate, y-coordinate, theta) where:
#           - (x-coordinate, y-coordinate) is the center position of the item (if packed),
#           - theta is the rotation angle in degrees (counter-clockwise from the horizontal). 90 or 0.
#           - For an unpacked item, x and y should be set to -1 and theta to 0 (or another default value).
#
#     Note: This is a placeholder. The actual solution logic is not implemented here.
#     """
#     ## placeholder.
#     return {'placements': []}

# EVOLVE-BLOCK-START
def solve(**kwargs):
    import math
    n = kwargs["n"]
    cx = kwargs["cx"]
    cy = kwargs["cy"]
    R = kwargs["R"]
    items = kwargs["items"]
    rotation = kwargs["rotation"]
    placements = [(-1, -1, 0)] * n
    placed = []  # (xmin, xmax, ymin, ymax)

    def find(hL, hW):
        g = 60
        cand = []
        for a in range(g + 1):
            x = cx - R + 2 * R * a / g
            for b in range(g + 1):
                y = cy - R + 2 * R * b / g
                if abs(x + 1) < 1e-6 and abs(y + 1) < 1e-6:
                    continue
                # all four corners inside the circle
                if math.hypot(abs(x - cx) + hL, abs(y - cy) + hW) > R - 1e-6:
                    continue
                cand.append((x, y))
        cand.sort(key=lambda p: (p[1], p[0]))
        for (x, y) in cand:
            xmin, xmax = x - hL, x + hL
            ymin, ymax = y - hW, y + hW
            ok = True
            for (oxmin, oxmax, oymin, oymax) in placed:
                if not (xmax <= oxmin + 1e-6 or xmin >= oxmax - 1e-6 or
                        ymax <= oymin + 1e-6 or ymin >= oymax - 1e-6):
                    ok = False
                    break
            if ok:
                return (x, y, xmin, xmax, ymin, ymax)
        return None

    def try_item(i):
        L, W = items[i]
        opts = [(0, L / 2.0, W / 2.0)]
        if rotation:
            opts.append((90, W / 2.0, L / 2.0))
        for theta, hL, hW in opts:
            res = find(hL, hW)
            if res is not None:
                x, y, xmin, xmax, ymin, ymax = res
                placements[i] = (x, y, theta)
                placed.append((xmin, xmax, ymin, ymax))
                return True
        return False

    # maximise packed area: largest-area items first.
    for i in sorted(range(n), key=lambda k: -(items[k][0] * items[k][1])):
        try_item(i)
    return {"placements": placements}
# EVOLVE-BLOCK-END
