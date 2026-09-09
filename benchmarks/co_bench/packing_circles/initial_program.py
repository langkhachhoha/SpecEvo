# CO-Bench :: Packing unequal circles  (category: Packing)
#
# The problem involves packing a subset of unequal circles into a fixed circular container with radius R_0 and center at the origin, where each circle i has a given radius R_i (sorted in non-decreasing order) and is associated with a binary decision variable lpha_i indicating whether it is packed. The goal is to maximize the number of circles packed—that is, maximize \sum_{i=1}^{n}lpha_i—subject to two sets of nonlinear constraints: (1) each packed circle must lie entirely within the container, which is enforced by ensuring that the distance from its center to the container’s center plus its radius does not exceed R_0; and (2) any two packed circles must not overlap, meaning the distance between their centers must be at least the sum of their radii.
#
# # Implement in Solve Function
#
# def solve(**kwargs):
#     """
#     Solve the unequal circle packing problem for the maximize-number case.
#     Problem Description:
#       Given a circular container with center (cx, cy) and radius R, and n circles with specified radii (sorted in increasing order),
#       the task is to select and pack a prefix of the sorted list—i.e., if circle i is packed, then all circles with a smaller index must also be packed—in order to maximize the number of circles placed.
#       Each packed circle must be fully contained within the container, meaning that the distance from its center to (cx, cy) plus its radius must not exceed R, and no two packed circles may overlap, which requires that the distance between any two centers is at least the sum of their respective radii.
#
#     Input kwargs:
#       - n     : int, the number of circles.
#       - cx    : float, x-coordinate of the container's center.
#       - cy    : float, y-coordinate of the container's center.
#       - R     : float, the radius of the container.
#       - radii : list of float, the radius of each circle (assumed sorted in increasing order).
#
#     Returns:
#       A dictionary with one key:
#         - "coords": a list of n (x, y) tuples corresponding to the centers of the circles.
#           For circles that are not packed, the coordinates default to (-1, -1).
#     """
#     return {"coords": []}

# EVOLVE-BLOCK-START
def solve(**kwargs):
    n = kwargs["n"]
    cx = kwargs["cx"]
    cy = kwargs["cy"]
    R = kwargs["R"]
    radii = kwargs["radii"]
    coords = [(-1, -1)] * n
    placed = []  # (x, y, r)

    def find(r):
        lim = R - r
        if lim < -1e-12:
            return None
        if lim <= 1e-12:
            cand = [(cx, cy)]
        else:
            g = 70
            cand = []
            for a in range(g + 1):
                x = cx - lim + 2 * lim * a / g
                for b in range(g + 1):
                    y = cy - lim + 2 * lim * b / g
                    if (x - cx) ** 2 + (y - cy) ** 2 <= lim * lim + 1e-9:
                        cand.append((x, y))
            cand.sort(key=lambda p: (p[1], p[0]))
        for (x, y) in cand:
            if abs(x + 1) < 1e-6 and abs(y + 1) < 1e-6:
                continue
            ok = True
            for (px, py, pr) in placed:
                if (x - px) ** 2 + (y - py) ** 2 < (r + pr) ** 2 - 1e-9:
                    ok = False
                    break
            if ok:
                return (x, y)
        return None

    # maximise number: must pack a prefix of the (increasing-radius) list.
    for i in range(n):
        pos = find(radii[i])
        if pos is None:
            break
        coords[i] = (pos[0], pos[1])
        placed.append((pos[0], pos[1], radii[i]))
    return {"coords": coords}
# EVOLVE-BLOCK-END
