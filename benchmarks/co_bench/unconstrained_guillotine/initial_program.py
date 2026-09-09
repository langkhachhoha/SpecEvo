# CO-Bench :: Unconstrained guillotine cutting  (category: Cutting)
#
# The unconstrained guillotine cutting problem involves selecting and placing a subset of available pieces within a fixed stock rectangle to maximize the total value of the placed pieces. Each piece, defined by its length, width, and value, may be optionally rotated 90° if allowed and used at most once. The challenge is to determine both the selection and the positioning of these pieces such that they do not overlap and lie entirely within the stock’s boundaries. This optimization problem formalizes the decision variables as the x and y coordinates for the bottom-left placement of each piece and, if rotation is allowed, a binary variable indicating its orientation, while the objective function is to maximize the sum of the values of the pieces successfully placed within the stock.
#
# # Implement in Solve Function
#
# def solve(**kwargs):
#     """
#     Solves the unconstrained guillotine cutting problem.
#
#     Given a stock rectangle (with dimensions 'stock_width' and 'stock_height') and a set of pieces
#     (provided as a dictionary 'pieces' mapping each piece_id to its specification {'l', 'w', 'value'}),
#     the goal is to select and place some pieces (each used at most once) within the stock rectangle.
#     If the keyword argument 'allow_rotation' is True, each piece may be placed in its original orientation or rotated 90° (swapping its dimensions);
#     otherwise, pieces must be placed in their original orientation. In all cases, placements must not overlap and must lie entirely within the stock.
#
#     Input kwargs:
#         - m (int): Number of available pieces.
#         - stock_width (int): The width of the stock rectangle.
#         - stock_height (int): The height of the stock rectangle.
#         - pieces (dict): A dictionary mapping piece_id (1-indexed) to a dict with keys:
#               'l' (length), 'w' (width), and 'value' (value of the piece).
#         - allow_rotation (bool): Indicates whether a piece is allowed to be rotated 90°.
#
#     Evaluation metric:
#         The performance is measured as the total value of the placed pieces (sum of individual values).
#
#     Returns:
#         A dictionary with a key "placements" whose value is a list.
#         Each element in the list is a dictionary representing a placement with keys:
#             - piece_id (int): Identifier of the placed piece.
#             - x (int): x-coordinate of the bottom-left corner in the stock rectangle.
#             - y (int): y-coordinate of the bottom-left corner in the stock rectangle.
#             - orientation (int): 0 for original orientation; 1 if rotated 90° (only applicable if allow_rotation is True, otherwise default to 0).
#
#     NOTE: This is a placeholder function. Replace the body with an actual algorithm if desired.
#     """
#     ## placeholder. You do not need to write anything here.
#     return {"placements": []}

# EVOLVE-BLOCK-START
def solve(**kwargs):
    SW = kwargs["stock_width"]
    SH = kwargs["stock_height"]
    pieces = kwargs["pieces"]
    rot = kwargs["allow_rotation"]
    placements = []
    cur_x = 0
    cur_y = 0
    shelf_h = 0
    # shelf (next-fit) packing, most valuable pieces first; each piece used once.
    for pid in sorted(pieces.keys(), key=lambda k: -pieces[k]["value"]):
        p = pieces[pid]
        opts = [(0, p["l"], p["w"])]
        if rot:
            opts.append((1, p["w"], p["l"]))
        done = False
        for orient, pw, ph in opts:  # try current shelf
            if cur_x + pw <= SW and cur_y + ph <= SH:
                placements.append({"piece_id": pid, "x": cur_x, "y": cur_y, "orientation": orient})
                cur_x += pw
                shelf_h = max(shelf_h, ph)
                done = True
                break
        if done:
            continue
        ny = cur_y + shelf_h  # open a new shelf
        for orient, pw, ph in opts:
            if pw <= SW and ny + ph <= SH:
                cur_y = ny
                cur_x = 0
                shelf_h = ph
                placements.append({"piece_id": pid, "x": 0, "y": cur_y, "orientation": orient})
                cur_x = pw
                break
    return {"placements": placements}
# EVOLVE-BLOCK-END
