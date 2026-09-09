# CO-Bench :: Vehicle routing: period routing  (category: Routing)
#
# The Period Vehicle Routing Problem requires planning delivery routes over a multi‐day planning period.
#
# Each customer (other than the depot, whose id is 0) is provided with a list of candidate service schedules. A schedule is represented by a binary vector of length equal to the period (e.g., [1, 0, 1] for a 3‐day period), where a 1 in a given position indicates that the customer must be visited on that day. The decision maker must select exactly one candidate schedule for each customer.
#
# For every day in the planning period, if a customer’s chosen schedule indicates a delivery (i.e., a 1),
# then exactly one vehicle must visit that customer on that day. Otherwise, the customer should not be visited. The decision maker must also design, for each day, the tours for the vehicles. Each tour is a continuous route that starts at the depot (id 0) and, after visiting a subset of customers, returns to the depot. Each vehicle is only allowed to visit the depot once per day—namely, as its starting and ending point—and it is not allowed to return to the depot in the middle of a tour.
#
# Moreover, each vehicle route must obey a capacity constraint: the total demand of the customers visited on that tour must not exceed the vehicle capacity each day. Although multiple vehicles are available per day (as specified by the input), not all available vehicles have to be used, but the number of tours in a given day cannot exceed the provided number of vehicles. In addition, the tours on each day must cover exactly those customers who require service per the selected schedules, and no customer may be visited more than once in a given day.
#
# The objective is to choose a schedule for every customer and plan the daily tours so as to minimize the overall distance traveled
# by all vehicles during the entire planning period. Distances are measured using Euclidean distance.
#
# # Implement in Solve Function
#
# def solve(**kwargs):
#     """
#     Solves an instance of the Period Vehicle Routing Problem.
#
#     Input kwargs includes:
#       - depot: dict with keys:
#             "id": int, always 0.
#             "x": float, the x-coordinate.
#             "y": float, the y-coordinate.
#       - customers: list of dictionaries (with customer id ≠ 0) having keys:
#             "id": int, the customer id.
#             "x": float, the x-coordinate.
#             "y": float, the y-coordinate.
#             "demand": numeric, the customer demand.
#             "schedules": list of candidate schedules, each a list (of length period_length) with binary entries.
#       - vehicles_per_day: list of ints (length period_length) indicating the number of vehicles available each day.
#       - vehicle_capacity: numeric, the capacity of each vehicle.
#       - period_length: int, the number of days in the planning period.
#
#     The solution must decide:
#       1. Which service schedule (from the candidate schedules) is selected for each customer.
#       2. For each day (days are 1-indexed), the daily tours: a list of tours—one per available vehicle.
#          Each tour is a continuous route that starts at the depot (0), visits some customers (each exactly once),
#          and returns to the depot. The depot may only appear as the first and last vertex in each tour.
#          The number of tours for day d must be exactly equal to vehicles_per_day[d-1].
#
#     The returned solution is a dictionary containing:
#       - "selected_schedules": dict mapping each customer id (integer) to the chosen schedule (a list of binary integers).
#       - "tours": dict mapping day (an integer between 1 and period_length) to a list of tours.
#                  Each tour is a list of vertex ids (integers), starting and ending at the depot (id 0).
#     """
#     # ------------------------------
#
#     return {
#         "selected_schedules": ...,
#         "tours": ...
#     }

# EVOLVE-BLOCK-START
def solve(**kwargs):
    import math
    depot = kwargs["depot"]
    customers = kwargs["customers"]
    vpd = kwargs["vehicles_per_day"]
    cap = kwargs["vehicle_capacity"]
    P = kwargs["period_length"]
    by_id = {c["id"]: c for c in customers}

    def dist(a, b):
        return math.hypot(a["x"] - b["x"], a["y"] - b["y"])

    budget = [vpd[d] * cap for d in range(P)]
    load = [0.0] * P
    selected = {}
    # choose the candidate schedule that keeps daily load most balanced (heaviest first)
    for c in sorted(customers, key=lambda c: -c["demand"]):
        scheds = c["schedules"]
        if not scheds:
            selected[c["id"]] = [0] * P
            continue
        best, best_key = None, None
        for sc in scheds:
            days = [d for d in range(P) if sc[d] == 1]
            key = max(((load[d] + c["demand"]) / budget[d] if budget[d] > 0 else 9e9)
                      for d in days) if days else 0.0
            if best_key is None or key < best_key:
                best_key, best = key, sc
        selected[c["id"]] = list(best)
        for d in range(P):
            if best[d] == 1:
                load[d] += c["demand"]

    tours = {}
    for d in range(1, P + 1):
        today = sorted((cid for cid in selected if selected[cid][d - 1] == 1),
                       key=lambda cid: -by_id[cid]["demand"])
        V = vpd[d - 1]
        bins = [[0.0, []] for _ in range(max(V, 1))]  # never exceed V tours
        for cid in today:
            dem = by_id[cid]["demand"]
            fit = [b for b in bins if b[0] + dem <= cap + 1e-9]
            b = max(fit, key=lambda b: cap - b[0]) if fit else min(bins, key=lambda b: b[0])
            b[0] += dem
            b[1].append(cid)
        day_tours = []
        for _, cids in bins:
            if not cids:
                continue
            route, remaining, prev = [], set(cids), depot  # nearest-neighbour order
            while remaining:
                nxt = min(remaining, key=lambda cid: dist(prev, by_id[cid]))
                route.append(nxt)
                remaining.discard(nxt)
                prev = by_id[nxt]
            day_tours.append([0] + route + [0])
        tours[d] = day_tours
    return {"selected_schedules": selected, "tours": tours}
# EVOLVE-BLOCK-END
