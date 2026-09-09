# CO-Bench :: Aircraft landing  (category: Scheduling)
#
# The problem is to schedule landing times for a set of planes across one or more runways such that each landing occurs within its prescribed time window and all pairwise separation requirements are satisfied; specifically, if plane i lands at or before plane j on the same runway, then the gap between their landing times must be at least the specified separation time provided in the input. In a multiple-runway setting, each plane must also be assigned to one runway, and if planes land on different runways, the separation requirement (which may differ) is applied accordingly. Each plane has an earliest, target, and latest landing time, with penalties incurred proportionally for landing before (earliness) or after (lateness) its target time. The objective is to minimize the total penalty cost while ensuring that no constraints are violated—if any constraint is breached, the solution receives no score.
#
# # Implement in Solve Function
#
# def solve(**kwargs):
#     """
#     Problem:
#         Given an instance of the Aircraft Landing Scheduling Problem, schedule the landing time for each plane and assign a runway so that:
#           - Each landing time is within its allowed time window.
#           - Each plane is assigned to one runway (from the available runways).
#           - For any two planes assigned to the same runway, if plane i lands at or before plane j, then the landing times must be separated by at least
#             the specified separation time (provided in the input data).
#           - The overall penalty is minimized. For each plane, if its landing time is earlier than its target time, a penalty
#             is incurred proportional to the earliness; if later than its target time, a penalty proportional to the lateness is incurred.
#           - If any constraint is violated, the solution receives no score.
#
#     Input kwargs:
#         num_planes  : (int) Number of planes.
#         num_runways : (int) Number of runways.
#         freeze_time : (float) Freeze time (unused in scheduling decisions).
#         planes      : (list of dict) Each dictionary contains:
#                         - "appearance"    : float, time the plane appears.
#                         - "earliest"      : float, earliest landing time.
#                         - "target"        : float, target landing time.
#                         - "latest"        : float, latest landing time.
#                         - "penalty_early" : float, penalty per unit time landing early.
#                         - "penalty_late"  : float, penalty per unit time landing late.
#         separation  : (list of lists) separation[i][j] is the required gap after plane i lands before plane j can land
#                       when they are assigned to the same runway.
#
#     Returns:
#         A dictionary named "schedule" mapping each plane id (1-indexed) to a dictionary with its scheduled landing time
#         and assigned runway, e.g., { plane_id: {"landing_time": float, "runway": int}, ... }.
#     """
#     # -----------------------
#     # For demonstration purposes, we simply schedule each plane at its target time
#     # and assign all planes to runway 1.
#     # (Note: This solution may be infeasible if targets do not satisfy separation constraints.)
#     schedule = {}
#     for i, plane in enumerate(kwargs["planes"], start=1):
#         schedule[i] = {"landing_time": plane["target"], "runway": 1}
#     return {"schedule": schedule}

# EVOLVE-BLOCK-START
def solve(**kwargs):
    planes = kwargs["planes"]
    sep = kwargs["separation"]
    R = kwargs["num_runways"]
    n = kwargs["num_planes"]
    # process planes by target time; greedily place on the runway giving least penalty
    order = sorted(range(n), key=lambda i: planes[i]["target"])
    runways = [[] for _ in range(R)]  # each: list of (landing_time, plane_idx)
    schedule = {}
    for idx in order:
        p = planes[idx]
        e, t, l = p["earliest"], p["target"], p["latest"]
        best = None  # (penalty, landing, runway)
        for r in range(R):
            needed = e
            for (lt, pi) in runways[r]:
                needed = max(needed, lt + sep[pi][idx])
            if needed > l + 1e-9:
                continue
            landing = max(needed, min(t, l))  # closest feasible time to target
            if landing < t:
                pen = (t - landing) * p["penalty_early"]
            elif landing > t:
                pen = (landing - t) * p["penalty_late"]
            else:
                pen = 0.0
            if best is None or pen < best[0]:
                best = (pen, landing, r)
        if best is None:  # infeasible everywhere -> place at latest on runway 0
            schedule[idx + 1] = {"landing_time": float(l), "runway": 1}
            runways[0].append((l, idx))
        else:
            _, landing, r = best
            schedule[idx + 1] = {"landing_time": float(landing), "runway": r + 1}
            runways[r].append((landing, idx))
    return {"schedule": schedule}
# EVOLVE-BLOCK-END
