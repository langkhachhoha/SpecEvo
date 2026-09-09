# CO-Bench :: Generalised assignment problem  (category: Assignment)
#
# Generalized Assignment Problem (GAP)
#
# The Generalized Assignment Problem (GAP) involves assigning \( n \) jobs to \( m \) agents such that each job is assigned to exactly one agent, and the resource consumption for each agent does not exceed its capacity. The objective is to optimize the total cost based on the problem type. When formulated as a maximization problem, the goal is to maximize the total cost; when formulated as a minimization problem, the goal is to minimize the total cost. Given a cost matrix (representing the cost of assigning jobs to agents), a consumption matrix (indicating the resource usage per assignment), and capacities (the resource limits for each agent), the task is to find a valid assignment that meets the capacity constraints while optimizing the total cost as specified by the problem indicator.
#
#
# # Implement in Solve Function
#
# def solve(**kwargs):
#     """
#     Solve the Generalised Assignment Problem (GAP) for a single case.
#
#     Input arguments (passed as keyword arguments):
#       - m: (int) Number of agents.
#       - n: (int) Number of jobs.
#       - cost_matrix: (list of list of float) A matrix of size m×n where cost_matrix[i][j]
#                      represents the cost of assigning job j to agent i.
#       - consumption_matrix: (list of list of float) A matrix of size m×n where consumption_matrix[i][j]
#                      represents the resource consumed when job j is assigned to agent i.
#       - capacities: (list of float) A list of length m containing the resource capacity for each agent.
#       - problem_type: (str, optional) Indicates whether the problem is a 'max' or 'min' problem.
#                      Defaults to 'max'.
#
#     Returns:
#       A dictionary with the key 'assignments' whose value is a list of n integers.
#       Each integer is an agent number (using 1-indexing) that is assigned to the corresponding job.
#     """
#     # For illustration purposes, we provide a trivial solution that assigns every job to agent 1.
#     assignments = [1] * kwargs['n']
#     return {'assignments': assignments}

# EVOLVE-BLOCK-START
def solve(**kwargs):
    m = kwargs["m"]
    n = kwargs["n"]
    cost = kwargs["cost_matrix"]
    cons = kwargs["consumption_matrix"]
    cap = list(kwargs["capacities"])
    ptype = kwargs.get("problem_type", "max")
    # Start: each job on the agent it consumes least on, then repair overflows.
    assign = [min(range(m), key=lambda i: cons[i][j]) for j in range(n)]
    load = [0.0] * m
    for j in range(n):
        load[assign[j]] += cons[assign[j]][j]
    for _ in range(200 * n):
        over = [i for i in range(m) if load[i] > cap[i] + 1e-9]
        if not over:
            break
        i = max(over, key=lambda i: load[i] - cap[i])
        best = None
        for j in (j for j in range(n) if assign[j] == i):
            for t in range(m):
                if t == i or load[t] + cons[t][j] > cap[t] + 1e-9:
                    continue
                pen = (cost[i][j] - cost[t][j]) if ptype == "max" else (cost[t][j] - cost[i][j])
                if best is None or pen < best[0]:
                    best = (pen, j, t)
        if best is None:
            break
        _, j, t = best
        load[i] -= cons[i][j]
        load[t] += cons[t][j]
        assign[j] = t
    return {"assignments": [a + 1 for a in assign]}
# EVOLVE-BLOCK-END
