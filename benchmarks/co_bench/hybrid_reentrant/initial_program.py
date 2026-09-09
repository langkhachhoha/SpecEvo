# CO-Bench :: Hybrid Reentrant Shop Scheduling  (category: Scheduling)
#
# The problem is a Hybrid Reentrant Shop Scheduling problem where each of n jobs must sequentially undergo three operations: an initialization phase on one of m identical primary machines, a setup phase on a single remote server, and a final main processing phase on the same primary machine used for initialization. Jobs are initialized in a fixed natural order using list scheduling, while the setup phase is processed on the remote server in an order specified by a permutation decision variable. Additionally, each job is assigned to a primary machine for main processing via a batch_assignment, and on each machine, jobs are processed in natural (initialization) order. The objective is to minimize the makespan, defined as the time when the last job completes its main processing, while ensuring that no machine (primary or server) processes more than one job simultaneously and that all operational precedence constraints are satisfied.
#
# # Implement in Solve Function
#
# def solve(**kwargs):
#     """
#     Input:
#       - n_jobs: Integer; the number of jobs.
#       - n_machines: Integer; the number of primary machines.
#       - init_time: Integer; the initialization time for every job on a primary machine.
#       - setup_times: List of integers; the setup times for each job on the remote server.
#       - processing_times: List of integers; the processing times for each job in the main processing stage.
#
#     Output:
#       A dictionary with the following keys:
#         - 'permutation': A list of integers of length n_jobs. This list represents the order in which the jobs are processed on the remote server.
#         - 'batch_assignment': A list of integers of length n_jobs. Each element indicates the primary machine to which the corresponding job (or batch) is assigned.
#     """
#
#     # TODO: Implement the solution logic.
#
#     # Placeholder return
#     n_jobs = kwargs['n_jobs']
#     return {
#         'permutation': list(range(1, n_jobs + 1)),
#         'batch_assignment': [1 if i % 2 == 0 else 2 for i in range(n_jobs)]
#     }

# EVOLVE-BLOCK-START
def solve(**kwargs):
    import heapq
    n = kwargs["n_jobs"]
    m = kwargs["n_machines"]
    init_time = kwargs["init_time"]
    setup = kwargs["setup_times"]
    proc = kwargs["processing_times"]

    def makespan(perm):
        op1 = [0] * (n + 1)
        assign = [0] * (n + 1)
        heap = [(0, mid) for mid in range(1, m + 1)]
        heapq.heapify(heap)
        for job in range(1, n + 1):
            av, mid = heapq.heappop(heap)
            op1[job] = av + init_time
            assign[job] = mid
            heapq.heappush(heap, (op1[job], mid))
        op2 = [0] * (n + 1)
        cur = 0
        for job in perm:
            st = max(op1[job], cur)
            op2[job] = st + setup[job - 1]
            cur = op2[job]
        by_m = {mid: [] for mid in range(1, m + 1)}
        for job in range(1, n + 1):
            by_m[assign[job]].append(job)
        op3 = [0] * (n + 1)
        for mid in range(1, m + 1):
            cmt = 0
            for job in sorted(by_m[mid]):
                st = max(cmt, op2[job])
                cmt = st + proc[job - 1]
                op3[job] = cmt
        return max(op3)

    ids = list(range(1, n + 1))
    cands = [
        ids[:],                                          # natural
        sorted(ids, key=lambda j: setup[j - 1]),          # shortest setup first
        sorted(ids, key=lambda j: -setup[j - 1]),         # longest setup first
        sorted(ids, key=lambda j: proc[j - 1]),           # shortest processing first
    ]
    best = min(cands, key=makespan)
    return {"permutation": best, "batch_assignment": [(j % m) + 1 for j in range(n)]}
# EVOLVE-BLOCK-END
