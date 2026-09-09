# CO-Bench :: Common due date scheduling  (category: Scheduling)
#
# The **Restricted Single-Machine Common Due Date Scheduling Problem** involves scheduling a set of jobs on a single machine to minimize a total penalty. Each job is defined by a tuple \((p, a, b)\), where \( p \) represents the processing time, \( a \) is the earliness penalty coefficient, and \( b \) is the tardiness penalty coefficient. A common due date \( d \) is determined as \( d = \lfloor \sum p 	imes h
# floor \), where \( h \) is a predefined fraction (defaulting to 0.6). The goal is to determine an optimal job sequence that minimizes the penalty, calculated as follows: for each job, if its completion time \( C \) is earlier than \( d \), an earliness penalty of \( a 	imes (d - C) \) is incurred; if \( C \) exceeds \( d \), a tardiness penalty of \( b 	imes (C - d) \) is applied; otherwise, no penalty is incurred. The problem requires finding a permutation of job indices (1-based) that minimizes the total penalty. The evaluation metric sums these penalties for a given schedule.
#
#
# # Implement in Solve Function
#
# def solve(**kwargs):
#     """
#     Solves the restricted single‐machine common due date scheduling problem.
#
#     The problem:
#        Given a list of jobs where each job is represented as a tuple (p, a, b):
#          • p: processing time
#          • a: earliness penalty coefficient
#          • b: tardiness penalty coefficient
#        and an optional parameter h (default 0.6), the common due date is computed as:
#              d = floor(sum(p) * h)
#        A schedule (i.e., a permutation of job indices in 1‐based numbering) is produced.
#        When processing the jobs in that order, the penalty is computed by:
#          • Adding a × (d − C) if a job’s completion time C is less than d,
#          • Adding b × (C − d) if C is greater than d,
#          • No penalty if C equals d.
#        The objective is to minimize the total penalty.
#
#     Input kwargs:
#          - 'jobs' (List[Tuple[int, int, int]]): a list of tuples where each tuple represents a job with:
#               • p (int): processing time,
#               • a (int): earliness penalty coefficient,
#               • b (int): tardiness penalty coefficient.
#          - Optional: 'h' (float): the factor used to compute the common due date (default is 0.6).
#
#     Evaluation Metric:
#          The computed schedule is evaluated by accumulating processing times and applying
#          the appropriate earliness/tardiness penalties with respect to the common due date.
#
#     Returns:
#          A dictionary with key 'schedule' whose value is a list of integers representing
#          a valid permutation of job indices (1-based).
#     """
#     # Placeholder implementation: simply return the jobs in their original order.
#     jobs = kwargs.get('jobs', [])
#     n = len(jobs)
#     return {'schedule': list(range(1, n + 1))}

# EVOLVE-BLOCK-START
def solve(**kwargs):
    jobs = kwargs["jobs"]
    h = kwargs.get("h", 0.6)
    n = len(jobs)
    total_p = sum(p for p, a, b in jobs)
    d = int(total_p * h)

    def penalty(perm):
        c = 0
        tot = 0
        for idx in perm:
            p, a, b = jobs[idx - 1]
            c += p
            if c < d:
                tot += a * (d - c)
            elif c > d:
                tot += b * (c - d)
        return tot

    ids = list(range(1, n + 1))
    cands = []
    cands.append(ids[:])                                            # identity
    cands.append(sorted(ids, key=lambda i: jobs[i - 1][0]))         # SPT
    cands.append(sorted(ids, key=lambda i: -jobs[i - 1][0]))        # LPT
    # V-shape around the due date: early jobs sorted by p/a desc, tardy by p/b asc
    early = sorted(ids, key=lambda i: -(jobs[i - 1][0] / max(jobs[i - 1][1], 1e-9)))
    tardy = sorted(ids, key=lambda i: (jobs[i - 1][0] / max(jobs[i - 1][2], 1e-9)))
    used, seq, load = set(), [], 0
    for i in early:  # fill up to the due date, then the rest
        if load + jobs[i - 1][0] <= d:
            seq.append(i)
            used.add(i)
            load += jobs[i - 1][0]
    seq += [i for i in tardy if i not in used]
    cands.append(seq)
    best = min(cands, key=penalty)
    return {"schedule": best}
# EVOLVE-BLOCK-END
