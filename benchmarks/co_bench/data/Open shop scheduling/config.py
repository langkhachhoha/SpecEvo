DESCRIPTION = '''The Open Shop Scheduling Problem involves scheduling a set of jobs across a set of machines with the goal of minimizing the total completion time (makespan). Each job consists of several operations, where each operation must be processed on a specific machine for a given duration. Unlike other scheduling problems, the Open Shop variant has no predetermined order for processing the operations of a job—operations can be scheduled in any order, but a job can only be processed on one machine at a time, and a machine can only process one job at a time. This creates a complex combinatorial optimization challenge where the scheduler must determine both the sequence of operations for each job and the timing of each operation to minimize the overall completion time while ensuring no resource conflicts.'''


def solve(**kwargs):
    """
    Solves a single open shop scheduling test case.

    Input kwargs:
        - n_jobs (int): Number of jobs.
        - n_machines (int): Number of machines (and operations per job).
        - times (list of list of int): A 2D list of processing times for each operation.
          Dimensions: n_jobs x n_machines.
        - machines (list of list of int): A 2D list specifying the machine assignment for each operation.
          Dimensions: n_jobs x n_machines. Note machine is 1-indexed.

    Output:
        solution (dict): A dictionary containing:
            - start_times (list of list of int): A 2D list of start times for each operation.
              Dimensions: n_jobs x n_machines.

            Each start time must be a non-negative integer, and the schedule must respect the following constraint:
                (i) Non-parallel operation: Each job must be processed on only one machine at a time
                (ii) Machine exclusivity: For operations assigned to the same machine, their processing intervals must not overlap.

            The evaluation function will use the start_times to compute the makespan and verify the constraints.
    """

    # Extract the case parameters
    n_jobs = kwargs["n_jobs"]
    n_machines = kwargs["n_machines"]
    times = kwargs["times"]
    machines = kwargs["machines"]

    # TODO: Implement the scheduling algorithm here.
    # For now, we provide a dummy solution where all operations start at time 0.

    # Create a start_times list with dimensions n_jobs x n_machines, initializing all start times to 0.
    start_times = [[0 for _ in range(n_machines)] for _ in range(n_jobs)]

    # Build the solution dictionary.
    solution = {"start_times": start_times}

    return solution


def load_data(file_path):
    cases = []
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]  # remove blank lines

    i = 0
    while i < len(lines):
        # Look for a header line starting with "Nb of jobs"
        if lines[i].startswith("number of jobs"):
            # Next line contains six numbers: n_jobs, n_machines, time_seed, machine_seed, upper_bound, lower_bound
            i += 1
            header_tokens = lines[i].split()
            if len(header_tokens) < 6:
                raise ValueError("Header line does not contain 6 values.")
            n_jobs = int(header_tokens[0])
            n_machines = int(header_tokens[1])
            time_seed = int(header_tokens[2])
            machine_seed = int(header_tokens[3])
            upper_bound = int(header_tokens[4])
            lower_bound = int(header_tokens[5])

            # Find the "Times" section
            i += 1
            if not lines[i].lower().startswith("processing"):
                raise ValueError("Expected 'Times' section, got: " + lines[i])
            i += 1  # move to first line of times
            times = []
            for _ in range(n_jobs):
                # Each line should contain n_machines numbers
                time_line = list(map(int, lines[i].split()))
                if len(time_line) != n_machines:
                    raise ValueError(f"Expected {n_machines} numbers in times row, got {len(time_line)}")
                times.append(time_line)
                i += 1

            # Find the "Machines" section
            if i >= len(lines) or not lines[i].lower().startswith("machines"):
                raise ValueError("Expected 'Machines' section, got: " + (lines[i] if i < len(lines) else "EOF"))
            i += 1  # move to first line of machines
            machines = []
            for _ in range(n_jobs):
                machine_line = list(map(int, lines[i].split()))
                if len(machine_line) != n_machines:
                    raise ValueError(f"Expected {n_machines} numbers in machines row, got {len(machine_line)}")
                machines.append(machine_line)
                i += 1

            # Build the test case dictionary and add to the list of cases.
            case = {
                "n_jobs": n_jobs,
                "n_machines": n_machines,
                "time_seed": time_seed,
                "machine_seed": machine_seed,
                "upper_bound": upper_bound,
                "lower_bound": lower_bound,
                "times": times,
                "machines": machines
            }
            cases.append(case)
        else:
            # If the current line is not a header, skip it.
            i += 1

    return cases


def eval_func(n_jobs, n_machines, times, machines, start_times, **kwargs):
    """
    Evaluates the solution for a open shop scheduling problem.

    Input:
        n_jobs (int): Number of jobs.
        n_machines (int): Number of machines.
        times (list of list of int): Processing times for each operation.
            Dimensions: n_jobs x n_machines.
        machines (list of list of int): Machine assignments for each operation.
            Dimensions: n_jobs x n_machines.
        start_times (list of list of int): Proposed start times for each operation.
            Dimensions: n_jobs x n_machines.
        kwargs: Other parameters that may be provided, which are ignored here.

    Output:
        score (int): The makespan, defined as the maximum completion time across all jobs.

    Raises:
        ValueError: If any scheduling constraints are violated.
    """

    # Check that start_times dimensions match the problem dimensions.
    if len(start_times) != n_jobs:
        raise ValueError(f"Expected start_times to have {n_jobs} rows, got {len(start_times)}")
    for i, row in enumerate(start_times):
        if len(row) != n_machines:
            raise ValueError(f"Expected start_times row {i} to have {n_machines} entries, got {len(row)}")
        for t in row:
            if t < 0:
                raise ValueError("Start times must be non-negative.")

    job_operations = []
    job_completion_times = []
    for i in range(n_jobs):
        job_operations.append([])
        finish_time = 0
        for j in range(n_machines):
            st = start_times[i][j]
            pt = times[i][j]
            finish_time = max(finish_time, st + pt)
            job_operations[i].append((st, st + pt))
        job_completion_times.append(finish_time)

    for job_id in range(n_jobs):
        ops = sorted(job_operations[job_id], key=lambda x: x[0])  # Sort by start time
        for i in range(len(ops) - 1):
            if ops[i][1] > ops[i + 1][0]:  # End time of current > start time of next
                raise ValueError(f"Overlapping operations for job {job_id}: {ops[i]} and {ops[i + 1]}")

    # Constraint: Machine non-overlap.
    # Build a dictionary mapping machine id to a list of (start_time, finish_time, job, op_index)
    machine_schedules = {}
    for i in range(n_jobs):
        for j in range(n_machines):
            machine_id = machines[i][j]
            st = start_times[i][j]
            pt = times[i][j]
            finish_time = st + pt
            if machine_id not in machine_schedules:
                machine_schedules[machine_id] = []
            machine_schedules[machine_id].append((st, finish_time, i, j))

    # For each machine, sort operations by start time and check for overlaps.
    for machine_id, ops in machine_schedules.items():
        ops_sorted = sorted(ops, key=lambda x: x[0])
        for k in range(1, len(ops_sorted)):
            prev_st, prev_finish, prev_job, prev_op = ops_sorted[k - 1]
            curr_st, curr_finish, curr_job, curr_op = ops_sorted[k]
            if prev_finish > curr_st:
                raise ValueError(
                    f"Machine {machine_id}: Operation from job {prev_job}, op {prev_op} (finishing at {prev_finish}) overlaps with job {curr_job}, op {curr_op} (starting at {curr_st}).")

    # Compute the makespan as the maximum completion time among all jobs.
    makespan = max(job_completion_times)

    score = kwargs['lower_bound'] / makespan

    return score


def get_dev():
    dev = {'tai10_10.txt': [7, 8, 3, 9, 2], 'tai15_15.txt': [7, 0, 8, 4, 5], 'tai20_20.txt': [6, 0, 3, 8, 2],
           'tai4_4.txt': [0, 7, 5, 8, 6], 'tai5_5.txt': [3, 0, 9, 8, 1], 'tai7_7.txt': [3, 0, 8, 2, 1]}

    return dev
