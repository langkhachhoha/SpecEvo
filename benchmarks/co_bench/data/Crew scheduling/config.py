DESCRIPTION = '''The Crew Scheduling Problem involves assigning each task—with defined start and finish times—to exactly one crew, aiming to minimize the total transition costs between consecutive tasks. Each crew’s schedule must satisfy three constraints: tasks within a crew must not overlap; valid transitions (with associated costs) must exist between every consecutive pair of tasks; and the crew’s total duty time (from the start of the first task to the finish of the last) cannot exceed a specified time limit. Additionally, no more than K crews can be used to cover all tasks. Solutions violating any of these constraints are considered infeasible and receive no score. The optimization objective is therefore to determine assignments of tasks to no more than K crews that minimize the sum of transition costs while strictly adhering to all constraints, yielding a feasible and cost-effective scheduling solution.'''


def solve(**kwargs):
    """
    Solves the crew scheduling problem.

    The problem consists of assigning each task (with a defined start and finish time) to exactly one crew,
    such that:
      - The tasks within each crew are executed in non-overlapping order.
      - For every consecutive pair of tasks in a crew’s schedule, a valid transition arc exists (with an associated cost).
      - The overall duty time (finish time of the last task minus start time of the first task) does not exceed the specified time limit.
      - Exactly K crews are used.

    Input kwargs (for one case):
      - N (int): Number of tasks.
      - K (int): Maximum number of crews to be used.
      - time_limit (float): Maximum allowed duty time.
      - tasks (dict): Dictionary mapping task ID (1 to N) to a tuple (start_time, finish_time).
      - arcs (dict): Dictionary mapping (from_task, to_task) pairs to transition cost.

    Evaluation metric:
      - If all constraints are met (no task overlap, valid transition arcs, duty time within the limit, and exactly K crews used), the score is the sum of transition costs across all crews.
      - If any constraint is violated, the solution is infeasible and receives no score.
      - A lower score indicates a more cost-effective solution.

    Returns:
      dict: A dictionary with one key "crews", whose value is a list of lists. Each inner list is a sequence of task IDs (integers)
            representing one crew’s schedule.
    """
    # --- placeholder implementation ---
    # For example, here we distribute tasks evenly across K crews.
    N = kwargs.get("N")
    K = kwargs.get("K")
    tasks_ids = list(range(1, N + 1))
    crews = [[] for _ in range(K)]
    for i, task in enumerate(tasks_ids):
        crews[i % K].append(task)
    # In practice, you would implement a heuristic or optimization method that groups tasks into exactly K crews
    # while satisfying the non-overlap, valid transitions, and duty time constraints.
    return {"crews": crews}


def load_data(input_path):
    """
    Loads input data from a provided text file path. This function supports multiple cases.

    The input file format for each case is as follows:
      - The first line contains two numbers: the number of tasks (N) and the maximum allowed duty time (time_limit).
      - The next N lines contain two numbers each: start time and finish time for each task (tasks are indexed from 1 to N).
      - The remaining lines describe transition arcs between tasks in the format: "i j cost".

    Cases are assumed to be separated by one or more blank lines.

    Returns:
      list: A list of dictionaries, each dictionary corresponds to one case with keys:
            "N", "time_limit", "tasks", "arcs".
    """
    cases = []
    try:
        with open(input_path, 'r') as f:
            # Read all lines
            raw_lines = [line.rstrip() for line in f]
    except Exception as e:
        raise ValueError("Failed to read input file: " + str(e))

    # Split lines into blocks separated by blank lines.
    blocks = []
    current_block = []
    for line in raw_lines:
        if line.strip() == "":
            if current_block:
                blocks.append(current_block)
                current_block = []
        else:
            current_block.append(line.strip())
    if current_block:
        blocks.append(current_block)

    # Parse each block as a separate case.
    for block in blocks:
        if not block:
            continue
        # Parse the first line: number of tasks and time limit.
        first_parts = block[0].split()
        if len(first_parts) < 2:
            raise ValueError("The first line must contain at least two values: number of tasks and time limit.")
        try:
            N = int(first_parts[0])
            time_limit = float(first_parts[1])
        except Exception as e:
            raise ValueError("Error parsing number of tasks or time limit: " + str(e))

        if len(block) < 1 + N:
            raise ValueError(f"Expected {N} task lines after the first line; found {len(block) - 1}.")

        tasks = {}
        # Parse tasks: next N lines.
        for i in range(1, 1 + N):
            parts = block[i].split()
            if len(parts) < 2:
                raise ValueError(f"Task line {i} does not contain two values.")
            try:
                start_time = float(parts[0])
                finish_time = float(parts[1])
            except Exception as e:
                raise ValueError(f"Invalid time values in task line {i}: " + str(e))
            tasks[i] = (start_time, finish_time)

        # Parse remaining lines: transition arcs.
        arcs = {}
        for line in block[1 + N:]:
            parts = line.split()
            if len(parts) < 3:
                continue  # Ignore lines that don't have the complete triple.
            try:
                from_task = int(parts[0])
                to_task = int(parts[1])
                cost = float(parts[2])
            except Exception:
                continue  # Skip lines with invalid formatting.
            arcs[(from_task, to_task)] = cost

        case_data = {"N": N, "time_limit": time_limit, "tasks": tasks, "arcs": arcs}

        crew_number = {
            'csp50.txt': [27, 32],
            'csp100.txt': [44, 49],
            'csp150.txt': [69, 74],
            'csp200.txt': [93, 98],
            'csp250.txt': [108, 113],
            'csp300.txt': [130, 134],
            'csp350.txt': [144, 149],
            'csp400.txt': [159, 164],
            'csp450.txt': [182, 187],
            'csp500.txt': [204, 209],
        }
        for case in crew_number.keys():
            if case in input_path:
                for k in range(crew_number[case][0], crew_number[case][1]):
                    cases.append(case_data | {'K': k})
                break

        # cases.append(case_data)

    return cases


def eval_func(**kwargs):
    """
    Evaluates the quality (i.e. total cost and feasibility) of a crew scheduling solution.
    Raises an error immediately if any feasibility constraint is violated.

    Input kwargs must include:
      - N (int): Number of tasks.
      - K (int): The exact number of crews required.
      - time_limit (float): Maximum allowed duty time.
      - tasks (dict): Mapping from task ID to (start_time, finish_time).
      - arcs (dict): Mapping from (from_task, to_task) to transition cost.
      - crews (list): List of lists, where each inner list is the sequence of task IDs for one crew.

    Returns:
      float: The total transition cost if the solution is feasible.
    """
    N = kwargs.get("N")
    K = kwargs.get("K")
    time_limit = kwargs.get("time_limit")
    tasks = kwargs.get("tasks")
    arcs = kwargs.get("arcs")
    crews = kwargs.get("crews")

    if crews is None:
        raise ValueError("Solution does not contain a 'crews' key.")

    # Check that exactly K crews are used.
    if K is None:
        raise ValueError("Parameter K (number of crews) is missing.")
    if len(crews) > K:
        raise ValueError(f"Invalid solution: number of crews in solution is larger than K={K}.")

    # Validate that every task appears exactly once.
    all_tasks_in_output = [task for crew in crews for task in crew]
    if len(all_tasks_in_output) != N:
        raise ValueError("Invalid solution: number of tasks in crews does not equal N.")
    if set(all_tasks_in_output) != set(range(1, N + 1)):
        raise ValueError("Invalid solution: tasks in crews do not match expected tasks set.")

    total_cost = 0.0

    # Evaluate each crew schedule.
    for crew in crews:
        if not crew:
            raise ValueError("Invalid solution: one crew has an empty schedule.")

        # Check the duty time.
        first_task = crew[0]
        last_task = crew[-1]
        duty_time = tasks[last_task][1] - tasks[first_task][0]
        if duty_time > time_limit:
            raise ValueError("Invalid solution: duty time for a crew exceeds the time limit.")

        # Check each consecutive pair of tasks.
        for idx in range(len(crew) - 1):
            current_task = crew[idx]
            next_task = crew[idx + 1]

            # Check that tasks do not overlap.
            if tasks[current_task][1] > tasks[next_task][0]:
                raise ValueError(f"Invalid solution: tasks {current_task} and {next_task} overlap.")

            # Check that a valid transition arc exists.
            if (current_task, next_task) not in arcs:
                raise ValueError(
                    f"Invalid solution: missing transition arc between tasks {current_task} and {next_task}.")

            # Add the transition cost.
            total_cost += arcs[(current_task, next_task)]

    return total_cost


def norm_score(results):
    optimal_scores = {
        'csp50.txt': [3139, 2706, 2399, 2092, 1872],
        'csp100.txt': [4812, 4514, 4310, 4107, 3905],
        'csp150.txt': [6275, 5999, 5754, 5551, 5347],
        'csp200.txt': [6914, 6747, 6583, 6430, 6288],
        'csp250.txt': [8406, 8212, 8023, 7863, 7707],
        'csp300.txt': [9580, 9378, 9200, 9026],
        'csp350.txt': [10991, 10833, 10677, 10525, 10378],
        'csp400.txt': [12341, 12163, 12006, 11848, 11696],
        'csp450.txt': [12785, 12639, 12497, 12357, 12232],
        'csp500.txt': [13302, 13169, 13032, 12899, 12772],
    }

    normed = {}
    for case, (scores, error_message) in results.items():
        if case not in optimal_scores:
            continue  # Skip if there's no optimal score defined.
        optimal_list = optimal_scores[case]
        normed_scores = []
        # Compute normalized score for each index.
        for idx, score in enumerate(scores):
            if isinstance(score, (int, float)):
                normed_scores.append(optimal_list[idx] / score)
            else:
                normed_scores.append(score)
        normed[case] = (normed_scores, error_message)

    return normed


def get_dev():
    dev = {'csp100.txt': [2, 1], 'csp150.txt': [1, 4], 'csp200.txt': [4, 2], 'csp250.txt': [2, 1], 'csp300.txt': [2, 0],
           'csp350.txt': [4, 3], 'csp400.txt': [2, 0], 'csp450.txt': [2, 1], 'csp50.txt': [1, 0], 'csp500.txt': [4, 1]}

    return dev
