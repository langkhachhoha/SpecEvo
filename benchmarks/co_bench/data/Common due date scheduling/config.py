DESCRIPTION = '''The **Restricted Single-Machine Common Due Date Scheduling Problem** involves scheduling a set of jobs on a single machine to minimize a total penalty. Each job is defined by a tuple \((p, a, b)\), where \( p \) represents the processing time, \( a \) is the earliness penalty coefficient, and \( b \) is the tardiness penalty coefficient. A common due date \( d \) is determined as \( d = \lfloor \sum p \times h \rfloor \), where \( h \) is a predefined fraction (defaulting to 0.6). The goal is to determine an optimal job sequence that minimizes the penalty, calculated as follows: for each job, if its completion time \( C \) is earlier than \( d \), an earliness penalty of \( a \times (d - C) \) is incurred; if \( C \) exceeds \( d \), a tardiness penalty of \( b \times (C - d) \) is applied; otherwise, no penalty is incurred. The problem requires finding a permutation of job indices (1-based) that minimizes the total penalty. The evaluation metric sums these penalties for a given schedule.
'''


def solve(**kwargs):
    """
    Solves the restricted single‐machine common due date scheduling problem.

    The problem:
       Given a list of jobs where each job is represented as a tuple (p, a, b):
         • p: processing time
         • a: earliness penalty coefficient
         • b: tardiness penalty coefficient
       and an optional parameter h (default 0.6), the common due date is computed as:
             d = floor(sum(p) * h)
       A schedule (i.e., a permutation of job indices in 1‐based numbering) is produced.
       When processing the jobs in that order, the penalty is computed by:
         • Adding a × (d − C) if a job’s completion time C is less than d,
         • Adding b × (C − d) if C is greater than d,
         • No penalty if C equals d.
       The objective is to minimize the total penalty.

    Input kwargs:
         - 'jobs' (List[Tuple[int, int, int]]): a list of tuples where each tuple represents a job with:
              • p (int): processing time,
              • a (int): earliness penalty coefficient,
              • b (int): tardiness penalty coefficient.
         - Optional: 'h' (float): the factor used to compute the common due date (default is 0.6).

    Evaluation Metric:
         The computed schedule is evaluated by accumulating processing times and applying
         the appropriate earliness/tardiness penalties with respect to the common due date.

    Returns:
         A dictionary with key 'schedule' whose value is a list of integers representing
         a valid permutation of job indices (1-based).
    """
    # Placeholder implementation: simply return the jobs in their original order.
    jobs = kwargs.get('jobs', [])
    n = len(jobs)
    return {'schedule': list(range(1, n + 1))}


def load_data(input_path):
    """
    Reads the input file and returns a list of cases.

    Each case is represented as a dictionary containing:
         - 'jobs': a list of tuples (p, a, b) for each job.
         - 'h': a float parameter for due date computation (default set to 0.6).

    The input file format:
         • The first token is an integer T indicating the number of cases.
         • For each case:
               – The first integer is n, the number of jobs.
               – The following n lines each contain three space-separated integers: p, a, and b.

    Returns:
         List[dict]: A list where each element is a dictionary with the keys 'jobs' and 'h'.
    """
    cases = []
    try:
        with open(input_path, 'r') as f:
            tokens = f.read().strip().split()
    except Exception as e:
        raise ValueError(f"Error reading input file: {e}")

    index = 0
    try:
        T = int(tokens[index])
    except Exception as e:
        raise ValueError("Invalid input format: first token must be an integer (number of cases).")
    index += 1

    for t in range(T):
        if index >= len(tokens):
            raise ValueError(f"Unexpected end of input while reading case {t + 1}.")
        try:
            n = int(tokens[index])
        except Exception as e:
            raise ValueError(f"Invalid job count for case {t + 1}.")
        index += 1

        jobs = []
        for i in range(n):
            if index + 2 >= len(tokens):
                raise ValueError(f"Unexpected end of input while reading job data for case {t + 1}.")
            try:
                p = int(tokens[index])
                a = int(tokens[index + 1])
                b = int(tokens[index + 2])
            except Exception as e:
                raise ValueError(f"Invalid job data for job {i + 1} in case {t + 1}.")
            index += 3
            jobs.append((p, a, b))

        # For each case, we include the jobs and set a default h value (can be adjusted if needed)
        cases.append({'jobs': jobs, 'h': 0.6})

    return cases


def eval_func(jobs, schedule, h=0.6):
    """
    Evaluates the quality of a schedule for the restricted single‐machine common due date problem.

    Parameters:
         - jobs (List[Tuple[int, int, int]]): List of jobs, each represented as (p, a, b).
         - schedule (List[int]): A permutation (1-based indices) representing the processing order.
         - h (float): Factor for computing the common due date d = floor(sum(p) * h).

    Returns:
         int: The total penalty computed for the schedule.

    The evaluation:
         1. Compute d = floor(total_processing_time * h).
         2. Process jobs in the given order, accumulating processing times.
         3. For each job, if the cumulative time C is less than d, add a penalty a × (d − C);
            if C is greater than d, add a penalty b × (C − d); no penalty is incurred if C equals d.
         4. Sum the penalties to yield the total score.
    """
    total_processing = sum(p for p, a, b in jobs)
    d = int(total_processing * h)  # floor operation via int conversion for non-negative totals

    cumulative_time = 0
    total_penalty = 0
    # Validate that schedule is a permutation of 1..n
    n = len(jobs)
    if sorted(schedule) != list(range(1, n + 1)):
        raise ValueError(f"Schedule must be a permutation of 1 to {n}. Provided schedule: {schedule}")

    for idx in schedule:
        try:
            p, a, b = jobs[idx - 1]  # Convert from 1-based to 0-based indexing
        except IndexError:
            raise ValueError(f"Job index {idx} is out of bounds for jobs list of length {n}.")
        cumulative_time += p
        if cumulative_time < d:
            total_penalty += a * (d - cumulative_time)
        elif cumulative_time > d:
            total_penalty += b * (cumulative_time - d)
        # No penalty if cumulative_time == d
    return total_penalty


def norm_score(results):
    """
    Given a dictionary `results` where each key is a test case filename (e.g., "sch10.txt")
    and the value is a tuple (scores, error_message), this function returns a new dictionary
    with the normed results. For each test case, the normed score for each k is computed as:

        norm = (optimal score for h=0.6) / (model's score)

    The optimal scores for h=0.6 are pre-defined for each job instance size n.
    If a score in the list is not numeric (e.g., "Timeout (10s)"), that entry is skipped.

    Parameters:
      results (dict): A dictionary where keys are filenames (e.g., "sch10.txt") and values
                      are tuples (scores, error_message).

    Returns:
      dict: A dictionary with the same keys, where each value is a list of normed scores
            computed for the numeric entries only.
    """
    # Pre-defined optimal scores for h = 0.6 by instance size (n)
    optimal_scores = {
        10: [841, 615, 793, 815, 521, 755, 1101, 610, 582, 710],
        20: [2986, 3260, 3600, 3336, 2206, 3016, 4175, 1638, 1992, 2116],
        50: [17990, 14231, 16497, 14105, 14650, 14251, 17715, 21367, 14298, 14377],
        100: [72019, 59351, 68537, 69231, 55291, 62519, 62213, 80844, 58771, 61419],
        200: [254268, 266028, 254647, 297269, 260455, 236160, 247555, 225572, 255029, 269236],
        500: [1581233, 1715332, 1644947, 1640942, 1468325, 1413345, 1634912, 1542090, 1684055, 1520515],
        1000: [6411581, 6112598, 5985538, 6096729, 6348242, 6082142, 6575879, 6069658, 6188416, 6147295],
    }

    normed = {}
    for case, (scores, error_message) in results.items():
        # Try to extract the number of jobs (n) from the filename.
        # Expected format: "sch{n}.txt", e.g., "sch10.txt" -> n = 10.
        try:
            n_val = int(case.replace("sch", "").replace(".txt", ""))
        except ValueError:
            continue  # Skip if the filename is not in expected format.

        # Only process if we have optimal scores for this instance size.
        if n_val not in optimal_scores:
            continue

        optimal_list = optimal_scores[n_val]
        normed_scores = []
        # Process each score in the scores list, along with its index (for k=1,...,10).
        for idx, score in enumerate(scores):
            # If the score is not numeric, skip it.
            if isinstance(score, (int, float)):
                # Compute normalized score as (optimal / model score)
                norm_val = optimal_list[idx] / score
                normed_scores.append(norm_val)
            else:
                normed_scores.append(score)
        normed[case] = (normed_scores, error_message)

    return normed


def get_dev():
    dev = {'sch10.txt': [4, 5, 6], 'sch100.txt': [9, 8, 5], 'sch1000.txt': [4, 9, 0],
           'sch20.txt': [6, 5, 3], 'sch200.txt': [2, 4, 5], 'sch50.txt': [1, 8, 2],
           'sch500.txt': [3, 6, 9]}

    return dev
