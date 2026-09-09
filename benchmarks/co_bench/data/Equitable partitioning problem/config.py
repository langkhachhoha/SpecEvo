DESCRIPTION = '''The task is to partition a set of individuals—each characterized by multiple binary attributes—into exactly 8 groups such that the distribution of attribute values is as balanced as possible across these groups. For each attribute, count the number of individuals with a ‘1’ in each group. The optimization objective is to minimize the total imbalance, which is defined as follows: for each attribute, calculate the absolute differences between the count in each group and the mean count across all groups, take the average of these differences, and then sum these averages over all attributes. The goal is to determine a group assignment for each individual that achieves the lowest possible total imbalance score.'''


def solve(**kwargs):
    """
    Partition individuals into 8 groups so that for every binary attribute the count of 1's is as evenly
    distributed across the groups as possible.

    Input kwargs:
      - data (list of list of int): A matrix where each inner list represents the binary attributes (0 or 1)
        of one individual.

    Evaluation Metric:
      For each attribute, calculate the number of 1’s in each group,
      then compute the absolute difference between each group’s count and the mean count for that attribute.
      Average these differences over all groups to obtain the attribute’s imbalance.
      The final score is the sum of these attribute imbalances across all attributes.
      A lower score indicates a more balanced partitioning.

    Returns:
      dict: A dictionary with one key 'assignment' whose value is a list of positive integers (one per individual)
            indicating the group assignment (using 1-based indexing). For example:
            { "assignment": [1, 1, 1, ...] }
    """
    # --- Placeholder solution ---
    # For this placeholder, we assign every individual to group 1.
    data = kwargs.get('data', [])
    num_individuals = len(data)
    return {'assignment': [1] * num_individuals}


def load_data(input_path):
    """
    Reads an input file where each non-empty line represents an individual with space-separated binary attributes.
    In case the input file contains multiple cases (separated by one or more blank lines), this function will
    separate them into distinct cases.

    Parameters:
        input_path (str): The file path to the input data.

    Returns:
        list: A list of dictionaries. Each dictionary represents one case with the key 'data' mapping to a 2D list
              (matrix) of binary attributes (0 or 1). For example:
              [
                  {"data": [[0, 1, 0], [1, 0, 1], ...]},
                  {"data": [[1, 1], [0, 1], ...]},
                  ...
              ]

    Raises:
        Exception: If the file cannot be read, or if any line is invalid, contains non-integer tokens,
                   tokens not in {0, 1}, or if any row has an inconsistent number of attributes.
    """
    try:
        with open(input_path, 'r') as f:
            raw_lines = f.readlines()
    except Exception as e:
        raise Exception("Error reading input file: " + str(e))

    cases = []
    current_case = []
    for line_no, line in enumerate(raw_lines, start=1):
        stripped = line.strip()
        # A blank line indicates a separator between cases.
        if not stripped:
            if current_case:
                cases.append(current_case)
                current_case = []
            continue
        current_case.append(stripped)

    # Add last case if file did not end with a blank line.
    if current_case:
        cases.append(current_case)

    # Parse each case into a data matrix.
    list_of_cases = []
    for case_idx, case_lines in enumerate(cases, start=1):
        matrix = []
        n_attributes = None
        for line_no, line in enumerate(case_lines, start=1):
            tokens = line.split()
            if not tokens:
                raise Exception(f"Case {case_idx}, line {line_no} is empty or invalid.")
            try:
                row = [int(token) for token in tokens]
            except ValueError:
                raise Exception(f"Non-integer value found in case {case_idx}, line {line_no}.")
            for token in row:
                if token not in (0, 1):
                    raise Exception(
                        f"Invalid attribute value {token} found in case {case_idx}, line {line_no}; expected only 0 or 1.")
            if n_attributes is None:
                n_attributes = len(row)
            elif len(row) != n_attributes:
                raise Exception(f"Inconsistent number of attributes in case {case_idx}, line {line_no}.")
            matrix.append(row)
        list_of_cases.append({"data": matrix})

    if not list_of_cases:
        raise Exception("Input file is empty or contains no valid cases.")

    return list_of_cases


def eval_func(**kwargs):
    """
    Evaluates a partitioning solution for the equitable distribution problem using the new imbalance metric.

    Expected Parameters (provided via kwargs):
      - data (list of list of int): A matrix of binary attributes for individuals.
      - assignment (list of int): A list of positive integers representing group assignments for each individual.

    Evaluation Metric:
      For each attribute (column), compute the number of 1's per group. Then, compute the mean of these counts.
      The imbalance for the attribute is defined as the average of the absolute differences between each group's count and the mean count.
      The final score is the sum of these imbalances over all attributes.
      (A lower score indicates a more balanced partitioning.)

    Returns:
      dict: A dictionary containing:
            - 'total_imbalance': The computed total imbalance (float).

    Raises:
      Exception: If any expected parameter is missing, if the assignment format is invalid, or if the number of groups is not 8.
    """
    # Retrieve input data and assignment from kwargs
    if 'data' not in kwargs or 'assignment' not in kwargs:
        raise Exception("Missing required input parameters 'data' and/or 'assignment'.")

    data = kwargs['data']
    assignment = kwargs['assignment']
    #
    n_individuals = len(data)
    if len(assignment) != n_individuals:
        raise Exception(f"Expected {n_individuals} group assignments but found {len(assignment)}.")

    n_attributes = len(data[0])
    for idx, row in enumerate(data, start=1):
        if len(row) != n_attributes:
            raise Exception(f"Inconsistent number of attributes in data at individual {idx}.")

    # Ensure all group assignments are positive integers.
    for idx, g in enumerate(assignment, start=1):
        if not isinstance(g, int) or g < 1:
            raise Exception(f"Invalid group assignment at position {idx}: {g}. Must be a positive integer.")

    # Collect unique groups and check for exactly 8 groups.
    groups = set(assignment)
    if len(groups) != 8:
        raise Exception(f"Invalid number of groups: expected 8, but got {len(groups)}.")

    # Initialize per-group attribute sums.
    group_sums = {g: [0] * n_attributes for g in groups}
    for ind, group in enumerate(assignment):
        for j in range(n_attributes):
            group_sums[group][j] += data[ind][j]

    total_imbalance = 0.0
    for j in range(n_attributes):
        # Collect counts for attribute j from all groups
        attr_counts = [group_sums[g][j] for g in groups]
        mean_count = sum(attr_counts) / len(groups)
        # Compute average absolute difference from the mean
        # imbalance = sum(abs(count - mean_count) for count in attr_counts) / len(groups)
        imbalance = sum(abs(count - mean_count) for count in attr_counts)
        total_imbalance += imbalance

    return total_imbalance


def norm_score(results):
    optimal_scores = {
        "eppperf1.txt": [0],
        "eppperf2.txt": [0],
        "eppperf3.txt": [0],
        "eppperf4.txt": [0],
        "eppperf5.txt": [0],
        "epprandom1.txt": [11.5],
        "epprandom2.txt": [12.75],
        "epprandom3.txt": [13.75],
        "epprandom4.txt": [14.50],
        "epprandom5.txt": [16.25],
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
                if optimal_list[idx] == 0:
                    normed_scores.append((optimal_list[idx] + 1) / (score + 1))
                else:
                    normed_scores.append(optimal_list[idx] / score)
            else:
                normed_scores.append(score)
        normed[case] = (normed_scores, error_message)

    return normed


def get_dev():
    dev = {'eppperf1.txt': [0], 'eppperf3.txt': [0],
           'epprandom2.txt': [0], 'epprandom4.txt': [0]}

    return dev
