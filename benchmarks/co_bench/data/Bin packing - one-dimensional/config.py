DESCRIPTION = '''The **one-dimensional bin packing problem** seeks to minimize the number of bins required to pack a given set of items while ensuring that the sum of item sizes within each bin does not exceed the specified bin capacity. Given a test case with an identifier (`id`), a fixed `bin_capacity`, and a list of `num_items` with their respective sizes (`items`), the objective is to find a packing arrangement that uses the least number of bins. The solution is evaluated based on the total `num_bins` used, with invalid solutions (e.g., missing or duplicated items, or bins exceeding capacity) incurring a inf heavy penalty. The output must include the number of bins used and a valid assignment of item indices to bins.'''


def solve(**kwargs):
    """
    Solve the one-dimensional bin packing problem for a single test case.

    Input kwargs (for a single test case):
      - id:           The problem identifier (string)
      - bin_capacity: The capacity of each bin (int)
      - num_items:    The number of items (int)
      - items:        A list of item sizes (list of ints)
      - **kwargs:     Other unused keyword arguments

    Evaluation metric:
      - The solution is scored by the total number of bins used.
      - If the solution is invalid (e.g., items are missing or duplicated, or bin capacity is exceeded),
        a penalty of 1,000,000 is added.

    Returns:
      A dictionary with:
        - 'num_bins': An integer, the number of bins used.
        - 'bins': A list of lists, where each inner list contains the 1-based indices of items assigned to that bin.

    Note: This is a placeholder implementation.
    """
    # Placeholder: Replace with your bin packing solution.
    return {
        'num_bins': 0,
        'bins': []
    }


def load_data(input_file_path):
    """
    Load test cases from a TXT file for the bin packing problem.

    The input file format:
      1. The first nonempty line is an integer P, the number of test cases.
      2. For each test case:
         a. A line with the problem identifier (e.g., "u120_00").
         b. A line with three space-separated numbers: bin_capacity, num_items, best_known.
            (Note: bin_capacity and item sizes may be given as floats.)
         c. Then num_items lines, each with a number representing an item size.

    Returns:
      A list of dictionaries. Each dictionary contains the input data for one test case with keys:
        - 'id':           Problem identifier (string)
        - 'bin_capacity': Bin capacity (float)
        - 'num_items':    Number of items (int)
        - 'best_known':   Best known number of bins (int)
        - 'items':        List of item sizes (list of floats)
    """
    cases = []
    try:
        with open(input_file_path, 'r') as fin:
            # Get all nonempty, stripped lines.
            in_lines = [line.strip() for line in fin if line.strip() != '']
    except Exception as e:
        raise Exception("Error reading input file: " + str(e))

    if not in_lines:
        raise Exception("Input file is empty or improperly formatted.")

    try:
        num_cases = int(in_lines[0])
    except Exception as e:
        raise Exception("Error parsing the number of test cases: " + str(e))

    pos = 1
    for _ in range(num_cases):
        if pos >= len(in_lines):
            raise Exception("Unexpected end of file while reading a test case header.")
        # Read problem identifier.
        prob_id = in_lines[pos]
        pos += 1

        if pos >= len(in_lines):
            raise Exception(f"Missing header for problem {prob_id}.")
        header_parts = in_lines[pos].split()
        pos += 1
        if len(header_parts) < 3:
            raise Exception(
                f"Header for problem {prob_id} must contain bin capacity, number of items, and best known bins.")
        try:
            # Use float for bin_capacity since it might be provided as a float.
            bin_capacity = float(header_parts[0])
            num_items = int(header_parts[1])
            best_known = int(header_parts[2])
        except Exception as e:
            raise Exception(f"Error parsing header for problem {prob_id}: {e}")

        items = []
        for i in range(num_items):
            if pos >= len(in_lines):
                raise Exception(f"Unexpected end of file while reading items for problem {prob_id}.")
            try:
                # Parse item sizes as floats.
                item_size = float(in_lines[pos])
            except Exception as e:
                raise Exception(f"Error parsing item size for problem {prob_id} at line {pos + 1}: {e}")
            items.append(item_size)
            pos += 1

        cases.append({
            'id': prob_id,
            'bin_capacity': bin_capacity,
            'num_items': num_items,
            'best_known': best_known,
            'items': items
        })

    return cases


def eval_func(id, bin_capacity, num_items, best_known, items, num_bins, bins):
    """
    Evaluate the bin packing solution for a single test case.

    Parameters (from the input case and the solution):
      - id:           Problem identifier (string)
      - bin_capacity: Bin capacity (int)
      - num_items:    Number of items (int)
      - best_known:   Best known number of bins (int)
      - items:        List of item sizes (list of ints)
      - num_bins:     Number of bins used in the solution (int)
      - bins:         List of lists; each inner list contains 1-based item indices assigned to that bin.

    Returns:
      A scalar score (int). The score is the total number of bins used.
      If the solution is invalid (e.g., item indices are wrong, items not used exactly once, or bin capacity exceeded),
      a penalty of 1,000,000 is added to the score.
    """
    penalty = 1_000_000
    score = num_bins  # start with the number of bins used
    valid = True
    details = []

    # Check that the number of bin assignments matches num_bins.
    if len(bins) != num_bins:
        valid = False
        details.append("Declared number of bins does not match the number of bin assignments provided.")

    # Check each bin for capacity and valid item indices.
    # Also count item appearances.
    item_counts = [0] * (num_items + 1)  # index 0 unused
    for bin_index, bin_items in enumerate(bins, start=1):
        bin_total = 0
        for item_idx in bin_items:
            if item_idx < 1 or item_idx > num_items:
                valid = False
                details.append(f"Bin {bin_index} contains an invalid item index: {item_idx}.")
                continue
            bin_total += items[item_idx - 1]
            item_counts[item_idx] += 1
        if bin_total > bin_capacity:
            valid = False
            details.append(f"Bin {bin_index} exceeds capacity: total size {bin_total} > capacity {bin_capacity}.")

    # Check that every item appears exactly once.
    for i in range(1, num_items + 1):
        if item_counts[i] != 1:
            valid = False
            details.append(f"Item {i} appears {item_counts[i]} times (expected exactly once).")

    if not valid:
        score = 0
    else:
        score = best_known / score

    # For debugging purposes, one might print or log details.
    # For now, we simply return the computed score.
    return score


def get_dev():
    dev = {'binpack1.txt': [7, 5, 16, 9, 13], 'binpack2.txt': [1, 15, 16, 4, 18],
           'binpack3.txt': [10, 18, 0, 19, 14], 'binpack4.txt': [11, 3, 16, 18, 17],
           'binpack5.txt': [10, 13, 0, 11, 17], 'binpack6.txt': [18, 11, 0, 6, 2],
           'binpack7.txt': [12, 17, 9, 15, 13], 'binpack8.txt': [4, 11, 19, 6, 17]}

    return dev
