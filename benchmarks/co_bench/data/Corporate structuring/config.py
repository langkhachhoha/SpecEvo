DESCRIPTION = '''Given N countries, each defined by:
  • a tax code (1: Exemption, 2: Deduction, 3: Source-by-source Pooling, 4: World-wide Pooling),
  • a foreign income tax rate,
  • a domestic income tax rate, and
  • a profit,
and a withholding tax matrix W (where W[i][j] is the rate on dividends from country i to j), construct a valid tree‐structured corporate hierarchy (directed, acyclic, connected) rooted at a designated target (whose parent is 0) such that every country with profit > 0 appears exactly once.

For each country i, define S as the set of nodes in its subtree (note the subtree includes itself) with a positive profit. Also consider the set of child nodes C_i. 
If i is not a root country but in the tree, it will send all its income (after tax) to its parent j. Denote this amount as F[i][j]. Assume the total income after domestic tax and withholding tax for country i is: domestic_income_i*(1-domestic_rate_i) + (\sum_{k\in C_i} F[k][i]*(1-W[k][i]))
The extra foreign tax under different tax code is defined as follows:
    1. No extra tax.
    2. Foreign income tax from the child nodes: foreign_income_rate_i*(\sum_{k\in C_i} F[k][i]*(1-W[k][i]))
    3. Foreign income tax computed from the source nodes in each child's subtree: \sum_{k\in C_i} max(0, F[k][i]*(1-W[k][i]) - (1-foreign_income_rate_i)*(\sum_{s\in S_k} domestic_income_s))
    4. Foreign income tax from all source nodes in the subtree, excluding itself: max(0, \sum_{k\in C_i}  F[k][i]*(1-W[k][i]) - (1-foreign_income_rate_i)*((\sum_{s\in S_i} domestic_income_s)-domestic_income_i))'''


def solve(**kwargs):
    """
    Input kwargs:
      - N: (int) The number of countries.
      - target: (int) The target country (1-indexed) which must be the root (its parent is 0).
      - countries: (dict) Mapping country id (1-indexed) to a tuple:
                 (tax_code, foreign_income_tax_rate, domestic_income_tax_rate, profit).
      - withholding: (dict of dict) A nested dictionary where withholding[i][j] is the withholding tax rate
                     applied when country i sends dividends to country j.

    Returns:
      A dictionary with the key "structure" whose value is a dictionary representing the corporate tree,
      where each key is a child country and its value is the immediate parent (with the target country having parent 0).

      (Note: This is a placeholder implementation.)
    """
    # --- Placeholder implementation ---
    # For demonstration, we simply return a structure that includes only the target country.
    structure = {kwargs['target']: 0}
    # In an actual solution, you would build a tree covering all countries with positive profit.
    return {"structure": structure}


def load_data(input_path):
    """
    Reads an input file that may contain one or more cases.

    File Format for each case:
      - Line 1: Two space-separated numbers: N target
      - Next N lines: For each country i (1-indexed), four space-separated values:
                         tax_code, foreign_income_tax_rate, domestic_income_tax_rate, profit
      - Remaining tokens: N*N floating-point numbers representing the withholding tax matrix.
        (These numbers can be spread across one or more lines.)

    Returns:
      A list of dictionaries. Each dictionary corresponds to one test case and has the keys:
         - "N": (int) number of countries.
         - "target": (int) target country (1-indexed).
         - "countries": (dict) mapping each country id to its tuple of (tax_code, foreign_rate, domestic_rate, profit).
         - "withholding": (dict of dict) where withholding[i][j] is the withholding tax rate from country i to j.
    """
    cases = []
    with open(input_path, 'r') as f:
        # Read all non-empty lines.
        lines = [line.strip() for line in f if line.strip() != '']

    i = 0
    total_lines = len(lines)
    while i < total_lines:
        # Parse first line of a case: N and target.
        parts = lines[i].split()
        if len(parts) < 2:
            raise ValueError("Expected N and target on line {}.".format(i + 1))
        N = int(parts[0])
        target = int(parts[1])
        i += 1

        # Parse country data.
        if i + N > total_lines:
            raise ValueError("Not enough lines for country data in a case starting at line {}.".format(i + 1))
        countries = {}
        for country in range(1, N + 1):
            parts = lines[i].split()
            if len(parts) < 4:
                raise ValueError("Incomplete country data at line {}.".format(i + 1))
            tax_code = int(parts[0])
            foreign_rate = float(parts[1])
            domestic_rate = float(parts[2])
            profit = float(parts[3])
            countries[country] = (tax_code, foreign_rate, domestic_rate, profit)
            i += 1

        # Read all remaining tokens for the withholding tax matrix.
        withholding_tokens = []
        # We'll assume that the withholding matrix occupies the next N*N tokens.
        while i < total_lines and len(withholding_tokens) < N * N:
            withholding_tokens.extend(lines[i].split())
            i += 1

        if len(withholding_tokens) < N * N:
            raise ValueError("Incomplete withholding tax matrix: expected {} numbers, got {}.".format(N * N,
                                                                                                      len(withholding_tokens)))

        # Build the withholding matrix from tokens.
        withholding = {}
        token_index = 0
        for country in range(1, N + 1):
            withholding[country] = {}
            for j in range(1, N + 1):
                withholding[country][j] = float(withholding_tokens[token_index])
                token_index += 1

        # Append the parsed case to the list.
        cases.append({
            "N": N,
            "target": target,
            "countries": countries,
            "withholding": withholding
        })
    return cases


def eval_func(N, target, countries, withholding, structure):
    """
    Evaluates the score of a given tree structure.

    Inputs:
      - N: Number of countries.
      - target: The designated target country (1-indexed) that is the root (its parent is 0).
      - countries: A dict mapping country id (1-indexed) to a tuple:
              (tax_code, foreign_income_tax_rate, domestic_income_tax_rate, profit)
      - withholding: A dict of dicts where withholding[i][j] is the withholding tax rate
                     applied when country i sends dividends to j.
      - structure: A dict representing the corporate tree. Each key is a country (child) and its
                   value is its immediate parent (for the target, parent is 0).

    Returns:
      The score, defined as:
          total_profit = (sum of profits for all countries) - (total_tax)
      where total_tax is the sum of domestic tax and extra foreign tax paid in the tree.
    """

    # Build a mapping from each node to its children from the tree structure.
    children = {i: [] for i in range(1, N + 1)}
    for child, parent in structure.items():
        if parent != 0:  # Only non-root nodes appear in the structure mapping.
            children[parent].append(child)
    # It is possible that some countries (e.g. with profit <= 0) are not in the structure.
    # They will not incur any tax in the corporate hierarchy.

    # First, compute P[i] = sum of profits (only if >0) in the subtree of i.
    # This is used in the pooling tax rules.

    P_cache = {}

    def get_P(i):
        if i in P_cache:
            return P_cache[i]
        # Only count profit if positive (i.e. the node is a "source")
        profit_i = countries[i][3]
        total = profit_i
        for c in children.get(i, []):
            total += get_P(c)
        P_cache[i] = total
        return total

    for i in range(1, N + 1):
        P_cache[i] = get_P(i)

    print(P_cache)

    def outcome(i):
        d_income = countries[i][3] * (1 - countries[i][2])
        f_income = foreign_income(i)
        total_f_income = sum(f_income.values())
        if countries[i][0] == 1:
            return d_income + total_f_income
        elif countries[i][0] == 2:
            return d_income + total_f_income * (1 - countries[i][1])
        elif countries[i][0] == 3:
            return d_income + total_f_income - sum(
                [max(0, f_income[c] - (1 - countries[i][1]) * P_cache[c]) for c in children[i]])
        else:
            return d_income + total_f_income - max(0, total_f_income - (1 - countries[i][1]) * (
                    P_cache[i] - countries[i][3]))

    def foreign_income(i):
        if len(children.get(i, [])) == 0:
            return {}
        else:
            total = {}
            for c in children.get(i, []):
                a = outcome(c)
                total[c] = a * (1 - withholding[c][i])
        return total

    return outcome(target)


def norm_score(results):
    optimal_scores = {
        "tax1.txt": [647.51],
        "tax2.txt": [2153.45],
        "tax3.txt": [4329.83],
        "tax4.txt": [3491.62],
        "tax5.txt": [5435.79],
        "tax6.txt": [5058.07],
        "tax7.txt": [11872.37],
        "tax8.txt": [10206.65],
        "tax9.txt": [16584.32],
        "tax10.txt": [455],
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
                normed_scores.append(score / optimal_list[idx])
            else:
                normed_scores.append(score)
        normed[case] = (normed_scores, error_message)

    return normed


def get_dev():
    dev = {'tax1.txt': [], 'tax3.txt': [], 'tax5.txt': [],
           'tax7.txt': [], 'tax9.txt': []}

    return dev
