DESCRIPTION = '''The Uncapacitated Warehouse Location Problem aims to determine which warehouses to open and how to assign each customer entirely to an open warehouse in order to minimize the total cost. Given a set of potential warehouse locations, each with a fixed opening cost, and a set of customers, each with an associated assignment cost for being served by each warehouse, the objective is to select a subset of warehouses to open and assign every customer completely to one of these open warehouses. The optimization minimizes the sum of fixed warehouse opening costs and the customer assignment costs. Each customer must be assigned to exactly one warehouse; if any customer is left unassigned or assigned to more than one warehouse, the solution is considered infeasible.'''


def solve(**kwargs):
    """
    Solves the Uncapacitated Warehouse Location Problem.

    Input kwargs:
      - m: Number of potential warehouses (int)
      - n: Number of customers (int)
      - warehouses: A list of dictionaries, each with keys:
            'fixed_cost': Fixed cost for opening the warehouse.
      - customers: A list of dictionaries, each with keys:
            'costs': A list of floats representing the cost of assigning the entire customer to each warehouse.

    Evaluation Metric:
      The objective is to minimize the total cost, computed as:
         (Sum of fixed costs for all open warehouses)
       + (Sum of assignment costs for each customer assigned to a warehouse)
      Each customer must be assigned entirely to exactly one open warehouse.
      If a solution violates this constraint (i.e., a customer is unassigned or is assigned to more than one warehouse), then the solution is considered infeasible and no score is provided.

    Returns:
      A dictionary with the following keys:
         'total_cost': (float) The computed objective value (cost) if the solution is feasible; otherwise, no score is provided.
         'warehouse_open': (list of int) A list of m integers (0 or 1) indicating whether each warehouse is closed or open.
         'assignments': (list of list of int) A 2D list (n x m) where each entry is 1 if customer i is assigned to warehouse j, and 0 otherwise.
    """
    ## placeholder. You do not need to write anything here.
    return {
        "total_cost": 0.0,
        "warehouse_open": [0] * kwargs["m"],
        "assignments": [[0] * kwargs["m"] for _ in range(kwargs["n"])]
    }


def load_data(input_path):
    """
    Reads one or more problem cases from the input file.

    Expected Input File Format for each case:
      Line 1: Two integers: m n
      Next m lines: Each line contains two numbers: capacity fixed_cost for a warehouse.
      Next n lines: Each line contains: demand (a number) followed by m numbers representing the cost of
                  allocating the customer's demand to each warehouse.

    If the input file contains multiple cases, the cases appear sequentially in the file.

    Returns:
      A list of dictionaries, each corresponding to one case. Each dictionary has the keys:
         - 'm': Number of potential warehouses (int)
         - 'n': Number of customers (int)
         - 'warehouses': List of dictionaries; each with keys 'capacity' and 'fixed_cost'
         - 'customers': List of dictionaries; each with keys 'demand' and 'costs' (list of floats)
    """
    try:
        with open(input_path, 'r') as fin:
            input_lines = fin.readlines()
    except Exception as e:
        raise ValueError("Error reading input file: " + str(e))

    # Tokenize all non-empty lines.
    tokens = []
    for line in input_lines:
        line = line.strip()
        if line:
            tokens.extend(line.split())

    cases = []
    index = 0
    total_tokens = len(tokens)

    # Process tokens until we have exhausted them.
    while index < total_tokens:
        if index + 1 >= total_tokens:
            raise ValueError("Insufficient tokens to read m and n for a case.")
        try:
            m = int(tokens[index])
            n = int(tokens[index + 1])
        except Exception as e:
            raise ValueError("Error parsing m or n: " + str(e))
        index += 2

        # Parse warehouse data (m warehouses, each with 2 tokens).
        expected_warehouse_tokens = m * 2
        if index + expected_warehouse_tokens - 1 >= total_tokens:
            raise ValueError("Not enough tokens for warehouse data in a case.")
        warehouses = []
        for i in range(m):
            try:
                capacity = float(tokens[index])
                fixed_cost = float(tokens[index + 1])
            except Exception as e:
                raise ValueError("Error parsing warehouse data: " + str(e))
            warehouses.append({'capacity': capacity, 'fixed_cost': fixed_cost})
            index += 2

        # Parse customer data (n customers, each with 1 demand and m cost values).
        customers = []
        for j in range(n):
            if index >= total_tokens:
                raise ValueError(f"Not enough tokens for customer {j + 1} demand.")
            try:
                demand = float(tokens[index])
            except Exception as e:
                raise ValueError(f"Error parsing demand for customer {j + 1}: " + str(e))
            index += 1
            if index + m - 1 >= total_tokens:
                raise ValueError(f"Not enough tokens for cost data for customer {j + 1}.")
            costs = []
            for i in range(m):
                try:
                    cost = float(tokens[index])
                except Exception as e:
                    raise ValueError(f"Error parsing cost for customer {j + 1}, warehouse {i + 1}: " + str(e))
                costs.append(cost)
                index += 1
            customers.append({'demand': demand, 'costs': costs})

        case_data = {"m": m, "n": n, "warehouses": warehouses, "customers": customers}
        cases.append(case_data)

    return cases


def eval_func(m, n, warehouses, customers, warehouse_open, assignments, **kwargs):
    """
    Evaluates the solution for the Uncapacitated Warehouse Location Problem.

    For each customer:
      - The customer must be assigned to exactly one open warehouse.
      - The assignment cost is the cost associated with the warehouse to which the customer is assigned.
      - No assignment is allowed for a warehouse that is closed.

    The total cost is computed as:
         (Sum of fixed costs for all open warehouses)
       + (Sum of assignment costs for all customers)

    Input Parameters:
      - m: Number of potential warehouses (int)
      - n: Number of customers (int)
      - warehouses: List of dictionaries, each with keys:
            'fixed_cost': The fixed cost for opening the warehouse.
            'capacity': Provided but ignored in this problem.
      - customers: List of dictionaries, each with keys:
            'costs': A list of floats representing the cost of assigning the customer entirely to each warehouse.
            'demand': Provided but ignored in this problem.
      - warehouse_open: List of m integers (0 or 1) indicating whether each warehouse is closed or open.
      - assignments: List of n lists (each of length m) where assignments[j][i] is 1 if customer j is assigned to warehouse i, and 0 otherwise.
      - kwargs: Other parameters (not used here).

    Returns:
      A floating-point number representing the total cost if the solution is feasible.

    Raises:
      Exception: If any of the following conditions are violated:
          - The sum of assignments for any customer is not exactly 1.
          - Any positive assignment is made to a closed warehouse.
          - Any assignment value is not binary (0 or 1).
    """
    computed_total_cost = 0.0

    # Add fixed costs for open warehouses.
    for i in range(m):
        if warehouse_open[i] == 1:
            computed_total_cost += warehouses[i]['fixed_cost']

    # Evaluate assignment cost for each customer.
    for j in range(n):
        # Sum of assignments for customer j should be exactly 1.
        assigned_sum = sum(assignments[j])
        if abs(assigned_sum - 1.0) > 1e-6:
            raise Exception(
                f"Customer {j} assignment violation: total assigned value {assigned_sum} does not equal 1."
            )

        customer_cost = 0.0
        for i in range(m):
            allocation = assignments[j][i]
            # Ensure the assignment is binary (allowing for small floating point tolerance)
            if not (abs(allocation) < 1e-6 or abs(allocation - 1.0) < 1e-6):
                raise Exception(
                    f"Customer {j} has a non-binary assignment value {allocation} for warehouse {i + 1}."
                )
            if allocation > 0:
                if warehouse_open[i] != 1:
                    raise Exception(
                        f"Customer {j} is assigned to warehouse {i + 1}, which is closed."
                    )
                # Since assignment is binary, add the corresponding cost.
                customer_cost += customers[j]['costs'][i]
        computed_total_cost += customer_cost

    return computed_total_cost


def norm_score(results):
    optimal_scores = {
        "cap71.txt": [932615.750],
        "cap72.txt": [977799.400],
        "cap73.txt": [1010641.450],
        "cap74.txt": [1034976.975],
        "cap101.txt": [796648.437],
        "cap102.txt": [854704.200],
        "cap103.txt": [893782.112],
        "cap104.txt": [928941.750],
        "cap131.txt": [793439.562],
        "cap132.txt": [851495.325],
        "cap133.txt": [893076.712],
        "cap134.txt": [928941.750],
        "capa.txt": [17156454.478],
        "capb.txt": [12979071.582],
        "capc.txt": [11505594.329]
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
    dev = {'cap101.txt': [], 'cap103.txt': [],
           'cap131.txt': [],
           'cap133.txt': [],
           'cap71.txt': [], 'cap73.txt': [],
           'capb.txt': []}

    return dev
