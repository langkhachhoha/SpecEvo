DESCRIPTION = '''The Capacitated Warehouse Location Problem with Splittable Demand aims to determine which warehouses to open and how to allocate portions of customer demands among these warehouses in order to minimize total costs. Given a set of potential warehouse locations, each with a fixed opening cost and capacity limit, and a set of customers with individual demands and associated per-unit assignment costs to each warehouse, the objective is to decide which warehouses to open and how to distribute each customer's demand among these open warehouses. The allocation must satisfy the constraint that the sum of portions assigned to each customer equals their total demand, and that the total demand allocated to any warehouse does not exceed its capacity. The optimization seeks to minimize the sum of fixed warehouse opening costs and the total per-unit assignment costs. However, if any solution violates these constraints (i.e., a customer’s demand is not fully satisfied or a warehouse’s capacity is exceeded), then no score is provided.'''


def solve(**kwargs):
    """
    Solves the Capacitated Warehouse Location Problem with Splittable Customer Demand.

    Input kwargs:
  - m (int): Number of potential warehouses
  - n (int): Number of customers
  - warehouses (list of dict): A list of dictionaries, each with keys 'capacity' (float) and 'fixed_cost' (float)
  - customers (list of dict): A list of dictionaries, each with keys 'demand' (float) and 'costs' (list of float) representing the per-unit assignment cost from each warehouse

    Evaluation Metric:
      The objective is to minimize the total cost, computed as:
         (Sum of fixed costs for all open warehouses)
       + (Sum of per-unit assignment costs for each unit of demand allocated from warehouses to customers)
      For each customer, the sum of allocations from all warehouses must equal the customer's demand.
      For each warehouse, the total allocated demand across all customers must not exceed its capacity.
      If a solution violates any of these constraints, the solution is considered infeasible and no score is provided.

    Returns:
      A dictionary with the following keys:
         'total_cost': (float) The computed objective value (cost) if the solution is feasible;
                         otherwise, no score is provided.
         'warehouse_open': (list of int) A list of m integers (0 or 1) indicating whether each warehouse is closed or open.
         'assignments': (list of list of float) A 2D list (n x m) where each entry represents the amount of customer i's demand supplied by warehouse j.
    """
    ## placeholder. You do not need to write anything here.
    return {
        "total_cost": 0.0,
        "warehouse_open": [0] * kwargs["m"],
        "assignments": [[0.0] * kwargs["m"] for _ in range(kwargs["n"])]
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
    Evaluates the solution for the Capacitated Warehouse Location Problem with Splittable Customer Demand,
    using a weighted average cost for each customer.

    For each customer:
      - The sum of allocations across warehouses must equal the customer's demand.
      - The assignment cost is computed as the weighted average of the per-unit costs,
        i.e., for each warehouse i, the fraction of demand allocated from i multiplied by its cost.
      - No positive allocation is allowed for a warehouse that is closed.

    Additionally, for each warehouse:
      - The total allocated demand must not exceed its capacity.

    The total cost is computed as:
         (Sum of fixed costs for all open warehouses)
       + (Sum over customers of the weighted average assignment cost)

    Input Parameters:
      - m: Number of potential warehouses (int)
      - n: Number of customers (int)
      - warehouses: List of dictionaries (each with 'capacity' and 'fixed_cost')
      - customers: List of dictionaries (each with 'demand' and 'costs' (list of floats representing per-unit cost))
      - warehouse_open: List of m integers (0 or 1) indicating whether each warehouse is closed or open.
      - assignments: List of n lists (each of length m) where assignments[j][i] represents the amount of
                     customer j's demand allocated to warehouse i.
      - kwargs: Other parameters (not used here).

    Returns:
      A floating-point number representing the total cost if the solution is feasible.

    Raises:
      Exception: If any of the following conditions are violated:
          - The sum of allocations for any customer does not equal its demand.
          - Any positive allocation is made to a closed warehouse.
          - Any warehouse's total allocated demand exceeds its capacity.
    """
    computed_total_cost = 0.0

    # Add fixed costs for open warehouses.
    for i in range(m):
        if warehouse_open[i] == 1:
            computed_total_cost += warehouses[i]['fixed_cost']

    # Evaluate assignment cost for each customer as a weighted average.
    for j in range(n):
        customer_demand = customers[j]['demand']
        allocated_amount = sum(assignments[j])
        if abs(allocated_amount - customer_demand) > 1e-6:
            raise Exception(
                f"Customer {j} demand violation: total assigned amount {allocated_amount} does not equal demand {customer_demand}."
            )
        weighted_cost = 0.0
        for i in range(m):
            allocation = assignments[j][i]
            if allocation < 0:
                raise Exception(
                    f"Customer {j} has a negative allocation {allocation} for warehouse {i + 1}."
                )
            if allocation > 0 and warehouse_open[i] != 1:
                raise Exception(
                    f"Customer {j} has allocation {allocation} for warehouse {i + 1}, which is closed."
                )
            # Compute fraction of the customer's demand supplied from warehouse i.
            fraction = allocation / customer_demand if customer_demand > 0 else 0.0
            weighted_cost += fraction * customers[j]['costs'][i]
        # Add the weighted cost (applied once per customer).
        computed_total_cost += weighted_cost

    # Compute total demand allocated to each warehouse and check capacity constraints.
    assigned_demand = [0.0] * m
    for i in range(m):
        for j in range(n):
            assigned_demand[i] += assignments[j][i]
    for i in range(m):
        if assigned_demand[i] > warehouses[i]['capacity'] + 1e-6:
            excess = assigned_demand[i] - warehouses[i]['capacity']
            raise Exception(
                f"Warehouse {i + 1} exceeds its capacity by {excess} units."
            )

    return computed_total_cost


def norm_score(results):
    optimal_scores = {
        "cap41.txt": [1040444.375],
        "cap42.txt": [1098000.450],
        "cap43.txt": [1153000.450],
        "cap44.txt": [1235500.450],
        "cap51.txt": [1025208.225],
        "cap61.txt": [932615.750],
        "cap62.txt": [977799.400],
        "cap63.txt": [1014062.050],
        "cap64.txt": [1045650.250],
        "cap71.txt": [932615.750],
        "cap72.txt": [977799.400],
        "cap73.txt": [1010641.450],
        "cap74.txt": [1034976.975],
        "cap81.txt": [838499.288],
        "cap82.txt": [910889.563],
        "cap83.txt": [975889.563],
        "cap84.txt": [1069369.525],
        "cap91.txt": [796648.438],
        "cap92.txt": [855733.500],
        "cap93.txt": [896617.538],
        "cap94.txt": [946051.325],
        "cap101.txt": [796648.437],
        "cap102.txt": [854704.200],
        "cap103.txt": [893782.112],
        "cap104.txt": [928941.750],
        "cap111.txt": [826124.713],
        "cap112.txt": [901377.213],
        "cap113.txt": [970567.750],
        "cap114.txt": [1063356.488],
        "cap121.txt": [793439.563],
        "cap122.txt": [852524.625],
        "cap123.txt": [895302.325],
        "cap124.txt": [946051.325],
        "cap131.txt": [793439.562],
        "cap132.txt": [851495.325],
        "cap133.txt": [893076.712],
        "cap134.txt": [928941.750],
        "capa-8000.txt": [19240822.449],
        "capa-10000.txt": [18438046.543],
        "capa-12000.txt": [17765201.949],
        "capa-14000.txt": [17160439.012],
        "capb-5000.txt": [13656379.578],
        "capb-6000.txt": [13361927.449],
        "capb-7000.txt": [13198556.434],
        "capb-8000.txt": [13082516.496],
        "capc-5000.txt": [11646596.974],
        "capc-5750.txt": [11570340.289],
        "capc-6500.txt": [11518743.744],
        "capc-7250.txt": [11505767.394]
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
    dev = {'cap101.txt': [], 'cap112.txt': [],
           'cap123.txt': [],
           'cap134.txt': [],
           'cap41.txt': [], 'cap62.txt': [], 'cap73.txt': [], 'cap84.txt': [],
           'cap91.txt': [],
           'capa-12000.txt': [],
           'capb-5000.txt': [],
           'capc-7250.txt': []}

    return dev
