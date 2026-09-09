DESCRIPTION = '''The Period Vehicle Routing Problem requires planning delivery routes over a multi‐day planning period.

Each customer (other than the depot, whose id is 0) is provided with a list of candidate service schedules. A schedule is represented by a binary vector of length equal to the period (e.g., [1, 0, 1] for a 3‐day period), where a 1 in a given position indicates that the customer must be visited on that day. The decision maker must select exactly one candidate schedule for each customer.

For every day in the planning period, if a customer’s chosen schedule indicates a delivery (i.e., a 1),
then exactly one vehicle must visit that customer on that day. Otherwise, the customer should not be visited. The decision maker must also design, for each day, the tours for the vehicles. Each tour is a continuous route that starts at the depot (id 0) and, after visiting a subset of customers, returns to the depot. Each vehicle is only allowed to visit the depot once per day—namely, as its starting and ending point—and it is not allowed to return to the depot in the middle of a tour.

Moreover, each vehicle route must obey a capacity constraint: the total demand of the customers visited on that tour must not exceed the vehicle capacity each day. Although multiple vehicles are available per day (as specified by the input), not all available vehicles have to be used, but the number of tours in a given day cannot exceed the provided number of vehicles. In addition, the tours on each day must cover exactly those customers who require service per the selected schedules, and no customer may be visited more than once in a given day.

The objective is to choose a schedule for every customer and plan the daily tours so as to minimize the overall distance traveled
by all vehicles during the entire planning period. Distances are measured using Euclidean distance.'''


def solve(**kwargs):
    """
    Solves an instance of the Period Vehicle Routing Problem.

    Input kwargs includes:
      - depot: dict with keys:
            "id": int, always 0.
            "x": float, the x-coordinate.
            "y": float, the y-coordinate.
      - customers: list of dictionaries (with customer id ≠ 0) having keys:
            "id": int, the customer id.
            "x": float, the x-coordinate.
            "y": float, the y-coordinate.
            "demand": numeric, the customer demand.
            "schedules": list of candidate schedules, each a list (of length period_length) with binary entries.
      - vehicles_per_day: list of ints (length period_length) indicating the number of vehicles available each day.
      - vehicle_capacity: numeric, the capacity of each vehicle.
      - period_length: int, the number of days in the planning period.

    The solution must decide:
      1. Which service schedule (from the candidate schedules) is selected for each customer.
      2. For each day (days are 1-indexed), the daily tours: a list of tours—one per available vehicle.
         Each tour is a continuous route that starts at the depot (0), visits some customers (each exactly once),
         and returns to the depot. The depot may only appear as the first and last vertex in each tour.
         The number of tours for day d must be exactly equal to vehicles_per_day[d-1].

    The returned solution is a dictionary containing:
      - "selected_schedules": dict mapping each customer id (integer) to the chosen schedule (a list of binary integers).
      - "tours": dict mapping day (an integer between 1 and period_length) to a list of tours.
                 Each tour is a list of vertex ids (integers), starting and ending at the depot (id 0).
    """
    # ------------------------------

    return {
        "selected_schedules": ...,
        "tours": ...
    }


def load_data(file_path):
    """
    Reads a period vehicle routing problem file and returns a dictionary with the problem data.

    The file is expected to have the following format:
      Line 1: Two integers: <num_customers> <period_length>
              (Note: the depot is specified as customer_id = 0.)
      Line 2: A list of period_length integers representing the number of vehicles on each day.
      Line 3: A single number representing the constant capacity of every vehicle.
      Lines 4 onward: Each line represents a vertex (depot or customer) in the format:
                        customer_id x_coordinate y_coordinate demand possible_schedule_list
                        For the depot (customer_id = 0) the demand and schedule are omitted or ignored.
                        e.g., depot line: 0 30 40 0  
                              customer line: 1 37 52 7 [[1, 0], [0, 1]]

    Parameters:
      file_path (str): The path to the input TXT file.

    Returns:
      A dictionary with keys:
          - "period_length" (int)
          - "vehicles_per_day" (list of ints)
          - "vehicle_capacity" (number)
          - "depot": dict with keys: "id", "x", "y"
          - "customers": list of customer dictionaries (for customer id ≠ 0)
                      Each customer dictionary contains keys:
                          "id": int, the customer id.
                          "x": float, the x coordinate.
                          "y": float, the y coordinate.
                          "demand": float, the customer demand.
                          "schedules": list of lists, each sub-list is a binary schedule for the period.
    """
    import ast

    # Read file and filter out any empty lines.
    with open(file_path, 'r') as f:
        all_lines = [line.strip() for line in f if line.strip() != '']

    # Check that we have at least 3 lines for headers.
    if len(all_lines) < 3:
        raise ValueError("Insufficient data in the file. Expect at least three header lines.")

    # Parse header
    # First line: number of customers and period length:
    header1 = all_lines[0].split()
    if len(header1) != 2:
        print(header1)
        raise ValueError("The first line must have exactly 2 tokens: <num_customers> <period_length>.")
    try:
        num_customers = int(header1[0])
        period_length = int(header1[1])
    except Exception as e:
        raise ValueError("Error parsing the number of customers or period length.") from e

    # Second line: number of vehicles on each day
    vehicles_tokens = all_lines[1].split()
    if len(vehicles_tokens) != period_length:
        raise ValueError("The number of vehicle counts provided does not equal the period length.")
    try:
        vehicles_per_day = [int(x) for x in vehicles_tokens]
    except Exception as e:
        raise ValueError("Error parsing the vehicles per day.") from e

    # Third line: vehicle capacity (all vehicles have same capacity)
    try:
        vehicle_capacity = float(all_lines[2])
    except Exception as e:
        raise ValueError("Error parsing vehicle capacity.") from e

    depot = None
    customers = []
    # Process the remaining lines.
    for line in all_lines[3:]:
        # Split into at most five tokens; the first four are assumed to be id, x, y and demand.
        parts = line.split(maxsplit=4)
        if len(parts) < 3:
            continue  # Skip lines that do not have minimum required data.

        try:
            cid = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
        except Exception as ex:
            raise ValueError("Error parsing id or coordinates in line: " + line) from ex

        # Check for depot (id == 0). For depot, we ignore demand and schedule.
        if cid == 0:
            depot = {"id": cid, "x": x, "y": y}
            # Skip further processing of demand/schedules for the depot.
            continue

        # For a customer, we expect a demand value.
        if len(parts) < 4:
            raise ValueError("Insufficient data for customer (id=%s) in line: %s" % (cid, line))
        try:
            demand = float(parts[3])
        except Exception as ex:
            raise ValueError("Error parsing demand for customer (id=%s) in line: %s" % (cid, line)) from ex

        # Parse possible schedule if provided.
        schedules = []
        if len(parts) == 5:
            try:
                schedules = ast.literal_eval(parts[4])
            except Exception as ex:
                raise ValueError("Error parsing delivery schedules in line: " + line) from ex

        customers.append({
            "id": cid,
            "x": x,
            "y": y,
            "demand": demand,
            "schedules": schedules
        })

    # Optionally, you can check if depot was found.
    if depot is None:
        raise ValueError("Depot (customer id 0) was not found in the file.")

    return [{
        "period_length": period_length,
        "vehicles_per_day": vehicles_per_day,
        "vehicle_capacity": vehicle_capacity,
        "depot": depot,
        "customers": customers
    }]


def eval_func(**kwargs):
    """
    Evaluates the solution of the Period Vehicle Routing Problem for a single case.
    Input kwargs should include:
      - from data:
            "depot": dict with keys "id", "x", "y".
            "customers": list of customer dictionaries (each with keys "id", "x", "y", "demand", "schedules").
            "vehicles_per_day": list of ints (indicating the number of available vehicles per day).
            "vehicle_capacity": numeric, the capacity of each vehicle.
            "period_length": int, the number of days.
      - from solve:
            "selected_schedules": a mapping from customer id to the chosen schedule (a list of binary integers).
            "tours": a mapping from day (1-indexed) to a list of tours;
                     each tour is a list of vertex ids (integers), starting and ending at depot (id 0),
                     with no intermediate depot visits.

    The evaluator checks the following:
      1. For each customer (other than the depot), verifies that there is a chosen schedule,
         and that the chosen schedule is one of that customer's candidate schedules.
      2. For each day:
           - Verifies that the number of tours does not exceed the available vehicles for that day.
           - Checks that every customer whose chosen schedule requires service is visited exactly once.
      3. Each tour must:
           - Start at the depot (id 0) and end at the depot (id 0).
           - Not include any depot visit in the middle (the depot may appear only as the first and the last vertex).
           - Not visit the same customer more than once.
      4. Each tour must satisfy the capacity constraint: the total customer demand on the tour does not exceed vehicle_capacity.
      5. Finally, the evaluator computes the total tour length (using Euclidean distance) over all days.

    Returns:
      A numeric value representing the total tour length computed from the solution.

    Raises an error if any constraint is violated.
    """
    import math

    depot = kwargs["depot"]
    customers = kwargs["customers"]
    vehicles_per_day = kwargs["vehicles_per_day"]
    vehicle_capacity = kwargs["vehicle_capacity"]
    period_length = kwargs["period_length"]

    # Build a lookup table for customers by id.
    customer_lookup = {cust["id"]: cust for cust in customers}

    # Validate the selected schedules.
    selected_schedules = kwargs.get("selected_schedules")
    if not isinstance(selected_schedules, dict):
        raise ValueError("Solution must include a dictionary 'selected_schedules'.")

    # Ensure that every customer (except the depot) has a selected schedule.
    for cust in customers:
        # Assuming depot has id 0.
        if cust["id"] == 0:
            continue
        if cust["id"] not in selected_schedules:
            raise ValueError(f"Missing selected schedule for customer {cust['id']}.")

    # Now validate each provided schedule.
    for cid, sel_sched in selected_schedules.items():
        cust = customer_lookup.get(cid)
        if cust is None:
            raise ValueError(f"Customer id {cid} in selected_schedules not found in customer list.")
        if sel_sched not in cust["schedules"]:
            raise ValueError(
                f"Selected schedule {sel_sched} for customer {cid} is not among candidate schedules {cust['schedules']}.")
        if len(sel_sched) != period_length:
            raise ValueError(f"Selected schedule for customer {cid} does not match period_length {period_length}.")

    # Process tours for each day.
    tours = kwargs.get("tours")
    if not isinstance(tours, dict):
        raise ValueError("Solution must include a dictionary 'tours'.")

    total_length = 0.0

    def euclidean(a, b):
        return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2)

    # Evaluate each day.
    for day in range(1, period_length + 1):
        # Validate the number of tours does not exceed the available vehicles.
        tours_day = tours.get(day, [])
        vehicles_available = vehicles_per_day[day - 1]
        if len(tours_day) > vehicles_available:
            raise ValueError(
                f"On day {day}: Number of tours ({len(tours_day)}) exceeds available vehicles ({vehicles_available}).")

        # Determine all customers that should receive service today.
        expected_customers = set()
        for cust in customers:
            if cust["id"] == 0:
                continue
            sched = selected_schedules.get(cust["id"])
            if sched is not None and sched[day - 1] == 1:
                expected_customers.add(cust["id"])

        visited_today = []
        for tour in tours_day:
            # A valid tour must have at least depot, one customer, and depot again.
            if len(tour) < 3:
                raise ValueError(f"Tour {tour} on day {day} is too short.")
            # Check that the tour starts and ends with the depot.
            if tour[0] != 0 or tour[-1] != 0:
                raise ValueError(f"Tour {tour} on day {day} must start and end at the depot (id 0).")
            # Ensure no depot visits occur in the middle.
            if 0 in tour[1:-1]:
                raise ValueError(f"Tour {tour} on day {day} contains an extra depot visit in the middle.")

            seen_in_tour = set()
            # Process customer visits in the tour (excluding depot at the beginning and end).
            for vid in tour[1:-1]:
                if vid in seen_in_tour:
                    raise ValueError(f"Tour on day {day} visits customer {vid} more than once.")
                seen_in_tour.add(vid)
                visited_today.append(vid)

            # Check the capacity constraint for the tour.
            capacity_used = sum(customer_lookup[vid]["demand"] for vid in tour[1:-1])
            if capacity_used > vehicle_capacity:
                raise ValueError(
                    f"Tour on day {day} exceeds capacity: used {capacity_used}, capacity is {vehicle_capacity}.")

            # Compute the tour's travel distance.
            tour_length = 0.0
            prev = depot
            for vid in tour[1:]:
                curr = depot if vid == 0 else customer_lookup.get(vid)
                if curr is None:
                    raise ValueError(f"Customer id {vid} in tour on day {day} not found.")
                tour_length += euclidean(prev, curr)
                prev = curr
            total_length += tour_length

        # Ensure that the visited customers exactly match those expected for the day.
        if set(visited_today) != expected_customers:
            missing = expected_customers - set(visited_today)
            extra = set(visited_today) - expected_customers
            err_msg = f"On day {day}: "
            if missing:
                # Only showing a sample of missing customers
                err_msg += f"Missing visits for customers such as {list(missing)[:10]}. "
            if extra:
                err_msg += f"Extra visits for customers {list(extra)}."
            raise ValueError(err_msg)

    return total_length


def norm_score(results):
    optimal_scores = {
        "prvp1.txt": [547.9],
        "prvp2.txt": [1487.6],
        "prvp3.txt": [550.1],
        "prvp4.txt": [872.3],
        "prvp5.txt": [2207.9],
        "prvp6.txt": [965.7],
        "prvp7.txt": [839.2],
        "prvp8.txt": [2294.2],
        "prvp9.txt": [925.0],
        "prvp10.txt": [1819.2],
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
    dev = {'prvp1.txt': [], 'prvp3.txt': [], 'prvp5.txt': [],
           'prvp7.txt': [], 'prvp9.txt': []}

    return dev
