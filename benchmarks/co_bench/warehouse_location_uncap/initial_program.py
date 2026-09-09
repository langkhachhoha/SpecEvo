# CO-Bench :: Uncapacitated warehouse location  (category: Facility location)
#
# The Uncapacitated Warehouse Location Problem aims to determine which warehouses to open and how to assign each customer entirely to an open warehouse in order to minimize the total cost. Given a set of potential warehouse locations, each with a fixed opening cost, and a set of customers, each with an associated assignment cost for being served by each warehouse, the objective is to select a subset of warehouses to open and assign every customer completely to one of these open warehouses. The optimization minimizes the sum of fixed warehouse opening costs and the customer assignment costs. Each customer must be assigned to exactly one warehouse; if any customer is left unassigned or assigned to more than one warehouse, the solution is considered infeasible.
#
# # Implement in Solve Function
#
# def solve(**kwargs):
#     """
#     Solves the Uncapacitated Warehouse Location Problem.
#
#     Input kwargs:
#       - m: Number of potential warehouses (int)
#       - n: Number of customers (int)
#       - warehouses: A list of dictionaries, each with keys:
#             'fixed_cost': Fixed cost for opening the warehouse.
#       - customers: A list of dictionaries, each with keys:
#             'costs': A list of floats representing the cost of assigning the entire customer to each warehouse.
#
#     Evaluation Metric:
#       The objective is to minimize the total cost, computed as:
#          (Sum of fixed costs for all open warehouses)
#        + (Sum of assignment costs for each customer assigned to a warehouse)
#       Each customer must be assigned entirely to exactly one open warehouse.
#       If a solution violates this constraint (i.e., a customer is unassigned or is assigned to more than one warehouse), then the solution is considered infeasible and no score is provided.
#
#     Returns:
#       A dictionary with the following keys:
#          'total_cost': (float) The computed objective value (cost) if the solution is feasible; otherwise, no score is provided.
#          'warehouse_open': (list of int) A list of m integers (0 or 1) indicating whether each warehouse is closed or open.
#          'assignments': (list of list of int) A 2D list (n x m) where each entry is 1 if customer i is assigned to warehouse j, and 0 otherwise.
#     """
#     ## placeholder. You do not need to write anything here.
#     return {
#         "total_cost": 0.0,
#         "warehouse_open": [0] * kwargs["m"],
#         "assignments": [[0] * kwargs["m"] for _ in range(kwargs["n"])]
#     }

# EVOLVE-BLOCK-START
def solve(**kwargs):
    m = kwargs["m"]
    n = kwargs["n"]
    warehouses = kwargs["warehouses"]
    customers = kwargs["customers"]
    warehouse_open = [0] * m
    assignments = [[0] * m for _ in range(n)]
    for j in range(n):
        costs = customers[j]["costs"]
        best = min(range(m), key=lambda i: costs[i])
        assignments[j][best] = 1
        warehouse_open[best] = 1
    total = sum(warehouses[i]["fixed_cost"] for i in range(m) if warehouse_open[i])
    total += sum(customers[j]["costs"][assignments[j].index(1)] for j in range(n))
    return {"total_cost": total, "warehouse_open": warehouse_open, "assignments": assignments}
# EVOLVE-BLOCK-END
