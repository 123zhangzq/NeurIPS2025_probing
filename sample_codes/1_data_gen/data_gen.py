import numpy as np
import gurobipy as gp
from gurobipy import GRB
import itertools
import pickle


# para
num_nodes = 100
num_instances = 10000
seed_start = 1

# output results
data = []


it_seed = seed_start - 1
while True:
    it_seed += 1
    if len(data) >= num_instances:
        break
        
    print("**********************number of current instances ", str(len(data)),"**********************")
    print("**********************SEED ", str(it_seed),"**********************")
    
    instance_i = {}
    
    # Set the seed for reproducibility
    np.random.seed(it_seed)
    n = num_nodes

    # Generate 20 random points in the range (0, 1) for x and y (not including 0 and 1)
    points = np.random.uniform(0, 1, size=(n, 2))
    
    # Gurobi code to get the optimal solution
    # Calculate the distance matrix
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            dist[i, j] = dist[j, i] = np.linalg.norm(points[i] - points[j])

    # Create a Gurobi model
    m = gp.Model("TSP")

    # Add binary variables for each pair of points
    x = m.addVars(n, n, vtype=GRB.BINARY, name="x")

    # Set the objective function to minimize the total distance
    m.setObjective(gp.quicksum(x[i, j] * dist[i, j] for i in range(n) for j in range(n)), GRB.MINIMIZE)

    # Add degree-2 constraint (each node has exactly one entering and one leaving edge)
    m.addConstrs(gp.quicksum(x[i, j] for j in range(n) if i != j) == 1 for i in range(n))
    m.addConstrs(gp.quicksum(x[j, i] for j in range(n) if i != j) == 1 for i in range(n))

    # Eliminate sub-tours with lazy constraints
    def subtour_elimination(model, where):
        if where == GRB.Callback.MIPSOL:
            # Get the solution
            sol = model.cbGetSolution(model._vars)

            # Select edges that are part of the tour
            selected = [(i, j) for i in range(n) for j in range(n) if sol[i, j] > 0.5]

            # Find the smallest subtour
            unvisited = list(range(n))
            tour = []
            while unvisited:
                this_tour = []
                neighbors = unvisited
                while neighbors:
                    current = neighbors[0]
                    this_tour.append(current)
                    unvisited.remove(current)
                    neighbors = [j for i, j in selected if i == current and j in unvisited]
                if len(this_tour) < len(tour) or not tour:
                    tour = this_tour

            # Add a subtour elimination constraint
            if len(tour) < n:
                model.cbLazy(gp.quicksum(x[i, j] for i in tour for j in tour if i != j) <= len(tour) - 1)

    # Optimize the model with the subtour elimination callback
    m._vars = x
    m.Params.LazyConstraints = 1
    m.optimize(subtour_elimination)

    # Extract the optimal tour
    if m.status == gp.GRB.OPTIMAL:
        solution = m.getAttr('X', x)
        tour = []
        for i in range(n):
            for j in range(n):
                if solution[i, j] > 0.5:
                    tour.append((i, j))
    else:
        print("Not optimal!!!!!!!!!!!!")
        continue

    # Function to get the tour sequence
    def find_tour(tour):
        sequence = [0]
        while len(sequence) < n:
            for i, j in tour:
                if i == sequence[-1] and j not in sequence:
                    sequence.append(j)
                elif j == sequence[-1] and i not in sequence:
                    sequence.append(i)
        return sequence

    # Get the sequence of the tour
    optimal_tour = find_tour(tour)
    opt_value = m.getObjective().getValue()
    
    
    # greedy
    # Precompute the distance matrix
    def compute_distance_matrix(points):
        n = len(points)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist_matrix[i, j] = np.linalg.norm(points[i] - points[j])
        return dist_matrix

    # Implementing greedy TSP starting from each of the points
    def greedy_tsp_optimized(points, start_point, dist_matrix):
        n = len(points)
        visited = [False] * n
        current_point = start_point
        tour = [current_point]
        visited[current_point] = True

        while len(tour) < n:
            next_point = None
            min_dist = float('inf')

            for i in range(n):
                if not visited[i]:
                    distance = dist_matrix[current_point, i]
                    if distance < min_dist:
                        min_dist = distance
                        next_point = i

            tour.append(next_point)
            visited[next_point] = True
            current_point = next_point

        # Return to the starting point
        tour.append(tour[0])

        # Calculate the total distance of this tour using the precomputed distance matrix
        total_distance = sum(dist_matrix[tour[i], tour[i + 1]] for i in range(n))

        return tour, total_distance


    # Initialize best_distance and best_start_point
    best_distance = float('inf')
    best_start_point = None
    dist_matrix = compute_distance_matrix(points)

    for i in range(n):
        tour, distance = greedy_tsp_optimized(points, i, dist_matrix)
        if distance < best_distance:
            best_distance = distance
            best_tour = tour
            best_start_point = i

    # Reorder the best tour to start from point 0
    best_tour_index = best_tour.index(0)
    best_tour = best_tour[best_tour_index:] + best_tour[1:best_tour_index + 1]
    
    
    # ignore the greedy is optimal
    if abs(opt_value - best_distance) <= 1e-6:
        continue
    
    
    # get the node where greedy is wrong
    opt_route = optimal_tour+[0]
    greedy_route = best_tour
    start_node = best_start_point


    def compare_routes(opt_route, greedy_route, start_node):
        # Step 1: Generate new_greedy_route starting from start_node
        start_idx_greedy = greedy_route.index(start_node)
        new_greedy_route = greedy_route[start_idx_greedy:] + greedy_route[1:start_idx_greedy]

        # Step 2: Find the position of start_node in opt_route and check both sides for the second element in new_greedy_route
        start_idx_opt = opt_route.index(start_node)
        second_node = new_greedy_route[1]

        if opt_route[(start_idx_opt - 1) % len(opt_route)] == second_node:
            # Construct new_opt_route by going backward
            new_opt_route = opt_route[start_idx_opt::-1] + opt_route[:start_idx_opt][::-1]
        elif opt_route[(start_idx_opt + 1) % len(opt_route)] == second_node:
            # Construct new_opt_route by going forward
            new_opt_route = opt_route[start_idx_opt:] + opt_route[:start_idx_opt]
        else:
            # If second_node is not next to start_node in opt_route, return None
            return None

        # Step 3: Compare new_greedy_route and new_opt_route, find the last same element and the first different element
        last_same_node, greedy_node, opt_node = None, None, None
        for i in range(min(len(new_greedy_route), len(new_opt_route))):
            if new_greedy_route[i] == new_opt_route[i]:
                last_same_node = new_greedy_route[i]
            else:
                greedy_node = new_greedy_route[i]
                opt_node = new_opt_route[i]
                break

        # Step 4: Return the results
        return last_same_node, greedy_node, opt_node


    result = compare_routes(opt_route, greedy_route, start_node)
    if result == None:
        continue
    
    
    # create the data of the current instance
    instance_i['ran_seed'] = it_seed
    instance_i['nodes_coor'] = points.tolist()
    instance_i['opt_route'] = optimal_tour
    instance_i['opt_sol'] = opt_value
    instance_i['greedy_route'] = best_tour
    instance_i['greedy_sol'] = float(best_distance)
    instance_i['node_probe_data'] = result
    instance_i['best_start_point'] = best_start_point
    data.append(instance_i)
    
    
    
    
    
    
    
with open('data.pkl', 'wb') as file:
    pickle.dump(data, file)
