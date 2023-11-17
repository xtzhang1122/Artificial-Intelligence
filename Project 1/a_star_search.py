import queue
import numpy as np
from search_problems import Node, GridSearchProblem, get_random_grid_problem
import time

def trace(node, target_state):
    path = [node.state]
    node_current = node
    while node_current != target_state:
        node_current = node_current.parent
        # insert the parent at the front (shorter runtime than .reverse)
        path.insert(0, node_current.state)
    
    return path

def a_star_search(problem):
    """
    Uses the A* algorithm to solve an instance of GridSearchProblem. Use the methods of GridSearchProblem along with
    structures and functions from the allowed imports (see above) to implement A*.

    :param problem: an instance of GridSearchProblem to solve
    :return: path: a list of states (ints) describing the path from problem.init_state to problem.goal_state[0]
             num_nodes_expanded: number of nodes expanded by your search
             max_frontier_size: maximum frontier size during search
    """
    
    num_nodes_expanded = 0
    max_frontier_size = 0
    path = []

    # frontier_state: a priority queue that store the (node.path_cost, node)
        # using .get() for frontier_state ensures that we get the node with least cost
    # frontier_state_cost: a dictionary that keeps track of the cost of the nodes
    # f: a set of frontiers
    # open_list: a set of explored nodes
    x, y = problem.get_position(problem.init_state)
    start = Node(parent=None, state=problem.init_state, action=problem.get_state(x, y), path_cost=problem.manhattan_heuristic(problem.init_state, problem.goal_states[0]))
    frontier_state = queue.PriorityQueue()
    frontier_state.put((start.path_cost, start))
    frontier_state_cost = {start.state: 0}
    f = set()
    f.add(start.state)

    open_list = set()

    while frontier_state.empty() == False:
        # get the minimum node -> frontier
        frontier = frontier_state.get()
        current_node = frontier[1]
        
        # remove this node from f if it is in it
        try:
            f.remove(current_node.state)
        except:
            pass

        if problem.goal_test(current_node.state):
            path = trace(current_node, start)
            return path, num_nodes_expanded, max_frontier_size
       
        open_list.add(current_node.state)

        for action in problem.get_actions(current_node.state):
            next_node = problem.get_child_node(current_node, action)
            
            # if next_node has not been explored
            if next_node.state not in open_list and next_node.state not in f:
                next_node.path_cost = current_node.path_cost - problem.heuristic(current_node.state) + problem.heuristic(next_node.state) + 1
                frontier_state_cost[next_node.state] = frontier_state_cost[current_node.state] + 1
                frontier_state.put((next_node.path_cost, next_node))
                f.add(next_node.state)

            # if next_node has been explored but this iteration is more optimal
            elif next_node.state in f and frontier_state_cost[next_node.state] > frontier_state_cost[current_node.state] + 1:
                next_node.path_cost = current_node.path_cost - problem.heuristic(current_node.state) + problem.heuristic(next_node.state) + 1
                frontier_state_cost[next_node.state] = frontier_state_cost[current_node.state] + 1
                frontier_state.put((next_node.path_cost, next_node))
                f.add(next_node.state)
                                
        max_frontier_size = max(max_frontier_size, frontier_state.qsize())
        num_nodes_expanded += 1
    return path, num_nodes_expanded, max_frontier_size


def search_phase_transition():
    """
    Simply fill in the prob. of occupancy values for the 'phase transition' and peak nodes expanded within 0.05. You do
    NOT need to submit your code that determines the values here: that should be computed on your own machine. Simply
    fill in the values!

    :return: tuple containing (transition_start_probability, transition_end_probability, peak_probability)
    """
    ####
    #   REPLACE THESE VALUES
    ####
    transition_start_probability = 0.35
    transition_end_probability = 0.5
    peak_nodes_expanded_probability = 0.35
    return transition_start_probability, transition_end_probability, peak_nodes_expanded_probability


if __name__ == '__main__':
    # Test your code here!
    # Create a random instance of GridSearchProblem
    p_occ = 0.25
    M = 500
    N = 500
    problem = get_random_grid_problem(p_occ, M, N)
    # Solve it
    s = time.time()
    path, num_nodes_expanded, max_frontier_size = a_star_search(problem)
    d = time.time()
    print(d-s)
    # Check the result
    correct = problem.check_solution(path)
   # print(path)
    print("Solution is correct: {:}".format(correct))
    # Plot the result
    problem.plot_solution(path)

    # Experiment and compare with BFS