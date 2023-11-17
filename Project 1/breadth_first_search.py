from collections import deque
import numpy as np
from search_problems import Node, GraphSearchProblem

def trace(node, target_state):
    path = [node.state]
    node_current = node
    while node_current != target_state:
        node_current = node_current.parent
        path.insert(0, node_current.state)
    
    return path


def breadth_first_search(problem):
    """
    Implement a simple breadth-first search algorithm that takes instances of SimpleSearchProblem (or its derived
    classes) and provides a valid and optimal path from the initial state to the goal state. Useful for testing your
    bidirectional and A* search algorithms.

    :param problem: instance of SimpleSearchProblem
    :return: path: a list of states (ints) describing the path from problem.init_state to problem.goal_state[0]
             num_nodes_expanded: number of nodes expanded by your search
             max_frontier_size: maximum frontier size during search
    """
    
    num_nodes_expanded = 0
    max_frontier_size = 0
    path = []

    
    start = Node(parent=None, state=problem.init_state, action=problem.get_actions(problem.init_state), path_cost=0)
    frontier_state = {problem.init_state: start}
    explored = set()

    while len(frontier_state) > 0:
        # find the state with the shallowest depth
        frontier = sorted(frontier_state.items(), key=lambda x:x[1].path_cost)
        current_node = frontier_state[frontier[0][0]]
        frontier_state.pop(current_node.state)
    
        if problem.goal_test(current_node.state):
            path = trace(current_node, start)
            return path, num_nodes_expanded, max_frontier_size
       
        explored.add(current_node.state)

        for action in problem.get_actions(current_node.state):
            next_node = problem.get_child_node(current_node, action)
            
            if next_node.state not in explored or next_node.state not in frontier_state:
                if problem.goal_test(next_node.state):
                    path = trace(next_node, start)
                    return path, num_nodes_expanded, max_frontier_size
                frontier_state[next_node.state] = next_node

        max_frontier_size = max(max_frontier_size, len(frontier_state))
        num_nodes_expanded += 1
    return path, num_nodes_expanded, max_frontier_size


if __name__ == '__main__':
    # Simple example
    goal_states = [0]
    init_state = 9
    V = np.arange(0, 10)
    E = np.array([[0, 1],
                  [1, 2],
                  [2, 3],
                  [3, 4],
                  [4, 5],
                  [5, 6],
                  [6, 7],
                  [7, 8],
                  [8, 9],
                  [0, 6],
                  [1, 7],
                  [2, 5],
                  [9, 4]])
    problem = GraphSearchProblem(goal_states, init_state, V, E)
    path, num_nodes_expanded, max_frontier_size = breadth_first_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)

    # Use stanford_large_network_facebook_combined.txt to make your own test instances
    E = np.loadtxt('C:/Users/xtzha/Desktop/ROB311/Project 1/project_01/rob311_winter_2022_project_01_handout/stanford_large_network_facebook_combined.txt', dtype=int)
    V = np.unique(E)
    goal_states = [349]
    init_state = 0
    problem = GraphSearchProblem(goal_states, init_state, V, E)
    path, num_nodes_expanded, max_frontier_size = breadth_first_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)