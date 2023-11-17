from collections import deque
from platform import node
import numpy as np
from search_problems import Node, GraphSearchProblem

def trace(node, target_state):
    path = [node.state]
    node_current = node
    while node_current.state != target_state:
        node_current = node_current.parent
        path.insert(0, node_current.state)
    
    return path


def bidirectional_search(problem):
    """
        Implement a bidirectional search algorithm that takes instances of SimpleSearchProblem (or its derived
        classes) and provides a valid and optimal path from the initial state to the goal state.

        :param problem: instance of SimpleSearchProblem
        :return: path: a list of states (ints) describing the path from problem.init_state to problem.goal_state[0]
                 num_nodes_expanded: number of nodes expanded by your search
                 max_frontier1_size: maximum frontier1 size during search
        """
    
    max_frontier_size = 0
    num_nodes_expanded = 0
    path = []

    start = Node(parent=None, state=problem.init_state, action=problem.get_actions(problem.init_state), path_cost=0)
    forward = {problem.init_state: start}

    for g in problem.goal_states:
        goal = Node(parent=None, state=g, action=problem.get_actions(g), path_cost=0)
        backward = {g: goal}

    explored_f = set()
    explored_b = set()
    cost = np.inf

    while start.path_cost + goal.path_cost < cost:
        
        for key in explored_f:
            if key in explored_b:
                path = trace(forward[key], problem.init_state)
                path2 = trace(backward[key], problem.goal_states[0])
                path2.reverse()
                path.extend(path2[1:])
                return path, num_nodes_expanded, max_frontier_size
        
        frontier1 = sorted(forward.items(), key=lambda x:x[1])
        frontier2 = sorted(backward.items(), key=lambda x:x[1])
        node1 = forward[frontier1[0][0]]
        node2 = backward[frontier2[0][0]]
        
        if node1 > node2: # backward
            current_node = node2
            backward.pop(current_node.state)
            direction = 1
          
        else: # forward
            current_node = node1
            forward.pop(current_node.state)
            direction = 0
       
        for action in problem.get_actions(current_node.state):
            next_node = problem.get_child_node(current_node, action)

            if direction == 0:
                if next_node.state not in explored_f and next_node.state not in forward: # if this node has not been explored
                    forward[next_node.state] = next_node
                    explored_f.add(next_node.state)

                    # there is an optimal cost in the forward direction
                    if next_node.path_cost + current_node.path_cost < forward[next_node.state].path_cost:
                        forward[next_node.state].path_cost = next_node.path_cost + current_node.path_cost
                    
                    # there is an optimal cost in bi-direction
                    if next_node.state in explored_b and forward[next_node.state].path_cost + backward[next_node.state].path_cost < cost:
                        cost = forward[next_node.state].path_cost + backward[next_node.state].path_cost

            else:
                if next_node.state not in explored_b and next_node.state not in backward: # if this node has not been explored
                    backward[next_node.state] = next_node
                    explored_b.add(next_node.state)

                     # there is an optimal cost in the forward direction
                    if next_node.path_cost + current_node.path_cost < backward[next_node.state].path_cost:
                        backward[next_node.state].path_cost = next_node.path_cost + current_node.path_cost

                    # there is an optimal cost in bi-direction
                    if next_node.state in explored_f and forward[next_node.state].path_cost + backward[next_node.state].path_cost < cost:
                        cost = forward[next_node.state].path_cost + backward[next_node.state].path_cost

        max_frontier_size = max(max_frontier_size, len(forward), len(backward))
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
    path, num_nodes_expanded, max_frontier1_size = bidirectional_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)

    # Use stanford_large_network_facebook_combined.txt to make your own test instances
    E = np.loadtxt('C:/Users/xtzha/Desktop/ROB311/Project 1/project_01/rob311_winter_2022_project_01_handout/stanford_large_network_facebook_combined.txt', dtype=int)
    V = np.unique(E)
    goal_states = [349]
    init_state = 0
    problem = GraphSearchProblem(goal_states, init_state, V, E)
    path, num_nodes_expanded, max_frontier1_size = bidirectional_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)

    # Be sure to compare with breadth_first_search!