import numpy as np
from numpy import nonzero
from sympy import true
import time
### WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS

def min_conflicts_n_queens(initialization: list) -> (list, int):
    """
    Solve the N-queens problem with no conflicts (i.e. each row, column, and diagonal contains at most 1 queen).
    Given an initialization for the N-queens problem, which may contain conflicts, this function uses the min-conflicts
    heuristic(see AIMA, pg. 221) to produce a conflict-free solution.

    Be sure to break 'ties' (in terms of minimial conflicts produced by a placement in a row) randomly.
    You should have a hard limit of 1000 steps, as your algorithm should be able to find a solution in far fewer (this
    is assuming you implemented initialize_greedy_n_queens.py correctly).

    Return the solution and the number of steps taken as a tuple. You will only be graded on the solution, but the
    number of steps is useful for your debugging and learning. If this algorithm and your initialization algorithm are
    implemented correctly, you should only take an average of 50 steps for values of N up to 1e6.

    As usual, do not change the import statements at the top of the file. You may import your initialize_greedy_n_queens
    function for testing on your machine, but it will be removed on the autograder (our test script will import both of
    your functions).

    On failure to find a solution after 1000 steps, return the tuple ([], -1).

    :param initialization: numpy array of shape (N,) where the i-th entry is the row of the queen in the ith column (may
                           contain conflicts)

    :return: solution - numpy array of shape (N,) containing a-conflict free assignment of queens (i-th entry represents
    the row of the i-th column, indexed from 0 to N-1)
             num_steps - number of steps (i.e. reassignment of 1 queen's position) required to find the solution.
    """

    def conflict_count(board, num, row, column):

        for i in range(1, len(board)):

            if column - i >= 0: # left update
                board[column - i][row] += num

            if column + i < len(board): # right update
                board[column + i][row] += num

            if row - i >= 0 and column - i >= 0: # top left
                board[column - i][row - i] += num

            if row - i >= 0 and column + i < len(board): # top right
                board[column + i][row - i] += num

            if row + i < len(board) and column - i >= 0: # bottom left
                board[column - i][row + i] += num

            if row + i < len(board) and column + i < len(board): # bottom right
                board[column - i][row + i] += num

        return board

    N = len(initialization)
    solution = initialization.copy()
    num_steps = 0
    max_steps = 1000
    
    board = np.zeros([N, N])
    for column in range(len(solution)):
        board = conflict_count(board, 1, solution[column], column)

    for idx in range(max_steps):
        
        # if there are conflicts, there would not be a zero loop through rows of the board
        check = True
        for i in range(len(board)):
            if np.count_nonzero(board[:, i] == 0) == 0:
                check = False
                break
     
        if check == True:
            return solution, num_steps

        # find the conflict queens
        conflict_queens = []
        for i in range(len(solution)):
            if board[i][solution[i]] > 0:
                conflict_queens.append(i)
        
        # picked a random column that has conflict
        random_column = conflict_queens[np.random.randint(0, len(conflict_queens))] 
        
        # picked a random row with minimum conflict using the random column
        min_conflict = min(board[random_column])
        min_conflict_index = []
        for index, element in enumerate(board[random_column]):
            if element == min_conflict:
                min_conflict_index.append(index)
      
        random_row = min_conflict_index[np.random.randint(0, len(min_conflict_index))]

        # update
        board = conflict_count(board, -1, solution[random_column], random_column)
        solution[random_column] = random_row
        board = conflict_count(board, 1, solution[random_column], random_column)

        num_steps += 1

    return ([], -1) # failure


if __name__ == '__main__':
    # Test your code here!
    from initialize_greedy_n_queens import initialize_greedy_n_queens
    from support import plot_n_queens_solution

    N = 10
    # Use this after implementing initialize_greedy_n_queens.py
    assignment_initial = initialize_greedy_n_queens(N)
    # Plot the initial greedy assignment
    plot_n_queens_solution(assignment_initial)
    # s = time.time()
    assignment_solved, n_steps = min_conflicts_n_queens(assignment_initial)
    # d = time.time()
    # print(d - s)
    # Plot the solution produced by your algorithm
    plot_n_queens_solution(assignment_solved)
