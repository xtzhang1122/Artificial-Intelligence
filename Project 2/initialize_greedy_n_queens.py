import numpy as np
### WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS


def initialize_greedy_n_queens(N: int) -> list:
    """
    This function takes an integer N and produces an initial assignment that greedily (in terms of minimizing conflicts)
    assigns the row for each successive column. Note that if placing the i-th column's queen in multiple row positions j
    produces the same minimal number of conflicts, then you must break the tie RANDOMLY! This strongly affects the
    algorithm's performance!

    Example:
    Input N = 4 might produce greedy_init = np.array([0, 3, 1, 2]), which represents the following "chessboard":

     _ _ _ _
    |Q|_|_|_|
    |_|_|Q|_|
    |_|_|_|Q|
    |_|Q|_|_|

    which has one diagonal conflict between its two rightmost columns.

    You many only use numpy, which is imported as np, for this question. Access all functions needed via this name (np)
    as any additional import statements will be removed by the autograder.

    :param N: integer representing the size of the NxN chessboard
    :return: numpy array of shape (N,) containing an initial solution using greedy min-conflicts (this may contain
    conflicts). The i-th entry's value j represents the row  given as 0 <= j < N.
    """

    def update(board, row, col):

        for i in range(1, len(board)):
        
            if col + i < len(board): # right
                board[col + i][row] += 1
            
            if row - i >= 0 and col + i < len(board): # top right
                board[col + i][row - i] += 1
        
            if row + i < len(board) and col + i < len(board):  # bottom right
                board[col + i][row + i] += 1

        return board
    
    board = np.zeros([N, N])
    greedy_init = np.zeros(N, dtype=int)

    # First queen goes in a random spot
    greedy_init[0] = np.random.randint(0, N)
    board[0][greedy_init[0]] = np.inf

    board = update(board, greedy_init[0], 0)

    # randomly assigns the queen by looping through all the columns
    for i in range(1, N):
        
        min_conflict = min(board[i])
        min_conflict_index = []
        for index, element in enumerate(board[i]):
            if element == min_conflict:
                min_conflict_index.append(index)
      
        random_row = min_conflict_index[np.random.randint(0, len(min_conflict_index))]

        greedy_init[i] = random_row
        board[i][random_row] = np.inf
        
        board = update(board, random_row, i)

    return greedy_init


if __name__ == '__main__':
    # You can test your code here
    pass
