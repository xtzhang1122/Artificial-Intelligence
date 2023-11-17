from abc import ABC, abstractmethod
import itertools
import numpy as np

from numpy import matrix
from sklearn import random_projection


class SingleMoveGamePlayer(ABC):
    """
    Abstract base class for a symmetric, zero-sum single move game player.
    """
    def __init__(self, game_matrix: np.ndarray):
        self.game_matrix = game_matrix
        self.n_moves = game_matrix.shape[0]
        super().__init__()

    @abstractmethod
    def make_move(self) -> int:
        pass


class IteratedGamePlayer(SingleMoveGamePlayer):
    """
    Abstract base class for a player of an iterated symmetric, zero-sum single move game.
    """
    def __init__(self, game_matrix: np.ndarray):
        super(IteratedGamePlayer, self).__init__(game_matrix)

    @abstractmethod
    def make_move(self) -> int:
        pass

    @abstractmethod
    def update_results(self, my_move, other_move):
        """
        This method is called after each round is played
        :param my_move: the move this agent played in the round that just finished
        :param other_move:
        :return:
        """
        pass

    @abstractmethod
    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        pass


class UniformPlayer(IteratedGamePlayer):
    def __init__(self, game_matrix: np.ndarray):
        super(UniformPlayer, self).__init__(game_matrix)

    def make_move(self) -> int:
        """

        :return:
        """
        return np.random.randint(0, self.n_moves)

    def update_results(self, my_move, other_move):
        """
        The UniformPlayer player does not use prior rounds' results during iterated games.
        :param my_move:
        :param other_move:
        :return:
        """
        pass

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        pass


class FirstMovePlayer(IteratedGamePlayer):
    def __init__(self, game_matrix: np.ndarray):
        super(FirstMovePlayer, self).__init__(game_matrix)

    def make_move(self) -> int:
        """
        Always chooses the first move
        :return:
        """
        return 0

    def update_results(self, my_move, other_move):
        """
        The FirstMovePlayer player does not use prior rounds' results during iterated games.
        :param my_move:
        :param other_move:
        :return:
        """
        pass

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        pass


class CopycatPlayer(IteratedGamePlayer):
    def __init__(self, game_matrix: np.ndarray):
        super(CopycatPlayer, self).__init__(game_matrix)
        self.last_move = np.random.randint(self.n_moves)

    def make_move(self) -> int:
        """
        Always copies the last move played
        :return:
        """
        return self.last_move

    def update_results(self, my_move, other_move):
        """
        The CopyCat player simply remembers the opponent's last move.
        :param my_move:
        :param other_move:
        :return:
        """
        self.last_move = other_move

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        self.last_move = np.random.randint(self.n_moves)


def play_game(player1, player2, game_matrix: np.ndarray, N: int = 1000) -> (int, int):
    """

    :param player1: instance of an IteratedGamePlayer subclass for player 1
    :param player2: instance of an IteratedGamePlayer subclass for player 2
    :param game_matrix: square payoff matrix
    :param N: number of rounds of the game to be played
    :return: tuple containing player1's score and player2's score
    """
    p1_score = 0.0
    p2_score = 0.0
    n_moves = game_matrix.shape[0]
    legal_moves = set(range(n_moves))
    
    for idx in range(N):
        move1 = player1.make_move()
        move2 = player2.make_move()
        if move1 not in legal_moves:
            print("WARNING: Player1 made an illegal move: {:}".format(move1))
            if move2 not in legal_moves:
                print("WARNING: Player2 made an illegal move: {:}".format(move2))
            else:
                p2_score += np.max(game_matrix)
                p1_score -= np.max(game_matrix)
            continue
        elif move2 not in legal_moves:
            print("WARNING: Player2 made an illegal move: {:}".format(move2))
            p1_score += np.max(game_matrix)
            p2_score -= np.max(game_matrix)
            continue
        player1.update_results(move1, move2)
        player2.update_results(move2, move1)

        p1_score += game_matrix[move1, move2]
        p2_score += game_matrix[move2, move1]

    return p1_score, p2_score


class StudentAgent(IteratedGamePlayer):
    """
    My agent plays the game differently based on the number of rounds in a game. 
    In general, it tries to capture the pattern of the oppoent and should perform better as the number of rounds increases. 

        During the first 2 round, as there is no pattern, the agent will execute the action that ideally return the most amount of awards.
    It achieves so by returning the index of the largest element in the game/reward matrix.

        From round 3 to 10, as the history starts builiding up, the agent traces the probabilties of winning with each action and returns the Nash Equilibrium.
        
        Through these rounds, the agent is also updating a 3-dimensional 3 x 3 matrix. 
        
        This matrix can be viewed as a simple 3 step Markov deicision process where the first index represents my history,
            the second index represents my opponent's history and the third index represents my move this round.
        Once this value in the matrix is found using these 3 indexes, the matrix is updated using the following formula:
        U(s_t, a_t) = r * U(s_t, a_t) + R(s_t) + a * U(s'_t, a'_t)
        In this equation, U(s_t, a_t) is the history state, U(s'_t, a'_t) is the future state; r is decay rate (0.9) and a is a small estimate for the future (0.01).
        
        Therefore, when the number of rounds exceeds 10, the agent will use its knowledge from the previous 10 rounds and 
            execute the action that results in the optimal utility.
        Every time a round is finished, the agent will continue to update this matrix so it remembers what happened in this past round and change its actions
            to maximize utility and rewards.
    """
    def __init__(self, game_matrix: np.ndarray):
        """
        Initialize your game playing agent. here
        :param game_matrix: square payoff matrix for the game being played.
        """
        super(StudentAgent, self).__init__(game_matrix)

        self.decay_rate = 0.9
        self.simple_matrix = np.ones((3, 3, 3))
        # self.complex_matrix = np.ones((3, 3, 3))
        self.my_history = np.random.randint(3)
        self.other_history = np.random.randint(3)
        self.my_last_move = 0
        self.other_last_move = 0
        self.reward = game_matrix
        self.turn = 0

        
    def make_move(self) -> int:
        """
        Play your move based on previous moves or whatever reasoning you want.
        :return: an int in (0, ..., n_moves-1) representing your move
        """
        
        if self.turn <= 1:
            optimal = np.unravel_index(np.argmax(self.reward, axis=None), self.reward.shape)
            return optimal[0]
        elif self.turn <= 10:
            paper_win = self.reward[1][0]
            scissor_win = self.reward[2][1]
            rock_win = self.reward[0][2]
            total = paper_win + scissor_win + rock_win

            nash_equilibrium = [paper_win / total, scissor_win / total, rock_win / total]
            return np.random.choice([0, 1, 2], p = nash_equilibrium)
        else:
            optimal_sindex = np.argmax(self.simple_matrix[self.my_last_move][self.other_last_move][:])
            # optimal_cindex = np.argmax(self.complex_matrix[self.my_last_move][self.other_last_move][:])

            # if self.simple_matrix[self.my_last_move][self.other_last_move][optimal_sindex] > self.complex_matrix[self.my_last_move][self.other_last_move][optimal_cindex]:
            return optimal_sindex
            # else:
                # return optimal_cindex

    def update_results(self, my_move, other_move):
        """
        Update your agent based on the round that was just played.
        :param my_move:
        :param other_move:
        :return: nothing
        """

        self.my_history = self.my_last_move
        self.other_history = self.other_last_move
        
        if self.turn > 1:
            self.simple_matrix[self.my_last_move][self.other_last_move][my_move] = self.simple_matrix[self.my_last_move][self.other_last_move][my_move]  * self.decay_rate + self.reward[my_move][other_move]  + \
                 + 0.01 * np.max(self.simple_matrix[my_move][other_move][:])

            # self.complex_matrix[self.my_last_move][self.other_last_move][my_move] = self.complex_matrix[self.my_last_move][self.other_last_move][my_move] + \
            #     0.5 * (self.reward[my_move][other_move] + 0.5 * np.max(self.complex_matrix[my_move][other_move][:]) - self.complex_matrix[self.my_last_move][self.other_last_move][my_move])
        
        self.my_last_move = my_move
        self.other_last_move = other_move    
        self.turn += 1

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.).
        :return: nothing
        """
        self.simple_matrix = np.ones((3, 3, 3))
        # self.complex_matrix = np.ones((3, 3, 3))
        self.my_history = np.random.randint(3)
        self.other_history = np.random.randint(3)
        self.my_last_move = 0
        self.other_last_move = 0
        self.turn = 0

if __name__ == '__main__':
    """
    Simple test on standard rock-paper-scissors
    The game matrix's row (first index) is indexed by player 1 (P1)'s move (i.e., your move)
    The game matrix's column (second index) is indexed by player 2 (P2)'s move (i.e., the opponent's move)
    Thus, for example, game_matrix[0, 1] represents the score for P1 when P1 plays rock and P2 plays paper: -1.0 
    because rock loses to paper.
    """
    game_matrix = np.array([[0.0, -1.0, 1.0],
                            [1.0, 0.0, -1.0],
                            [-1.0, 1.0, 0.0]])
    uniform_player = UniformPlayer(game_matrix)
    first_move_player = FirstMovePlayer(game_matrix)
    copy_player = CopycatPlayer(game_matrix)
    # uniform_score, first_move_score = play_game(uniform_player, first_move_player, game_matrix)

    # print("Uniform player's score: {:}".format(uniform_score))
    # print("First-move player's score: {:}".format(first_move_score))

    # Now try your agent
    student_player = StudentAgent(game_matrix)
    student_score, first_move_score = play_game(student_player, first_move_player, game_matrix)

    print("Your player's score: {:}".format(student_score))
    print("First-move player's score: {:}".format(first_move_score))

    student_score, uniform_score = play_game(student_player, uniform_player, game_matrix)

    print("Your player's score: {:}".format(student_score))
    print("Uniform player's score: {:}".format(uniform_score))

    student_score, copy_player = play_game(student_player, copy_player, game_matrix)

    print("Your player's score: {:}".format(student_score))
    print("Copy player's score: {:}".format(copy_player))