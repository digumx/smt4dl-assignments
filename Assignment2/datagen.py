"""

File to generate data for the tic-tac-toe game for training and testing the network. Classification
of a generated valid config and move is done via z3. The board state is encoded as a 27-size bit
vector, three bits for each of the nine positions, the bits represent the board being empty, marked
by 1, or by 2 respectively. A move is represented by a 9-size bit vector, representing which board
position is being marked.

"""

import z3
import random


def encode_has_winning_move(z3_board, player):
    """
    Given a list of 27 z3 BoolSort constants encoding a board state as described above, return a z3
    boolean expression encoding the condition that the given player has a winning move from this
    board state.
    """

    ret = False

    # Check if there is a possible victory for player by completing a row or a column
    for i in range(3):
        for j in range(3):
            # Check if ith row can be completed in jth position
            ret = z3.Or(ret, z3.And([z3_board[9*i + 3*k + (0 if j == k else player)] for k in range(3)]))
            # Check if ith column can be completed in jth postion
            ret = z3.Or(ret, z3.And([z3_board[3*i + 9*k + (0 if j == k else player)] for k in range(3)]))

    # Check diagonals
    for i in range(3):
        # Can the primary diagonal be completed in ith position
        ret = z3.Or(ret, z3.And([z3_board[3*k + 9*k + (0 if i == k else player)] for k in range(3)]))
        # Can the secondary diagonal be completed in the ith position
        ret = z3.Or(ret, z3.And([z3_board[3*(2-k) + 9*k + (0 if i == k else player)] for k in range(3)]))

    return ret


def encode_has_won(z3_board, player):
    """
    Given a list of 27 z3 BoolSort constants encoding a board state as described above, return a z3
    boolean expression encoding the condition that the given player has won the game
    """

    ret = False

    # Check if a row or a column is complete
    for i in range(3):
        # Check ith row
        ret = z3.Or(ret, z3.And([z3_board[9*i + 3*j + player] for j in range(3)]))
        # Check column
        ret = z3.Or(ret, z3.And([z3_board[3*i + 9*j + player] for j in range(3)]))

    # Check primary diagonal
    ret = z3.Or(ret, z3.And([z3_board[3*i + 9*i + player] for i in range(3)]))
    # Check other diagonal
    ret = z3.Or(ret, z3.And([z3_board[3*(2-i) + 9*i + player] for i in range(3)]))

    return ret


def encode_move(z3_board_from, z3_board_to, z3_move, player):
    """
    Return a boolean z3 expression that encodes the condition that `z3_board_to` is a board
    representation that is obtained from the board representation `z3_board_from` by performing the
    move represented by `z3_move` for the player `player`.
    """

    ret = True

    for i in range(9):
        ret = z3.And(ret, z3.If(z3_move[i], 
                        z3.And([z3_board_to[3*i + j] if j==player else z3.Not(z3_board_to[3*i + j])
                                    for j in range(3)]),
                        z3.And([z3_board_from[3*i + j] == z3_board_to[3*i + j] 
                                    for j in range(3)])))
    return ret 


def check_move(move):
    """
    Given a move, use Z3 to check if it is good or bad. The move is encodeed as a 36-size bit vector
    as described above.
    
    """

    # Initialize and introduce z3 constants
    solver = z3.Solver()
    z3_board        = [z3.Const("board_state_" + str(i), z3.BoolSort()) for i in range(27)]
    z3_board_res    = [z3.Const("res_board_state_" + str(i), z3.BoolSort()) for i in range(27)]

    # Add constraints for input
    for z3_const, cond in zip(z3_board, move[:27]):
        solver.add(z3_const if cond else z3.Not(z3_const))

    # Add constraints for the fact that z3_board_res is obtained via move on z3_board
    solver.add(encode_move(z3_board, z3_board_res, move[27:], 1))

    # If player one can win, move must be winnig
    solver.add(z3.Implies(encode_has_winning_move(z3_board, 1), encode_has_won(z3_board_res, 1)))
    
    # If player one cannot win, player one should not win in next round
    solver.add(z3.Implies(z3.Not(encode_has_winning_move(z3_board, 1)),
                            z3.Not(encode_has_winning_move(z3_board_res, 2))))
    # Finally, check sat
    return solver.check() == z3.sat


def gen_data(num):
    """
    Generates `num` counts of moves encoded as 36-size bit vectors and corresponding labels as a
    good or bad move. This only generates valid board positions, with valid as defined in the
    assignment question. It returns a list of tuples, with each tuple represing an encoding of the
    move and corresponding label, the label being True for a good move, and false otherwise.
    """

    data = []
    
    for n in range(num):
        print("Generating point", n, "of", num, end="\r")

        # Pick number of moves
        m = random.randrange(2, 4)
        
        # Permute markings into board positions
        raw_board = [0]*(9 - 2*m)
        raw_board.extend([1]*m)
        raw_board.extend([2]*m)
        random.shuffle(raw_board)

        # Chose move
        raw_pick = random.choice(list(filter(lambda i: raw_board[i] == 0, range(9))))
       
        # Encode board and move
        move = []
        pick = []
        for b, i in zip(raw_board, range(9)):
            move.extend([k==b for k in range(3)])
            pick.append(i==raw_pick)
        move.extend(pick)

        # Check if move is good and append to dataset
        data.append((move, check_move(move)))

    return data
