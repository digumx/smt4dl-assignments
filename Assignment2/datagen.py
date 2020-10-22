"""

File to generate data for the tic-tac-toe game for training and testing the network. Classification
of a generated valid config and move is done via z3. The board state is encoded as a 27-size bit
vector, three bits for each of the nine positions, the bits represent the board being empty, marked
by 1, or by 2 respectively. A move is represented by a 9-size bit vector, representing which board
position is being marked.

"""

import os.path
import random

import z3

from game_props import *


def gen_data(num):
    """
    Generates `num` counts of moves encoded as 36-size bit vectors and corresponding labels as a
    good or bad move. This only generates valid board positions, with valid as defined in the
    assignment question. It returns a list of tuples, with each tuple represing an encoding of the
    move and corresponding label, the label being True for a good move, and false otherwise. The
    function catches the generated data in "./out/data.val" and if the file exists and matches or
    exceeds the length of the data requested, the cached data is returned. Else, the missing data is
    generated
    """
    data = []
    
    # Read catched data
    if os.path.isfile("./out/data.val"):
        with open("./out/data.val") as f:
            data = eval(f.read())
            print("Read catched data of length", len(data))
            random.shuffle(data)
            print("Shuffled read data")
            if num <= len(data):
                print("More cached data than requested")
                return data[:num]
            num -= len(data)
            print("Generating", num, "more data points")
   
    # Generate extra data
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
