BLACK = 0
WHITE = 1

# Numeric codes for each BC piece
INIT_TO_CODE = {'p': 2, 'P': 3, 'c': 4, 'C': 5, 'l': 6, 'L': 7, 'i': 8, 'I': 9,
                'w': 10, 'W': 11, 'k': 12, 'K': 13, 'f': 14, 'F': 15, '-': 0}

# Text representation of each BC piece's numeric code.
CODE_TO_INIT = {0: '-', 2: 'p', 3: 'P', 4: 'c', 5: 'C', 6: 'l', 7: 'L', 8: 'i',
                9: 'I', 10: 'w', 11: 'W', 12: 'k', 13: 'K', 14: 'f', 15: 'F'}


def who(piece):
    """Returns the number corresponding to the player owning this piece"""
    return piece % 2


def parse(bs):  # bs is board string
    '''Translate a board string into the list of lists representation.'''
    b = [[0, 0, 0, 0, 0, 0, 0, 0] for r in range(8)]
    rs9 = bs.split("\n")
    rs8 = rs9[1:]   # eliminate the empty first item.
    for iy in range(8):
        rss = rs8[iy].split(' ')
        for jx in range(8):
            b[iy][jx] = INIT_TO_CODE[rss[jx]]
    return b

# Baroque chess initial state
INITIAL = parse('''
c l i w k i l f
p p p p p p p p
- - - - - - - -
- - - - - - - -
- - - - - - - -
- - - - - - - -
P P P P P P P P
F L I W K I L C
''')


class BC_state:
    """Representation of a baroque chess state."""
    def __init__(self, old_board=INITIAL, whose_move=WHITE):
        new_board = [r[:] for r in old_board]
        self.board = new_board
        self.whose_move = whose_move

    def __repr__(self):
        s = ''
        for r in range(8):
            for c in range(8):
                s += CODE_TO_INIT[self.board[r][c]] + " "
            s += "\n"
        if self.whose_move == WHITE:
            s += "WHITE's move"
        else:
            s += "BLACK's move"
        s += "\n"
        return s


def introduce():
    """
    Returns a string that introduces the SmartBC baroque chess agent and
    its benevolent creators.
    """

    msg =\
        """Hi my name is SmartBC. My creators are Kevin Fong (kfong94) and Jordan
        Hand (jhand1). I is smart."""
    return msg


def nickname():
    """Returns the nickname of this baroque chess agent."""
    return "SmartBC"


def makeMove(currentState, currentRemark, timeLimit=10000):
    move = minimax(currentState)

    return [["Move", move[0]], "Take that!"]


def staticEval(state):
    score = 0
    for row in state.board:
        for space in row:
            if space != 0:
                if who(space) == BLACK:
                    score -= 1
                else:
                    score += 1
    return score


def minimax(state, curr_ply=0):
    """
    I don't really think I'm finished with this but its a start.

    returns:
        tuple containing (board_state, value_of_state)
    """
    def is_better(curr, best, p):
        return curr < best if p == 0 else curr > best

    max_ply = 5
    board = state.board

    if curr_ply == max_ply:
        return (state, staticEval(state))

    best_val = -9999
    best_state = None
    for row in range(board):
        for col in range(row):
            piece = state[row][col]
            new_pos = move_funcs[piece / 2]((col, row))

            if can_move_funcs[piece / 2]((col, row), new_pos, state):
                new_b = filter_board((col, row), new_pos)
                next_state = BC_state(new_b, state.whose_move % 1)
                val = minimax(next_state, curr_ply + 1)[1]
                if is_better(val, best_val, state.whose_move):
                    best_val = val
                    best_state = next_state

    if best_state is None:
        return None

    return (best_state, best_val)


class Operator:
    """
    Represents an operator that moves a piece to another index.
    Includes a move name, precondition function, and state transfer
    function.
    """
    def __init__(self, name, precond, state_transf):
        self.name = name
        self.precond = precond
        self.state_transf = state_transf

    def is_applicable(self, s):
        """Operator precondition"""
        return self.precond(s)

    def apply(self, s):
        """
        Operator precondition. Returns the state as a result of this
        move being applied.
        """
        return self.state_transf(s)

# Min and max x and y values on the given board. 0, 0 is the top left of the
# board.
min_x = 0
max_x = len(INITIAL[0])

min_y = 0
max_y = len(INITIAL)


def filter_board(start_pos, end_pos, state):
    """
    Creates a new board state based on the piece in start_pos being moved to
    end_pos.
    """
    new_b = state.board[:]
    x1 = start_pos[0]
    x2 = end_pos[0]
    y1 = start_pos[1]
    y2 = end_pos[1]

    new_b[x2][y2] = new_b[x1][y1]
    new_b[x1][y1] = 0

    new_state = BC_state(old_board=new_b, whose_move=state.whose_move % 1)
    return new_state

# --------------
# Make operators
# --------------


def gen_pincer_moves(pos):
    """
    Generates all possible moves for pincers regardless of what other pieces
    are on the board.
    """
    return gen_four_cardinal(pos, (min_x, max_x), (min_y, max_y))


def gen_all_dir(pos):
    """
    Generates all possible moves for all pieces except pincers and kings
    regardless of what other pieces are on the board.
    """
    cardinal = gen_four_cardinal(pos, (min_x, max_x), (min_y, max_y))
    diag = gen_four_diag(pos, (min_x, max_x), (min_y, max_y))
    return cardinal + diag


def gen_four_cardinal(pos, x_extrema, y_extrema):
    """
    Generates moves in the four cardinal directions where x and y are possible
    coordinates the piece at pos can move to and
    x_extrema[0] <= x < x_extrema[1] and y_extrema[0] <= y < x_extrema[1]
    """
    moves = []

    for i in range(x_extrema[0], x_extrema[1]):
        if i != pos[0]:
            m = (i, pos[1])
            moves.append(m)
    for j in range(y_extrema[0], y_extrema[1]):
        if j != pos[1]:
            m = (pos[0], j)
            moves.append(m)

    return moves


def gen_four_diag(pos, x_extrema, y_extrema):
    """
    Generates moves in the four diagonal directions where x and y are possible
    coordinates the piece at pos can move to and
    x_extrema[0] <= x < x_extrema[1] and y_extrema[0] <= y < x_extrema[1]
    """
    moves = []

    i = x_extrema[0]
    j = y_extrema[0]
    while i < max_x and j < max_y:
        if i != pos[0] and j != pos[1]:
            m = (i, j)
            moves.append(m)
        i += 1
        j += 1

    i = x_extrema[0]
    j = y_extrema[1]
    while i < max_x and j >= min_y:
        if i != pos[0] and j != pos[1]:
            m = (i, j)
            moves.append(m)
        i += 1
        j -= 1
    return moves


def gen_king_moves(pos):
    """
    Generates all possible moves for the king regardless of what other pieces
    are on the board.
    """
    moves = []

    minx = pos[0] - 1 if pos[0] > min_x else pos[0]
    maxx = pos[0] + 1 if pos[0] < max_x - 1 else pos[0]
    miny = pos[1] - 1 if pos[1] > min_y else pos[1]
    maxy = pos[1] + 1 if pos[1] < max_y else pos[1]

    moves += gen_four_cardinal(pos, (minx, maxx), (miny, maxy))
    moves += gen_four_diag(pos, (minx, maxx), (miny, maxy))

    return moves


def gen_king_captures(s, moves, player):
    """
    If movable tile contains enemy, add capture move
    Black = 0 (evens), White = 1 (odds)
    """
    captures = []
    for move in moves:
        tile = s.board[move[0]][move[1]]
        if (tile + 1) % 2 == player and tile != 0:
            captures.append(move)
    return captures


def gen_pincer_captures(s, moves, player, pos):
    """
    If movable tile contains enemy and tile past enemy is ally, add capture move
    """
    captures = []
    for move in moves:
        tile = s.board[move[0]][move[1]]
        x_dir = move[0] - pos[0]
        y_dir = move[1] - pos[1]
        if not (move[0] + x_dir) < min_x and not (move[0] + x_dir) >= max_x \
                and not (move[1] + y_dir) < min_y and\
                not (move[1] + y_dir) >= max_y:
            partner_tile = s.board[move[0] + x_dir][move[1] + y_dir]
            if (tile + 1) % 2 == player and partner_tile % 2 == player\
                    and partner_tile != 0:
                captures.append(move)
    return captures


def gen_withdrawer_captures(s, moves, player, pos):
    """
    If movable tile contains enemy and any tile away from enemy is empty,
    add capture move
    """
    captures = []
    for move in moves:
        tile = s.board[move[0]][move[1]]
        if (tile + 1) % 2 == player:
            x_dir = move[0] - pos[0]
            y_dir = move[1] - pos[1]
            withdraw_x = pos[0] - x_dir
            withdraw_y = pos[1] - y_dir
            if not withdraw_x < min_x and not withdraw_x >= max_x \
                    and not withdraw_y < min_y and not withdraw_y >= max_y:
                withdraw_tile = s.board[withdraw_x][withdraw_y]
                while withdraw_tile == 0:
                    captures.append(withdraw_tile)
                    withdraw_x -= x_dir
                    withdraw_y -= y_dir
                    if not withdraw_x < min_x and not withdraw_x >= max_x \
                            and not withdraw_y < min_y\
                            and not withdraw_y >= max_y:
                        withdraw_tile = s.board[withdraw_x][withdraw_y]
                    else:
                        withdraw_tile = -1
    return captures


def gen_leaper_captures(s, moves, player, pos):
    """
    If movable tile contains enemy, add all tiles after it until you reach end
    of board or reach non-blank tile
    """
    captures = []
    for move in moves:
        tile = s.board[move[0]][move[1]]
        if (tile + 1) % 2 == player:
            x_dir = move[0] - pos[0]
            y_dir = move[1] - pos[1]
            leap_x = move[0] + x_dir * 2
            leap_y = move[1] + y_dir * 2
            if not leap_x < min_x and not leap_x >= max_x \
                    and not leap_y < min_y and not leap_y >= max_y:
                leap_tile = s.board[leap_x][leap_y]
                while leap_tile == 0:
                    captures.append(leap_tile)
                    leap_x += x_dir * 2
                    leap_y += y_dir * 2
                    if not leap_x < min_x and not leap_x >= max_x \
                            and not leap_y < min_y and not leap_y >= max_y:
                        leap_tile = s.board[leap_x][leap_y]
                    else:
                        leap_tile = -1

    return captures


def gen_coordinator_captures(s, moves, player, pos):
    """
    Idk how to do the logistics of how to check the intersection cause various
    situations and stuff
    Will think about later
    """
    captures = []
    return captures


def gen_freezer_captures(s, moves, player, pos):
    """
    Freezers can't capture so return empty list
    Also, just realized we'll need to re-check the moves and make sure a piece
    isn't next to a freezer before it moves
    """
    captures = []
    return captures


def gen_imitator_captures(s, moves, player, pos):
    """
    Get tiles around it, look at what enemy pieces are around it, for each one
    call that method and add to captures.
    Should be easy to implement
    """
    captures = []
    return captures


# Move generator functions for different piece types. The key for a piece can
# be retrieved by taking the piece number (defined at the top of the file) and
# dividing by 2.
move_funcs = {
    1: gen_pincer_moves,
    2: gen_all_dir,
    3: gen_all_dir,
    4: gen_all_dir,
    5: gen_all_dir,
    6: gen_king_moves,
    7: gen_all_dir
}


def most_precond(prev_pos, new_pos, state):
    """
    Precondition piece for all pieces but the king.
    """
    board = state.board
    y1 = prev_pos[0]
    y2 = new_pos[0]
    x1 = prev_pos[1]
    x2 = new_pos[1]
    leaper = board[y1][x1] / 2 == 3
    capture_count = 0

    xvals = sorted((x2, x1))
    yvals = sorted((y2, y1))
    if y2 == y1:    # horizontal move
        for space in board[y1][xvals[0] + 1:xvals[1]]:
            if space != 0:
                capture_count += 1
                if not leaper:
                    return False
                elif capture_count > 1:
                    return False

    elif x1 == x2:  # vertical move
        for row in board[yvals[0] + 1:yvals[1]]:
            if row[x1] != 0:
                capture_count += 1
                if not leaper:
                    return False
                elif capture_count > 1:
                    return False
    else:   # Diagonal move
        x_dir = new_pos[1] - prev_pos[1]
        x_dir = x_dir / abs(x_dir)
        y_dir = new_pos[0] - prev_pos[0]
        y_dir = y_dir / abs(y_dir)
        space = (prev_pos[0] + y_dir, prev_pos[1] + x_dir)
        while space[0] != new_pos[0] and space[1] != new_pos[1]:
            if board[space[0]][space[1]] != 0:
                capture_count += 1
                if not leaper:
                    return False
                elif capture_count > 1:
                    return False
            space[0] += y_dir
            space[1] += x_dir
        # temp_x = x1 + 1
        # for row in board[yvals[0] + 1:yvals[1]]:

    return board[y2][x2] == 0


def all_precond(prev_pos, new_pos, state):
    board = state.board
    x1 = prev_pos[1]
    y1 = new_pos[1]
    x2 = prev_pos[0]
    y2 = new_pos[0]
    piece = board[y1][x1]
    dest_piece = board[y2][x2]

    return prev_pos != new_pos and who(piece) == state.whose_move\
        and who(dest_piece) != state.whose_move

"""
1 = pincer
2 = coordinator
3 = leaper
4 = imitator
5 = withdrawer
6 = king
7 = freezer
"""
can_move_funcs = {
    1: "",
    2: "",
    3: "",
    4: "",
    5: "",
    6: lambda prev, new, st: all_precond(prev, new, st),
    7: ""
}
