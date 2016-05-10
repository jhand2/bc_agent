import time
import random as rand
import math
# import bcs

BLACK = 0
WHITE = 1
S_TIME = 0

prev_eval = 0

# Numeric codes for each BC piece
INIT_TO_CODE = {'p': 2, 'P': 3, 'c': 4, 'C': 5, 'l': 6, 'L': 7, 'i': 8, 'I': 9,
                'w': 10, 'W': 11, 'k': 12, 'K': 13, 'f': 14, 'F': 15, '-': 0}

# Text representation of each BC piece's numeric code.
CODE_TO_INIT = {0: '-', 2: 'p', 3: 'P', 4: 'c', 5: 'C', 6: 'l', 7: 'L', 8: 'i',
                9: 'I', 10: 'w', 11: 'W', 12: 'k', 13: 'K', 14: 'f', 15: 'F'}


def who(piece):
    """Returns the number corresponding to the player owning this piece"""
    if piece == 0:
        return -1
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


def prepare(nickname):
    pass


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


weights = {
    1: 1,
    2: 5,
    3: 3,
    4: 3,
    5: 9,
    6: 200,
    7: 5
}


def getKing(s, player):
    for row in range(len(s.board)):
        for col in range(len(s.board[0])):
            piece = s.board[row][col]
            if piece == 12 + player:
                return [row, col]


def staticEval(state):
    score = 0
    y = -1
    for row in state.board:
        y += 1
        x = -1
        for space in row:
            mult = 1 if who(space) == 1 else -1
            x += 1
            if space != 0:
                score += mult * (weights[space // 2] * (space // 2))
                if space // 2 == 6:
                    # Points depending on where king is on board,
                    # good on same side, bad on other side
                    score += ((min_y + y) * .1) * mult

                    # More for each thing frozen by freezer
                    if space // 2 == 7:
                        for adj in get_surrounding(state, [y,x]):
                            if who(space) == WHITE:
                                piece = state.board[adj[0]][adj[1]]
                                score += mult * ((weights[piece // 2]) * .3)

                    # If coordinator in same col or row as king, minus some points
                    if space // 2 == 2:
                        king_pos = getKing(state, who(space))
                        if y == king_pos[0] or x == king_pos[1]:
                            score -= mult * (weights[2] * 0.5)
                # Other ideas:
                #     If piece cant move, minus some points
                #     Add weighted points for each possible capture a piece can do
                #     Add points just based on the number of legal moves compared to opponent
                    # score -= len(move_funcs[space // 2]((y,x))) * .01
    return score


def choose_message(eval_score, player):
    better = "Ha! Take that!"
    good = "This is a good one."
    bad = "Eh, not bad."
    worse = "Oh... I should just give up now..."
    if is_better(eval_score, prev_eval, player):
        if abs(eval_score - prev_eval) > 20:
            return better
        else:
            return good
    else:
        if abs(eval_score - prev_eval) > 20:
            return worse
        else:
            return bad
    

def is_better(curr, best, p):
    """
    Function to determine which minimax eval score is better based on which
    player p is currently moving.
    """
    if curr is None:
        return False
    elif best is None:
        return True
    else:
        return curr < best if p == 0 else curr > best


def stop_test(limit):
    multiplier = 0.9
    return get_elapsed() > limit - (limit * multiplier)


def makeMove(currentState, currentRemark, timeLimit):
    global S_TIME
    S_TIME = time.time() * 1000
    limit = timeLimit * 1000

    move = [currentState, 0]
    for i in range(1, 100):
        alpha = -math.inf
        beta = math.inf
        temp = minimax(currentState, 0, i, limit, alpha, beta)
        # print(i)
        if temp != None:
            # print("New Move")
            move = temp
        if stop_test(limit):
            break

    return [["Move", move[0]], "Take that!"]


def minimax(state, curr_ply, max_ply, time_limit, alpha, beta):
    """
    I don't really think I'm finished with this but its a start.

    returns:
        tuple containing (board_state, value_of_state)
    """
    # r = False
    r = True

    if stop_test(time_limit):
        return None

    board = state.board

    if curr_ply == max_ply:
        se = staticEval(state)
        print(se)
        return (state, se)

    states = []
    # Iterates through the whole board
    for row in range(len(board)):
        for col in range(len(board[0])):
            piece = board[row][col]
            if piece != 0:
                moves = move_funcs[piece // 2]((row, col))
                if stop_test(time_limit):
                    return None
                for m in moves:
                    if can_move_funcs[piece // 2]((row, col), m, state):
                        new_b = filter_board((row, col), m, state)
                        next_state = BC_state(new_b, 1 - state.whose_move)
                        states.append(next_state)

    if r:
        new_s = rand.choice(states)
        return (new_s, 0)
    else:
        # Alpha beta pruning search
        best_val = alpha if state.whose_move == WHITE else beta
        best_state = None
        a = alpha
        b = beta
        for s in states:
            move = minimax(s, curr_ply + 1, max_ply, time_limit, a, b)
            if move == None:
                return None
            val = move[1]
            if is_better(val, best_val, state.whose_move):
                best_val = val
                best_state = next_state
                if state.whose_move == WHITE:
                    a = best_val
                else:
                    b = best_val
            if b <= a:
                break

    return (best_state, best_val)


def get_elapsed():
    """
    Returns time elapsed from the start of the current makeMove search
    """
    return (time.time() * 1000) - S_TIME


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
    new_b = []
    for row in state.board:
        r = row[:]
        new_b.append(r)
    x1 = start_pos[1]
    x2 = end_pos[1]
    y1 = start_pos[0]
    y2 = end_pos[0]

    piece = new_b[y1][x1]
    new_b[y2][x2] = piece
    new_b[y1][x1] = 0
    if piece // 2 != 6:     # Cause king just takes over spot
        captures = cap_dict[piece // 2](state, end_pos, start_pos)
        for c in captures:
            # print("Piece taken")
            new_b[c[0]][c[1]] = 0

    return new_b

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
        if i != pos[1]:
            m = (pos[0], i)
            moves.append(m)
    for j in range(y_extrema[0], y_extrema[1]):
        if j != pos[0]:
            m = (j, pos[1])
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
    while i < x_extrema[1] and j < y_extrema[1]:
        if i != pos[1] and j != pos[0]:
            m = (j, i)
            moves.append(m)
        i += 1
        j += 1

    i = x_extrema[0]
    j = y_extrema[1] - 1
    while i < x_extrema[1] and j >= y_extrema[0]:
        if i != pos[1] and j != pos[0]:
            m = (j, i)
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

    minx = pos[1] - 1 if pos[1] > min_x else pos[1]
    maxx = pos[1] + 1 if pos[1] < max_x - 1 else pos[1]
    miny = pos[0] - 1 if pos[0] > min_y else pos[0]
    maxy = pos[0] + 1 if pos[0] < max_y else pos[0]

    moves += gen_four_cardinal(pos, (minx, maxx), (miny, maxy))
    moves += gen_four_diag(pos, (minx, maxx), (miny, maxy))

    return moves


def king_captures(s, move, pos):
    """
    If movable tile contains enemy, add capture move
    """
    captures = []
    tile = s.board[move[0]][move[1]]
    if who(tile) != s.whose_move and tile != 0:
        captures.append(move)

    return captures


def pincer_captures(s, move, pos):
    """
    If movable tile contains enemy and tile past enemy is ally, add capture move
    """
    captures = []

    to_test = [
        [(move[0] - 1, move[1]), (move[0] - 2, move[1])],
        [(move[0] + 1, move[1]), (move[0] + 2, move[1])],
        [(move[0], move[1] - 1), (move[0], move[1] - 2)],
        [(move[0], move[1] + 1), (move[0], move[1] + 2)]
    ]
    for pair in to_test:
        y1, x1 = pair[0][0], pair[0][1]
        y2, x2 = pair[1][0], pair[1][1]
        if y1 > 0 and x1 > 0 and y2 > 0 and x2 > 0:
            try:
                one_away = s.board[y1][x1]
                two_away = s.board[y2][x2]
            except IndexError:
                continue

            if who(two_away) == s.whose_move and\
                    who(one_away) == 1 - s.whose_move:
                captures.append(pair[0])
    return captures


def get_direction(p1, p2):
    x_dir = p2[1] - p1[1]
    if x_dir != 0:
        x_dir = x_dir // abs(x_dir)

    y_dir = p2[0] - p1[0]
    if y_dir != 0:
        y_dir = y_dir // abs(y_dir)
    return (y_dir, x_dir)



def withdrawer_captures(s, move, pos):
    """
    If movable tile contains enemy and any tile away from enemy is empty,
    add capture move
    """
    captures = []

    enemy_positions = get_surrounding(s, pos)
    for enemy in enemy_positions:
        e = s.board[enemy[0]][enemy[1]]
        init_dir = get_direction(enemy, pos)
        new_dir = get_direction(enemy, move) 
        cap = in_line(move, enemy) and (init_dir == new_dir)
        
        if who(e) != s.whose_move and e != 0 and cap:
            captures.append(enemy)

    return captures


def leaper_captures(s, move, pos):
    """
    If movable tile contains enemy, add all tiles after it until you reach end
    of board or reach non-blank tile
    """
    captures = get_line(s, move, pos)

    return captures


def get_surrounding(s, pos):
    """
    Returns pieces surrounding the space pos
    """
    spaces = []
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            if not (i == j == 0):
                if 0 <= pos[0] + i < 8 and 0 <= pos[1] + j < 8:
                    spaces.append((pos[0] + i, pos[1] + j))
    return spaces


def in_line(p1, p2):
    """
    Tests if p1 and p2 are in a straight line
    """
    cardinal = p1[0] == p2[0] or p1[1] == p2[1]
    if cardinal:
        return True
    else:
        slope = (p1[0] - p2[0]) / (p1[1] - p2[1])
        return slope == 1 or slope == -1


def get_line(s, p1, p2):
    """
    Returns an in order list of any nonblank pieces in a straight line from
    p1 to p2 in s.
    """
    pieces = []
    board = s.board
    x1, x2 = p1[1], p2[1]
    y1, y2 = p1[0], p2[0]


    xvals = sorted((x1, x2))
    yvals = sorted((y1, y2))
    if y1 == y2:    # horizontal move
        for x in range(xvals[0] + 1, xvals[1]):
            if board[y1][x] != 0:
                pieces.append((y1, x))
    elif x1 == x2:  # vertical move
        for y in range(yvals[0] + 1, yvals[1]):
            if board[y][x1] != 0:
                pieces.append((y, x1))
    else:   # Diagonal move
        x_dir = p2[1] - p1[1]
        x_dir = x_dir // abs(x_dir)

        y_dir = p2[0] - p1[0]
        y_dir = y_dir // abs(y_dir)
        space = [p1[0] + y_dir, p1[1] + x_dir]
        while space[0] != p2[0] and space[1] != p2[1]:
            if board[space[0]][space[1]] != 0:
                pieces.append((space[0], space[1]))
            space[0] += y_dir
            space[1] += x_dir
    return pieces


def leaper_captures(s, move, pos):
    """
    If movable tile contains enemy, add all tiles after it until you reach end
    of board or reach non-blank tile
    """
    captures = []
    line = get_line(s, move, pos)
    count = 0
    for space in line:
        piece = s.board[space[0]][space[1]]
        if who(piece) != s.whose_move:
            if piece != 0:
                if count > 1:
                    return []
                else:
                    captures.append(space)
                count += 1
        else:
            return []

    return captures


def coordinator_captures(s, move, pos):
    """
    Pretty sure this works
    """
    captures = []
    king = 12
    if s.whose_move == 1:
        king = 13
    k_space = None
    for row in range(len(s.board)):
        try:
            i = s.board[row].index(king)
            k_space = (row, i)
            break
        except:
            pass
    if k_space is not None:
        s1 = (k_space[0], move[1])
        s2 = (move[0], k_space[1])
        for space in (s1, s2):
            piece = s.board[space[0]][space[1]]
            if piece != 0 and who(piece) != s.whose_move:
                captures.append(space)
    return captures


def imitator_captures(s, move, pos):
    """
    This might work
    """
    captures = []
    b = s.board
    for y in range(len(b)):
        for x in range(len(b[0])):
            space = (y,x)
            piece = b[y][x]
            if piece != 0 and who(piece) != s.whose_move and piece // 2 != 4:
                caps = cap_dict[piece // 2](s, move, pos)
                if space in caps:
                    captures.append(space)
    return captures


"""
1 = pincer
2 = coordinator
3 = leaper
4 = imitator
5 = withdrawer
6 = king
7 = freezer
"""
cap_dict = {
    1: pincer_captures,
    2: coordinator_captures,
    3: leaper_captures,
    4: imitator_captures,
    5: withdrawer_captures,
    6: king_captures,
    7: lambda s, m, pos: []
}


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
    Precondition for all pieces but the king.
    """
    if not all_precond(prev_pos, new_pos, state):
        return False

    board = state.board
    y1 = prev_pos[0]
    y2 = new_pos[0]
    x1 = prev_pos[1]
    x2 = new_pos[1]
    piece = board[y1][x1]
    leaper = piece // 2 == 3
    cap_count = 0

    if in_line(prev_pos, new_pos):
        spaces = get_line(state, prev_pos, new_pos)
        capture_count = 0

        for s in spaces:
            enemy = board[s[0]][s[1]]
            if enemy != 0:
                if who(enemy) == state.whose_move:
                    return False
                cap_count += 1
                if (not leaper) or (leaper and capture_count > 1):
                    enemy_leaper = enemy // 2 == 3
                    imitator = piece // 2 == 4
                    if not (imitator and cap_count == 1 and enemy_leaper):
                        return False
    else:
        return False

    return board[y2][x2] == 0


def all_precond(prev_pos, new_pos, state):
    """
    A precondition that does some checking that applies to all pieces. Many
    pieces require further checking.
    """
    board = state.board
    x1 = prev_pos[1]
    x2 = new_pos[1]
    y1 = prev_pos[0]
    y2 = new_pos[0]
    piece = board[y1][x1]
    dest_piece = board[y2][x2]

    has_piece = who(piece) == state.whose_move
    space_avail = dest_piece == 0 or who(dest_piece) != state.whose_move
    pos_diff = prev_pos[0] != new_pos[0] or prev_pos[1] != new_pos[1]

    # Check if next to a freezer
    adj = get_surrounding(state, prev_pos)
    for space in adj:
        enemy = board[space[0]][space[1]]
        enemy_freezer = enemy // 2 == 7 and who(enemy) != state.whose_move
        friendly_freezer = piece // 2 == 7 and who(piece) == state.whose_move
        enemy_im = enemy // 2 == 4 and who(enemy) != state.whose_move


        if enemy_freezer or (friendly_freezer and enemy_im):
            return False

    return has_piece and space_avail and pos_diff

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
    1: most_precond,
    2: most_precond,
    3: most_precond,
    4: most_precond,
    5: most_precond,
    6: all_precond,
    7: most_precond
}
