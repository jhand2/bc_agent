'''new_succ.py
(formerly baroque_succ.py)

Steve Tanimoto,  May 2016, for CSE 415.

Functions to generate successors of states in Baroque Chess.

We consider 3 alternative representations of states:
 R1. ASCII, for display and initialization.
 R2. Array (list of lists), for computation of successors, etc.
 R3. Packed (8 integers of 32 bits each), for efficient storage.

 To go from R1 to R2, use function: parse.
 To go from R2 to R1, use function: __repr__ (or cast to str).
 To go from R2 to R3, use function: pack.
 To go from R3 to R2, use function: unpack.

Within R1, pieces are represented using initials; e.g., 'c', 'C', 'p', etc.
Within R2, pieces are represented using integers 1 through 14.
Within R3, pieces are represented using nibbles (half bytes).

Note: At this time, representation R3 is NOT USED AT ALL.  This is
a possible future enhancement in order to better support speedy play,
Zobrist hashing, etc.

We are following these rules:
 
  No leaper double jumps.
      SOME PEOPLE CONSIDER IT DETRIMENTAL TO THE GAME, AND IT INCREASES THE BRANCHING FACTOR,
      WHICH IS ALREADY LARGE.

  No altering the initial symmetries of the board, although Wikipedia suggests this is allowed.

  No "suicide" moves allowed.

  Pincers can pinch using any friendly piece as their partners, not just other pincers.
 
'''
BLACK = 0
WHITE = 1
NORTH = 0; SOUTH = 1; WEST = 2; EAST = 3; NW = 4; NE = 5; SW = 6; SE = 7

# Used in parsing the initial state and in testing:

INIT_TO_CODE = {'p':2, 'P':3, 'c':4, 'C':5, 'l':6, 'L':7, 'i':8, 'I':9,
  'w':10, 'W':11, 'k':12, 'K':13, 'f':14, 'F':15, '-':0}

# Used in printing out states:

CODE_TO_INIT = {0:'-',2:'p',3:'P',4:'c',5:'C',6:'l',7:'L',8:'i',9:'I',
  10:'w',11:'W',12:'k',13:'K',14:'f',15:'F'}

# Global variables representing the various types of pieces on the board:

BLACK_PINCER      = 2
BLACK_COORDINATOR = 4
BLACK_LEAPER      = 6
BLACK_IMITATOR    = 8
BLACK_WITHDRAWER  = 10
BLACK_KING        = 12
BLACK_FREEZER     = 14

WHITE_PINCER      = 3
WHITE_COORDINATOR = 5
WHITE_LEAPER      = 7
WHITE_IMITATOR    = 9
WHITE_WITHDRAWER  = 11
WHITE_KING        = 13
WHITE_FREEZER     = 15


def who(piece): return piece % 2  # BLACK's pieces are even; WHITE's are odd.

def parse(bs): # bs is board string
  '''Translate a board string into the list of lists representation.'''
  b = [[0,0,0,0,0,0,0,0] for r in range(8)]
  rs9 = bs.split("\n")
  rs8 = rs9[1:] # eliminate the empty first item.
  for iy in range(8):
    rss = rs8[iy].split(' ');
    for jx in range(8):
      b[iy][jx] = INIT_TO_CODE[rss[jx]]
  return b

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
    def __init__(self, old_board=INITIAL, whose_move=WHITE):
        new_board = [r[:] for r in old_board]  # Deeply copy the board.
        self.board = new_board
        self.whose_move = whose_move;

    def __repr__(self): # Produce an ASCII display of the state.
        s = ''
        for r in range(8):
            for c in range(8):
                s += CODE_TO_INIT[self.board[r][c]] + " "
            s += "\n"
        if self.whose_move==WHITE: s += "WHITE's move"
        else: s += "BLACK's move"
        s += "\n"
        return s

def test_starting_board():
  init_state = BC_state(INITIAL, WHITE)
  print(init_state)

#test_starting_board()

''' Now some code for move generation.
Scan through the board, testing each piece found to see if this
piece belongs to whose move it is.  If so, try to find moves for it.
'''
def successors(state):
    b = state.board
    wm = state.whose_move
    succs = []
    for ii in range(8):
        for jj in range(8):
            piece = b[ii][jj]
            if piece==0: continue  # No piece here.
            if who(piece) != wm: continue # can't move opponent's pieces
            if frozen(state,piece,ii,jj): # Frozen by freezer or imitator.
                #print("frozen at "+str(ii)+","+str(jj));
                continue
            if piece==BLACK_PINCER      or piece== WHITE_PINCER:      succs += pincer_moves(     state, piece, ii, jj)
            if piece==BLACK_KING        or piece== WHITE_KING:        succs += king_moves(       state, piece, ii, jj)
            if piece==BLACK_COORDINATOR or piece== WHITE_COORDINATOR: succs += coordinator_moves(state, piece, ii, jj)
            if piece==BLACK_LEAPER      or piece== WHITE_LEAPER:      succs += leaper_moves(     state, piece, ii, jj)
            if piece==BLACK_IMITATOR    or piece== WHITE_IMITATOR:    succs += imitator_moves(   state, piece, ii, jj)
            if piece==BLACK_WITHDRAWER  or piece== WHITE_WITHDRAWER:  succs += withdrawer_moves( state, piece, ii, jj)
            if piece==BLACK_FREEZER     or piece== WHITE_FREEZER:     succs += freezer_moves(    state, piece, ii, jj)
    return succs

def frozen(state, piece, pi, pj):
    '''If the opponent's freezer is nearby, return true. Also, if we are trying
    to move the freezer, it will be frozen if an enemy imitator is adjacent.'''
    b = state.board
    mini = max(0, pi-1)
    maxi = min(pi+1,7)
    minj = max(0, pj-1)
    maxj = min(pj+1, 7)
    opponent_freezer_1 = BLACK_FREEZER+(1-state.whose_move)
    opponent_freezer_2 = 0
    if piece==BLACK_FREEZER or piece==WHITE_FREEZER:
      opponent_freezer_2 = BLACK_IMITATOR+(1-state.whose_move) # look for an imitator
    #print("freezer is "+str(freezer))
    for ii in range(mini,maxi+1):
        for jj in range(minj,maxj+1):
            if b[ii][jj]==opponent_freezer_1: return True
            if opponent_freezer_2 > 0 and b[ii][jj]==opponent_freezer_2: return True
    return False
            
def pincer_moves(state, piece, i, j, requires_capture=False):
    b = state.board
    w = state.whose_move
    moves = []
    for direc in [('north', -1, -1, i, True),('south',1,8, i, True),('west',-1,-1, j, False),('east',1,8, j, False)]:
        d = direc[0]; delta=direc[1]; limit=direc[2]; start=direc[3]+delta; ns = direc[4]
        for idx in range(start, limit, delta):
            if ns:
                if b[idx][j] != 0: break # next square is occupied
                news = BC_state(b, 1-w)
                news.board[idx][j] = piece
                newi = idx; newj = j
            else:
                if b[i][idx] != 0: break
                news = BC_state(b, 1-w)
                news.board[i][idx] = piece
                newi = i; newj = idx
            news.board[i][j] = 0
            # test for possible capturing... First look for "partner piece" then opponent piece.
            capture = False
            if newi > 1 and b[newi-2][newj]>0 and who(b[newi-2][newj])==w and b[newi-1][newj]>0 and who(b[newi-1][newj])!=w:
                news.board[newi-1][newj]=0;
                capture = True
            if newi < 6 and b[newi+2][newj]>0 and who(b[newi+2][newj])==w and b[newi+1][newj]>0 and who(b[newi+1][newj])!=w:
                news.board[newi+1][newj]=0;
                capture = True
            if newj > 1 and b[newi][newj-2]>0 and who(b[newi][newj-2])==w and b[newi][newj-1]>0 and who(b[newi][newj-1])!=w:
                news.board[newi][newj-1]=0;
                capture = True
            if newj < 6 and b[newi][newj+2]>0 and who(b[newi][newj+2])==w and b[newi][newj+1]>0 and who(b[newi][newj+1])!=w:
                news.board[newi][newj+1]=0;
                capture = True
            if capture or not requires_capture: moves.append(news)
            #print("Appending state: "+news.toStr())
    return moves

def move(b,w,piece,oldi,oldj,newi,newj):
    news=BC_state(b,1-w)
    nb = news.board; nb[newi][newj]=piece
    nb[oldi][oldj]=0
    return news

MOVE_DIRECTION = None
MOVE_START = None

def king_moves(state, piece, i, j, requires_capture=False):
    b = state.board
    w = state.whose_move
    #print("In king_moves, w="+str(w))
    moves = []
    # For each of 8 possible directions, see first if the move would stay on the board,
    # and if not, skip this.
    # Next, check whether a capture is required and if so test whether the square
    # is empty. If empty, skip it.
    # If Not empty, see if an opponent piece is there.
    # If either no capture required or capture required and an opponent piece is there,
    # make the move and append it.
    if i>0:
        sq = b[i-1][j]
        if (not (requires_capture and sq==0)) or (requires_capture and sq>0 and who(sq)!=w):
            moves.append(move(b,w,piece,i,j,i-1,j))
    if i<7:
        sq = b[i+1][j]
        if (not (requires_capture and sq==0)) or (requires_capture and sq>0 and who(sq)!=w):
            moves.append(move(b,w,piece,i,j,i+1,j))
    if j>0:
        sq = b[i][j-1]
        if (not (requires_capture and sq==0)) or (requires_capture and sq>0 and who(sq)!=w):
            moves.append(move(b,w,piece,i,j,i,j-1))
    if j<7:
        sq = b[i][j+1]
        if (not (requires_capture and sq==0)) or (requires_capture and sq>0 and who(sq)!=w):
            moves.append(move(b,w,piece,i,j,i,j+1))
    if i>0 and j>0:
        sq = b[i-1][j-1]
        if (not (requires_capture and sq==0)) or (requires_capture and sq>0 and who(sq)!=w):
            moves.append(move(b,w,piece,i,j,i-1,j-1))
    if i>0 and j<7:
        sq = b[i-1][j+1]
        if (not (requires_capture and sq==0)) or (requires_capture and sq>0 and who(sq)!=w):
            moves.append(move(b,w,piece,i,j,i-1,j+1))
    if i<7 and j>0:
        sq = b[i+1][j-1]
        if (not (requires_capture and sq==0)) or (requires_capture and sq>0 and who(sq)!=w):
            moves.append(move(b,w,piece,i,j,i+1,j-1))
    if i<7 and j<7:
        sq = b[i+1][j+1]
        if (not (requires_capture and sq==0)) or (requires_capture and sq>0 and who(sq)!=w):
            moves.append(move(b,w,piece,i,j,i+1,j+1))
    return moves

def coordinator_moves(state, piece, i, j, requires_capture=False):
    return basic_moves(state, piece, i, j, requires_capture, coordinator_capture)

def leaper_moves(state, piece, i, j, requires_capture=False):
    return basic_moves(state, piece, i, j, requires_capture, leaper_capture)

def imitator_moves(state, piece, i, j, requires_capture=False):
    # Try each type of move, but only keep the move if it resulted in a capture.
    moves = []
    for movefn in [pincer_moves, withdrawer_moves, leaper_moves, coordinator_moves, king_moves]:
        moves += movefn(state, piece, i, j, requires_capture=True)
    # No need to do anything special here for freezer moves that freeze a freezer.
    # Get non-capturing imitator moves:
    moves += basic_moves(state, piece, i, j, False, imitator_capture)
    return moves;

def withdrawer_moves(state, piece, i, j, requires_capture=False):
    return basic_moves(state, piece, i, j, requires_capture, withdrawer_capture)

def freezer_moves(state, piece, i, j, requires_capture=False):
    #print("Looking for FREEZER moves.")
    return basic_moves(state, piece, i, j, requires_capture, freezer_capture)

def coordinator_capture(state, piece, i, j, requires_capture=False):
    '''Scan the board to find the king's position.
Then check the 2 coordinated corners for opponent pieces and
delete them from the board.'''
    def find_king():
      if who(piece)==BLACK:
        king = BLACK_KING
        for ii in range(8): # search from top of board
            for jj in range(8):
                if state.board[ii][jj]==king:
                    return(ii, jj)
        return (-1, -1)
      else:
        king = WHITE_KING # WHITE kind
        for ii in range(7,-1,-1): # seach from bottom of board
            for jj in range(8):
                if state.board[ii][jj]==king:
                    return(ii, jj)
        return (-1, -1)
    iking, jking = find_king()
    # capture up to 2 enemy pieces:
    for (itarget, jtarget) in [(i, jking), (iking, j)]:
        possible_victim = state.board[itarget][jtarget]
        if possible_victim==0: continue
        if who(possible_victim)==state.whose_move:
            state.board[itarget][jtarget]=0  # got one!
            return True
    return False

DISPLACEMENTS = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(-1,-1)]
def leaper_capture(state, piece, i, j):
    '''Continue scanning in the direction the leaper was going, and look for
a hole immediately after an adjacent opponent piece in this direction.
Use a global variable MOVE_DIRECTION to communication this direction.
Note: Whose move is already changed in the state, so the possible victim must match whose_move.
'''
    capture = False
    (di, dj) = DISPLACEMENTS[MOVE_DIRECTION]
    nexti = i
    nextj = j
    if nexti < 0 or nexti > 7 or nextj < 0 or nextj > 7: return capture
    possible_victim = state.board[nexti][nextj]
    if possible_victim==0: return capture
    if who(possible_victim) != state.whose_move: return capture  # "Poss. victim" is friendly.
    # enemy piece -- see if we can get it.
    nextii = nexti + di
    nextjj = nextj + dj
    if nextii < 0 or nextii > 7 or nextjj < 0 or nextjj > 7: return capture
    if state.board[nextii][nextjj]!=0: return capture  # not a vacant space beyond victim.
    # yes. jump him.
    print("Leaper can capture")
    capture = True
    state.board[nextii][nextjj]=piece # Leaper to final location.
    state.board[nexti][nextj]=0   # remove captured piece.
    # state.board[i][j]=0 # Leaper no longer at pre-capture position.
    return capture

def freezer_capture(state, piece, i, j): return False

REVERSE_DISPLACEMENT = [(1,0),(-1, 0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
def withdrawer_capture(state, piece, i, j):
    '''Test whether an opponent piece was next to the withdrawer when the move started
and whether the direction is correct.
Use global variables MOVE_START and MOVE_DIRECTION to get the info.
'''
    capture = False
    (oi, oj) = REVERSE_DISPLACEMENT[MOVE_DIRECTION]
    oi += MOVE_START[0]
    oj += MOVE_START[1]
    if (oi > -1) and (oi < 8) and (oj > -1) and (oj < 8):
        opiece = state.board[oi][oj]
        if opiece > 0 and (who(opiece) == state.whose_move): # whose m. just changed.
            state.board[oi][oj]=0  # captured!
            return True
    return capture

def imitator_capture(state, piece, i, j):
    '''Consider each possible piece and special-case the capturing to imitate it,
possibly calling the corresponding capture function to do the work.
Use a global variable MOVE_DIRECTION to communication this direction for the leaper case.
'''
    return False # for now.

def basic_moves(state, piece, i, j, requires_capture, capture_fun):
    '''Generate all the normal moves for a piece, and after each is created,
     apply the capture_fun to update the state in case there is a capture.

     Add diagonal moves.
    '''
    #print("Entering basic_moves")
    moves = []
    b = state.board
    w = state.whose_move
    global MOVE_DIRECTION, MOVE_START
    MOVE_START = (i,j)
    for MOVE_DIRECTION in range(8):
        d = MOVE_DIRECTION
        #print("Looking in direction "+str(d))
        if d==NORTH:
            startI = i-1; deltaI = -1
            startJ = j; deltaJ = 0
            maxSteps = startI;
        elif d==SOUTH:
            startI = i+1; deltaI = 1
            startJ = j; deltaJ = 0
            maxSteps = 7 - startI;
        elif d==WEST:
            startI = i; deltaI = 0
            startJ = j-1; deltaJ = -1
            maxSteps = startJ;
        elif d==EAST:
            startI = i; deltaI = 0
            startJ = j+1; deltaJ = 1
            maxSteps = 7 - startJ
        elif d==NW:
            startI = i-1; deltaI = -1
            startJ = j-1; deltaJ = -1
            maxSteps = min(startI, startJ)
        elif d==NE:
            startI = i-1; deltaI = -1
            startJ = j+1; deltaJ =  1
            maxSteps = min(startI, 7 - startJ)
        elif d==SW:
            startI = i+1; deltaI =  1
            startJ = j-1; deltaJ = -1
            maxSteps = min(7 - startI, startJ)
        elif d==SE:
            startI = i+1; deltaI =  1
            startJ = j+1; deltaJ =  1
            maxSteps = min(7 - startI, 7 - startJ)
        else: print("unknown direction: "+str(d))
        ii = startI; jj = startJ

        leaper = piece // 2 == 3
        occ_count = 0
        for kk in range(maxSteps+1):
            #print("trying kk as "+str(kk))
            #print("(ii,jj)="+str((ii,jj)))

            news = BC_state(b, 1-w) # New state is created, with alternation of whose move.
            if not leaper or occ_count == 1:
                if b[ii][jj] != 0:
                    break # next square is occupied

                news.board[ii][jj] = piece
            elif who(b[ii][jj]) == w:
                break
            else:
                occ_count += 1
            news.board[i][j] = 0

            # test for possible capturing...
            capture = capture_fun(news,piece,ii,jj) # modifies the new state only if capturing happens.
            if leaper and not capture:
                news.board[ii][jj] = piece

            if capture or not requires_capture:
                moves.append(news)
            if capture and leaper:
                break
                #print(news)
            ii += deltaI; jj += deltaJ;
    return moves

'''
from tests import testcase

def test():
  #ss = successors(init_state)
  ss = testcase()
  for s in ss: print(s)
  print(str(len(ss))+ " states generated. ")

test()
'''
