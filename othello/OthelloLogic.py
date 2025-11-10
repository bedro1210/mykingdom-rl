
class Board():

    # list of all 8 directions on the board, as (x,y) offsets
    __directions = [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1)]

    def __init__(self, n):
        "Set up initial board configuration."

        self.n = n
        # Create the empty board array.
        self.pieces = [None]*self.n
        for i in range(self.n):
            self.pieces[i] = [0]*self.n

        # Set up the initial 4 pieces.
        self.pieces[int(self.n/2)-1][int(self.n/2)] = 1
        self.pieces[int(self.n/2)][int(self.n/2)-1] = 1
        self.pieces[int(self.n/2)-1][int(self.n/2)-1] = -1;
        self.pieces[int(self.n/2)][int(self.n/2)] = -1;

    # add [][] indexer syntax to the Board
    def __getitem__(self, index): 
        return self.pieces[index]

    def countDiff(self, color):

        count = 0
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y]==color:
                    count += 1
                if self[x][y]==-color:
                    count -= 1
        return count

    def get_legal_moves(self, color):

        moves = set()  # stores the legal moves.

        # Get all the squares with pieces of the given color.
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y]==color:
                    newmoves = self.get_moves_for_square((x,y))
                    moves.update(newmoves)
        return list(moves)

    def has_legal_moves(self, color):
        for y in range(self.n):
            for x in range(self.n):
                if self[x][y]==color:
                    newmoves = self.get_moves_for_square((x,y))
                    if len(newmoves)>0:
                        return True
        return False

    def get_moves_for_square(self, square):

        (x,y) = square

        # determine the color of the piece.
        color = self[x][y]

        # skip empty source squares.
        if color==0:
            return None

        # search all possible directions.
        moves = []
        for direction in self.__directions:
            move = self._discover_move(square, direction)
            if move:
                # print(square,move,direction)
                moves.append(move)

        # return the generated move list
        return moves

    def execute_move(self, move, color):


        #Much like move generation, start at the new piece's square and
        #follow it on all 8 directions to look for a piece allowing flipping.

        # Add the piece to the empty square.
        # print(move)
        flips = [flip for direction in self.__directions
                      for flip in self._get_flips(move, direction, color)]
        assert len(list(flips))>0
        for x, y in flips:
            #print(self[x][y],color)
            self[x][y] = color

    def _discover_move(self, origin, direction):

        x, y = origin
        color = self[x][y]
        flips = []

        for x, y in Board._increment_move(origin, direction, self.n):
            if self[x][y] == 0:
                if flips:
                    # print("Found", x,y)
                    return (x, y)
                else:
                    return None
            elif self[x][y] == color:
                return None
            elif self[x][y] == -color:
                # print("Flip",x,y)
                flips.append((x, y))

    def _get_flips(self, origin, direction, color):


        flips = [origin]

        for x, y in Board._increment_move(origin, direction, self.n):
      
            if self[x][y] == 0:
                return []
            if self[x][y] == -color:
                flips.append((x, y))
            elif self[x][y] == color and len(flips) > 0:
           
                return flips

        return []

    @staticmethod
    def _increment_move(move, direction, n):
    
       
        move = list(map(sum, zip(move, direction)))
 
        while all(map(lambda x: 0 <= x < n, move)): 

            yield move
            move=list(map(sum,zip(move,direction)))
    

