import numpy as np

class ConnectFour:
    def __init__(self):
        self.row = 6
        self.col = 7
        self.possible_state = self.col
        self.in_a_row = 4
        
    def __repr__(self):
        return "ConnectFour"
        
    def initialise_state(self):
        return np.zeros((self.row, self.col))
    
    def make_move(self, state, action, player):
        row = np.max(np.where(state[:, action] == 0))
        state[row, action] = player
        return state
    
    def get_valid_moves(self, state):
        return (state[0] == 0).astype(np.uint8)
    
    def check_win(self, state, action):
        if action == None:
            return False
        
        row = np.min(np.where(state[:, action] != 0))
        column = action
        player = state[row][column]

        def count(offset_row, offset_column):
            for i in range(1, self.in_a_row):
                r = row + offset_row * i
                c = action + offset_column * i
                if (
                    r < 0 
                    or r >= self.row
                    or c < 0 
                    or c >= self.col
                    or state[r][c] != player
                ):
                    return i - 1
            return self.in_a_row - 1

        return (
            count(1, 0) >= self.in_a_row - 1 # vertical
            or (count(0, 1) + count(0, -1)) >= self.in_a_row - 1 # horizontal
            or (count(1, 1) + count(-1, -1)) >= self.in_a_row - 1 # top left diagonal
            or (count(1, -1) + count(-1, 1)) >= self.in_a_row - 1 # top right diagonal
        )
    
    def know_terminal_value(self, state, action):
        if self.check_win(state, action):
            return True, 1
        if np.sum(self.get_valid_moves(state)) == 0:
            return True, 0
        return False, 0
    
    def get_opponent(self, player):
        return -player
    
    def get_opponent_value(self, value):
        return -value
    
    def change_perspective(self, state, player):
        return state * player
    
    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)
        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)
        
        return encoded_state