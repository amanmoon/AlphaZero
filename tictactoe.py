import numpy as np

class TicTacToe:
    def __init__(self):
        self.col = 3
        self.row = 3
        self.possible_state = self.col * self.row 
        pass
    
    def initialise_state(self):
        return np.zeros((self.col,self.row))
        
    def get_moves(self, state):
        moves=list()
        for value,player in enumerate(state.reshape(-1)):
            if player==0:
                moves.append(value)
        return moves

    def know_terminal_value(self, state, action, player):
        
        if action is None:
            return False, 0
        
        row=action//3
        col=action%3
        
        if (sum(state[row,:])==player*3 
            or sum(state[:,col])==player*3
            or np.trace(state)==player*3 
            or np.trace(np.fliplr(state))==player*3):
            if player == -1:
                return True , 0
            else:
                return True, 1
        
        elif 0 not in state.reshape(-1):
            return True, 0.5
        else:
            return False,0       
    def make_move(self, state, action, player):
        row=action//3
        col=action%3
        state_copy = state.copy()
        state_copy[row,col]=player

        return state_copy
    
    def get_opponent(self, player):
        return -1 * player
    
    def change_perspective(self, state):
        return (-1 * state).astype(np.int8)