import numpy as np
class tic_tac_toe:
    
    def __init__(self):
        self.row_number=3
        self.column_number=3
        self.action_space = self.row_number * self.column_number
    def initialise_state(self):
        return np.zeros((self.column_number,self.row_number))

    def valid_moves(self,state):
        moves=list()
        for value,player in enumerate(state.reshape(-1)):
            if player==0:
                moves.append(value)
        return moves

    def state_modify(self,state,action,player):
        row=action//3
        col=action%3
        s = np.copy(state)
        s[row,col]=player
        return s

    def check_terminal_state(self,state,action):
        if action is None:
            return False,0
        row=action//3
        col=action%3
        for player in [-1, 1]:
            if (sum(state[row,:])==player*3 
                or sum(state[:,col])==player*3
                or np.trace(state)==player*3 
                or np.trace(np.fliplr(state))==player*3):
                return True ,player
        if 0 not in state.reshape(-1):
            return True,0
        return False, 0
            
    def change_player(self,player):
        return -1 * player
    
    def change_perspective(self, state,player):
        return (state * player).astype(np.int32)

    def get_opponent_value(self, value):
        return -1 * value