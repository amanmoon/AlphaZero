import numpy as np

class TicTacToe:

    def __init__(self):
        self.col = 3
        self.row = 3
        self.possible_state = self.col * self.row

    def initialise_state(self):
        return np.zeros((self.row, self.col))

    def get_valid_moves(self, state):
        return (state.reshape(-1) == 0).astype(np.uint8)

    def know_terminal_value(self, state, action):

        if action is None:
            return False, 0

        row=action//3
        col=action%3
        player = state[row,col]

        if (sum(state[row,:])==player*3
            or sum(state[:,col])==player*3
            or np.trace(state)==player*3
            or np.trace(np.fliplr(state))==player*3):
            return True, 1

        if 0 not in state.reshape(-1):
            return True, 0
        else:
            return False,0

    def make_move(self, state, action, player):
        row=action//3
        col=action%3

        state[row,col]=player

        return state

    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)
        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)

        return encoded_state
    def get_opponent(self, player):
        return -1 * player

    def get_opponent_value(self, value):
        return -1 * value

    def change_perspective(self, state, player = -1):
        return (player * state).astype(np.float32)
