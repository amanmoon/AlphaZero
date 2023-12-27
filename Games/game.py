"""NOTE : THIS IS THE FORMAT YOU SHOULD FOLLOW TO MAKE GAMES"""

class GAME:
    def __init__(self):
        self.row = None
        self.col = None
        self.possible_state = self.row * self.col # no of possible state game can have (no of possible moves)

    def initialise_state(self):
        """NOTE: this fuction creates starting position for your game"""

    def get_valid_moves(self, state):
        """ NOTE: inputs state and output all possible valid moves"""

    def know_terminal_value(self, state, action):
        """NOTE: input state and action and the output will be tuple True 
        if is terminal state and False if not and value 0 is
        player lost and 1 if the player won the game"""

    def make_move(self, state, action, player):
        """NOTE: input a action and a state and a player, the 
        action is taken on given state and output is the final state"""

    def get_encoded_state(self, state):
        """input a state the output should be the encoded state that will be
        given to the neural network"""

    def get_opponent(self, player):
        """NOTE: taking the input of current player this should change the 
        player to the opponent"""
        return -1 * player

    def get_opponent_value(self, value):
        """this should switch the reward obtained from the know_terminal_state() 
        to the reward of the opponent"""
        return -1 * value

    def change_perspective(self, state, player = -1):
        """this function taking input state and player flip the board and return 
        the game with the perspective of the opponent"""
        return player * state
