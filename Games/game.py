"""NOTE : THIS IS THE FORMAT YOU SHOULD FOLLOW TO MAKE GAMES"""

class GAME:
    def __init__(self):
        self.possible_state = 0 # no of possible state game can have (no of possible moves) 
        pass
    
    def initialise_state(self):
        """NOTE: this fuction creates starting position for your game"""
        pass
    
    def get_moves(self, state):        
        """ NOTE: inputs state and output all possible valid moves"""
        pass

    def know_terminal(self,state):
        """NOTE: input state and output will be tuple True 
        if is terminal state and False if not and value 0 is
        player lost and 1 if the player won the game"""
        pass

    def make_move(self, state, action, player):
        """NOTE: input a action and a state and a player, the 
        action is taken on given state and output is the final state"""
    
    def get_opponent(self, player):
        """NOTE: takin the input of current player this should change the 
        player to the opponent"""
        pass