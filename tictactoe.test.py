from MCTS import MCTS,Node
from tictactoe import tic_tac_toe
import math
args={
    'MCTS_searches' : 1000, # no of searches you want to perform 
    'Adversirial' : True,    # is the game two player ie. Adversirial
    'exploration_const' : 1.44
}


t = tic_tac_toe()
player=1
state=t.initialise_state()
print(state)
m = MCTS(t,args)
m.search(state)