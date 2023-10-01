import sys
sys.path.insert(0,'/home/adrinospy/Programming/Projects/AI ML/general_alpha_zero/Games/TicTacToe')
from tictactoe import TicTacToe
import numpy as np 
import math

args = {
    
    "NO_OF_SEARCHES" : 1000,
    "EXPLORATION_CONSTANT" : 1.42
}

class Node:
    def __init__(self, game, args, state, parent = None, action = None):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action = action
        
        self.children = list()
        self.expandable_moves = self.game.get_moves(self.state)
        self.visits = 0
        self.wins = 0
        
    def leaf_or_not(self):
        return (len(self.children) > 0 and len(self.expandable_moves) == 0)
        
    def search(self):
        best_child = None
        best_ucb = - np.inf
        for child in self.children:
            ucb = self.get_ucb(child)
            if best_ucb < ucb:
                best_ucb = ucb 
                best_child = child
        return best_child

    def get_ucb(self, child):
        q_value = 1 - ((child.wins / child.visits) + 1) / 2
        return q_value + self.args["EXPLORATION_CONSTANT"] * math.sqrt(math.log(self.visits) / child.visits)
    
    def expand(self, player):
        rand_move = np.random.choice(self.expandable_moves)
        self.expandable_moves.remove(rand_move)
        child = self.game.make_move(self.state, rand_move, player)
        child = self.game.change_perspective(child)
        child = Node(self.game, self.args, child, self, rand_move)
        self.children.append(child)
        return child
    
    def simulate(self, player):
        is_terminal, value = self.game.know_terminal_value(self.state, self.action, -1 * player)
        
        if is_terminal:
            return value
        state = self.state.copy()
        while True:
            possible_moves = self.game.get_moves(state)
            rand_move = np.random.choice(possible_moves)
            state = self.game.make_move(state, rand_move, player)
            is_terminal, value = self.game.know_terminal_value(state, rand_move,player)
            if is_terminal:
                return value
            possible_moves = self.game.get_moves(state)
            rand_move = np.random.choice(possible_moves)
            player = self.game.get_opponent(player)
        
    def backpropagate(self,value):
        self.wins += value
        self.visits += 1
        if self.parent is not None:
            self.parent.backpropagate(value)
            
class MCTS:
    def __init__(self, game, args):
        self.game = game
        self.args = args
        
    def search(self, node):
        root = Node(self.game, self.args, node)

        for _ in range(self.args["NO_OF_SEARCHES"]):
            node = root
            player = 1
            while node.leaf_or_not():
                node = node.search()
                player = self.game.get_opponent(player)
                
            is_terminal, value = self.game.know_terminal_value(node.state, node.action, player)
            if not is_terminal:
                node = node.expand(player)
                value = node.simulate(player)
            node.backpropagate(value)
            
        move_probability = np.zeros(self.game.possible_state)
        for children in root.children:
            move_probability[children.action] = children.visits
        move_probability /= np.sum(move_probability)

        return move_probability
 
# game = TicTacToe()
# state = game.initialise_state()
# mcts = MCTS(game, args)
# out = mcts.search(state)
# print(out)
# while True:
#     print(state)
#     move = int(input("enter move:"))
#     state = game.make_move(state, move,1)
#     out = mcts.search(state)
#     is_terminal ,value = game.know_terminal_value(state, out.argmax(), -1)
#     state = game.make_move(state,out.argmax(),-1)