import math
import numpy as np

args={
    'MCTS_searches' : 2, # no of searches you want to perform 
    'Adversirial' : True,    # is the game two player ie. Adversirial
    'exploration_const' : 5
}

class Node:
    def __init__(self, game, args, state, parent = None, action = None):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action = action
        
        self.childerns = list()
        self.expandable_moves = game.valid_moves(state)
        self.visit_count = 0
        self.value_sum = 0
    
    def select(self):
        best_child = None
        best_ucb = - np.inf
        for child in self.expandable_moves:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
                                
        return best_child 
    
    def get_ucb(self,state):
        q_value = ((state.value_sum / state.visit_count) + 1) / 2

        if self.args['Adversirial']:
            q_value = 1 - q_value + self.args['exploration_const'] * math.sqrt((math.log(self.visit_count) / state.visit_count))
        
        return q_value 
    
    def fully_expanded(self):
        return len(self.expandable_moves) == 0 and len(self.childerns) > 0
    
    def expand(self):
        action = np.random.choice(self.expandable_moves)
        self.expandable_moves.remove(action)
        child = np.copy(self.state)
        child = self.game.state_modify( child, action, 1)
        if self.args['Adversirial']:
            child = self.game.change_perspective(child, player = -1)
        child = Node(self.game, self.args, child, self, action)        
        self.childerns.append(child)
        
        return child
    
    def simulate(self):
        is_terminal, value = self.game.check_terminal_state(self.state,self.action)
        if is_terminal:
            return value

        rollout_state = self.state.copy()
        rollout_player = 1
        
        while True:
            valid_moves = self.game.valid_moves(rollout_state)
            action = np.random.choice(valid_moves)
            rollout_state = self.game.state_modify(rollout_state, action, rollout_player)
            is_terminal, value = self.game.check_terminal_state(rollout_state, action)
        
            if is_terminal:
                return value
            if self.args['adversirial']:
                rollout_player = self.game.change_player(rollout_player)
    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1
        self.value = self.game.get_opponent_value(value)
        
        if self.parent is not None:
            self.parent.backpropagate(value)
            
class MCTS:

    def __init__(self, game, args):
        self.game = game
        self.args = args
    
    def search(self,state):
        root = Node(self.game, self.args, state)
        
        
        for _ in range(self.args['MCTS_searches']):
            node = root
            # selection
            while node.fully_expanded():
                node = node.select()             
            is_terminal, value = self.game.check_terminal_state(node, node.action)
            if not is_terminal :    
                value = self.game.get_opponent_value(value) 
            # expansion
            if not is_terminal:
                node = node.expand()
                # simulation
                value = node.simulate()
            # backpropagation
            node.backpropagate(value)
        
        action_probability = np.zeros(self.game_action_size)    

        for childrens in root.childerns:
            action_probability[childrens.action] = childrens.visit_count

        action_probability /= np.sum(action_probability)
            
        