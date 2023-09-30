from tictactoe import tic_tac_toe 
import math
import numpy as np

args = {
    "NO_OF_SEARCHES" : 10000,
    "ADVERSIRIAL" : True
}
class Node:
    
    def __init__(self, game, args, state, parent = None, action = None):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action = action
        
        self.children = list()  # children in tree
        self.expandable_moves = self.game.valid_moves(state)
        self.visit_count = 0
        self.value_sum = 0
        
    def fully_expanded(self):
        return len(self.children) > 0 and self.expandable_moves == 0

    def search(self):
        max_ucb_child = None
        max_ucb = - np.inf
        for child in self.children:
            ucb = self.check_ucb(child)
            if ucb > max_ucb:
                max_ucb = ucb
                max_ucb_child = child
        return max_ucb_child
        
    def expand(self):
        print(self.expandable_moves)
        action = np.random.choice(self.expandable_moves)
        self.expandable_moves.remove(action)
        child = self.game.state_modify(self.state, action, 1)

        if self.args['ADVERSIRIAL']:
            child = self.game.change_perspective(child, -1)
        
        child = Node(self.game, self.args, child, self, action)
        self.children.append(child)
        
        return child
        
    def simulate(self):
        is_terminal, value = self.game.check_terminal_state(self.state, self.action)

        if is_terminal:
            return value
        rollout = self.state.copy()
        player = 1 # as we are changing state to current player perspective the player will always be 1
        while True:
            action = np.random.choice(self.game.valid_moves(rollout))
            rollout = self.game.state_modify(rollout, action, player)
            is_terminal, value = self.game.check_terminal_state(rollout, action)
            if is_terminal:
                return value
            player = self.game.change_player(player)
            
    def backpropagate(self, value):
        self.visit_count += 1
        self.value_sum += value
        
        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)
    
    def check_ucb(self,state):
        q_value = ((state.value_sum / state.visit_count) + 1) / 2

        if self.args['Adversirial']:
            q_value = 1 - q_value + self.args['exploration_const'] * math.sqrt((math.log(self.visit_count) / state.visit_count))
        
        return q_value 
        
        
class MCTS:
    
    def __init__(self, game, args):
        self.game = game
        self.args = args
    
    def search(self, root):
        node = root
        
        for _ in range(self.args['NO_OF_SEARCHES']):
            node = root
            # select
            while node.fully_expanded():
                node = node.search(node)
            
            is_terminal, value = self.game.check_terminal_state(node.state, node.action)
            if not is_terminal:
                # expand
                node = node.expand()
                # simulate
                value = node.simulate()
            # backpropagation
            node.backpropagate(value)
        
        action_prob = np.zeros(self.game.action_space)
        for child in root.children:
            # print(len(root.children))
            action_prob[child.action] = child.visit_count
        action_prob /= np.sum(action_prob)


t = tic_tac_toe()
state = t.initialise_state()
print(state)
root = Node(t, args, state)
mcts = MCTS(t, args)
out = mcts.search(root)
print(root.children[0].children[0].state)