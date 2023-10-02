import sys
import numpy as np 
import math
import torch

class Node:
    def __init__(self, game, args, state, parent = None, action = None, prob = 0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action = action
        self.prob = prob
        
        self.children = list()
        self.visits = 0
        self.value = 0
        
    def leaf_or_not(self):
        return (len(self.children) > 0)
    
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
        if child.visits == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value / child.visits) + 1) / 2
            
        return q_value + self.args['EXPLORATION_CONSTANT'] * (math.sqrt(self.visits) / (child.visits + 1)) * child.prob
    
    
    def expand(self, policy):
        for move, prob in enumerate(policy):
            if prob > 0:
                child = self.game.make_move(self.state.copy(), move, 1)
                child = self.game.change_perspective(child, -1)
                child = Node(self.game, self.args, child, self, move, prob)
                self.children.append(child)

    def backpropagate(self,value):
        self.value += value
        self.visits += 1
        
        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)
            
class Alpha_MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model
        
    @torch.no_grad()
    def search(self, node):
        root = Node(self.game, self.args, node)

        for _ in range(self.args["NO_OF_SEARCHES"]):
            node = root
                
            while node.leaf_or_not():
                node = node.search()
              
            is_terminal, value = self.game.know_terminal_value(node.state, node.action)
            value = self.game.get_opponent_value(value)
            
            if not is_terminal:
                
                encoded_state = torch.tensor(self.game.get_encoded_state(node.state)).unsqueeze(0) 
                policy, value = self.model(encoded_state)
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()

                valid_state = np.zeros(self.game.possible_state)
                for move in self.game.get_moves(node.state):
                    valid_state[move] = 1
                policy *= valid_state
                policy /= np.sum(policy)
                value = value.item()
                
                node.expand(policy)
            node.backpropagate(value)
            
        move_probability = np.zeros(self.game.possible_state)
        for children in root.children:
            move_probability[children.action] = children.visits
        move_probability /= np.sum(move_probability)

        return move_probability