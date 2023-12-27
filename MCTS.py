import math
import numpy as np

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
        self.total_value = 0

    def leaf_or_not(self):
        return (len(self.children) > 0 and len(self.expandable_moves) == 0)

    def search(self):
        best_child = None
        best_ucb = -np.inf
        for child in self.children:
            ucb = self.get_ucb(child)
            if best_ucb < ucb:
                best_ucb = ucb
                best_child = child

        return best_child

    def get_ucb(self, child):
        q_value = 1 - ((child.total_value / child.visits) + 1) / 2
        return q_value + self.args["EXPLORATION_CONSTANT"] * math.sqrt(math.log(self.visits) / child.visits)

    def expand(self):
        rand_move = np.random.choice(self.expandable_moves)
        self.expandable_moves.remove(rand_move)

        child = self.game.make_move(self.state.copy(), rand_move, 1)
        child = self.game.change_perspective(child)
        child = Node(self.game, self.args, child, self, rand_move)

        self.children.append(child)

        return child

    def simulate(self):
        is_terminal, value = self.game.know_terminal_value(self.state, self.action)
        value = self.game.get_opponent_value(value)

        if is_terminal:
            return value

        state = self.state.copy()
        player = 1
        while True:
            possible_moves = self.game.get_moves(state)
            rand_move = np.random.choice(possible_moves)
            state = self.game.make_move(state, rand_move, player)
            is_terminal, value = self.game.know_terminal_value(state, rand_move)
            if is_terminal:
                if player == -1:
                    value = self.game.get_opponent_value(value)
                return value
            player = self.game.get_opponent(player)

    def backpropagate(self,value):
        self.total_value += value
        self.visits += 1

        value = self.game.get_opponent_value(value)
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
            while node.leaf_or_not():
                node = node.search()

            is_terminal, value = self.game.know_terminal_value(node.state, node.action)
            value = self.game.get_opponent_value(value)

            if not is_terminal:
                node = node.expand()
                value = node.simulate()

            node.backpropagate(value)

        move_probability = np.zeros(self.game.possible_state)
        for children in root.children:
            move_probability[children.action] = children.visits
        move_probability /= np.sum(move_probability)

        return move_probability
