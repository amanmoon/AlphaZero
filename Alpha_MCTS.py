import math
import torch
import numpy as np

class Node:

    def __init__(self, game, args, state, parent = None, action = None, prob = 0, visits = 0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action = action
        self.prob = prob

        self.children = []
        self.visits = visits
        self.value = 0

    def leaf_or_not(self):
        return len(self.children) > 0

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
        if child.visits == 0:
            q_value = 0
        else:
            q_value = 1 - (((child.value / child.visits) + 1) / 2)

        return q_value + self.args['EXPLORATION_CONSTANT'] * (math.sqrt(self.visits) / (child.visits + 1)) * child.prob

    def expand(self, policy):

        for move, prob in enumerate(policy):
            if prob > 0:
                child = self.state.copy()
                child = self.game.make_move(child, move, 1)
                if self.args["ADVERSARIAL"]:
                    child = self.game.change_perspective(child, player = -1)

                child = Node(self.game, self.args, child, self, move, prob)
                self.children.append(child)

    def backpropagate(self,state_value):
        self.value += state_value
        self.visits += 1
        if self.args["ADVERSARIAL"]:
            state_value = self.game.get_opponent_value(state_value)
        if self.parent is not None:
            self.parent.backpropagate(state_value)


class Alpha_MCTS:

    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, state):
        root = Node(self.game, self.args, state, visits = 1)

        if self.args["ROOT_RANDOMNESS"]:
            policy, _ = self.model(
                        torch.tensor(self.game.get_encoded_state(state), device = self.model.device
                    ).unsqueeze(0))

            policy = torch.softmax(policy, axis = 1).squeeze(0).cpu().numpy()

            policy = (1 - self.args["DIRICHLET_EPSILON"]) * policy + self.args["DIRICHLET_EPSILON"] * np.random.dirichlet([self.args["DIRICHLET_ALPHA"]] * self.game.possible_state)

            valid_state = self.game.get_valid_moves(state)
            policy *= valid_state
            policy /= np.sum(policy)
            root.expand(policy)

        for _ in range(self.args["NO_OF_SEARCHES"]):
            node = root
            no_moves = 0
            while node.leaf_or_not():
                node = node.search()
                no_moves += 1

            is_terminal, value = self.game.know_terminal_value(node.state, node.action)

            if self.args["ADVERSARIAL"]:
                value = self.game.get_opponent_value(value)

            if not is_terminal:

                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(node.state), device = self.model.device).unsqueeze(0)
                )

                valid_state = self.game.get_valid_moves(node.state)
                policy = torch.softmax(policy, axis = 1).squeeze(0).cpu().numpy().astype(np.float64)

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
