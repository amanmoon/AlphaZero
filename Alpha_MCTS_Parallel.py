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
            q_value = 1 - ((child.value / child.visits) + 1) / 2

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
    def search(self, states, spGames):

        policy, _ = self.model(
                        torch.tensor(self.game.get_encoded_state(states), device = self.model.device
                    ))

        policy = torch.softmax(policy, axis = 1).cpu().numpy()

        if self.args["ROOT_RANDOMNESS"]:
            policy = (1 - self.args["DIRICHLET_EPSILON"]) * policy + self.args["DIRICHLET_EPSILON"] * np.random.dirichlet([self.args["DIRICHLET_ALPHA"]] * self.game.possible_state, size = policy.shape[0])

        for i, spg in enumerate(spGames):
            spg_policy = policy[i]
            valid_state = self.game.get_valid_moves(states[i])
            spg_policy *= valid_state
            spg_policy /= np.sum(spg_policy)

            spg.root = Node(self.game, self.args, states[i], visits = 1)
            spg.root.expand(spg_policy)

        for _ in range(self.args["NO_OF_SEARCHES"]):

            for spg in spGames:
                spg.node = None
                node = spg.root

                while node.leaf_or_not():
                    node = node.search()

                is_terminal, value = self.game.know_terminal_value(node.state, node.action)

                if self.args["ADVERSARIAL"]:
                    value = self.game.get_opponent_value(value)

                if is_terminal:
                    node.backpropagate(value)
                else:
                    spg.node = node

            expandabel_spgs = [mapping_index for mapping_index in range(len(spGames)) if spGames[mapping_index].node is not None]

            if len(expandabel_spgs) > 0:
                states = np.stack([spGames[mapping_index].node.state for mapping_index in expandabel_spgs])
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(states), device = self.model.device)
                )

                policy = torch.softmax(policy, axis = 1).cpu().numpy().astype(np.float64)
                value = value.cpu().numpy()

            for i, mapping_index in enumerate(expandabel_spgs):
                node = spGames[mapping_index].node
                spg_policy, spg_value = policy[i], value[i]
                valid_state = self.game.get_valid_moves(node.state)

                spg_policy *= valid_state
                spg_policy /= np.sum(spg_policy)

                node.expand(spg_policy)
                node.backpropagate(spg_value)
