import random
import os
import copy
import numpy as np

import torch
import torch.nn.functional as F

from tqdm import trange
from Alpha_MCTS_Parallel import Alpha_MCTS
from Arena import Arena

class Colors:
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"


class Alpha_Zero:

    def __init__(self, game, args, model, optimizer):
        self.game = game
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.mcts = Alpha_MCTS(game, args, model)

    def self_play(self):

        return_memory = []
        player = 1
        spGames = [SPG(self.game) for _ in range(self.args["PARALLEL_PROCESS"])]

        while len(spGames) > 0:
            states = np.stack([spg.state for spg in spGames])
            neutral_states = self.game.change_perspective(states, player) if self.args["ADVERSARIAL"] else states
            self.mcts.search(neutral_states, spGames)

            for i in range(len(spGames))[::-1]:

                spg = spGames[i]

                move_probability = np.zeros(self.game.possible_state)

                for children in spg.root.children:
                    move_probability[children.action] = children.visits
                prob = move_probability / np.sum(move_probability)

                spg.memory.append((spg.root.state, prob, player))

                temp_prob = prob ** (1 / self.args["TEMPERATURE"])
                temp_prob /= np.sum(temp_prob)

                move = np.random.choice(self.game.possible_state, p = temp_prob)

                spg.state = self.game.make_move(spg.state, move, player)
                is_terminal, value = self.game.know_terminal_value(spg.state, move)

                if is_terminal:
                    for return_state, return_action_prob, return_player in spg.memory:
                        if self.args["ADVERSARIAL"]:
                            return_value = value if return_player == player else self.game.get_opponent_value(value)
                        else:
                            return_value = value

                        return_memory.append((
                            self.game.get_encoded_state(return_state),
                            return_action_prob,
                            return_value
                        ))
                    del spGames[i]

            if self.args["ADVERSARIAL"]:
                player = self.game.get_opponent(player)
        return return_memory

    def train(self, memory):

        random.shuffle(memory)

        for batch_start in range(0, len(memory), self.args["BATCH_SIZE"]):
            batch_end = batch_start + self.args["BATCH_SIZE"]

            training_memory = memory[batch_start : batch_end]

            state, action_prob, value = zip(*training_memory)

            state, action_prob, value =  np.array(state), np.array(action_prob), np.array(value).reshape(-1, 1)

            state = torch.tensor(state, device = self.model.device, dtype=torch.float32)
            policy_targets = torch.tensor(action_prob, device = self.model.device, dtype=torch.float32)
            value_targets = torch.tensor(value, device = self.model.device, dtype=torch.float32)

            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def learn(self):
        try:
            model_path = os.path.join(self.args["MODEL_PATH"], 'model01.pt')
            optimizer_path = os.path.join(self.args["MODEL_PATH"], 'optimizer01.pt')

            self.model.load_state_dict(torch.load(model_path))
            self.optimizer.load_state_dict(torch.load(optimizer_path))
        except:
            print(Colors.RED + "UNABLE TO LOAD MODEL")
            print(Colors.GREEN + "SETTING UP NEW MODEL..." + Colors.RESET)

        else:
            print(Colors.GREEN + "MODEL FOUND\nLOADING MODEL..." + Colors.RESET)
        finally:
            initial_model = copy.copy(self.model)
            for iteration in range(self.args["NO_ITERATIONS"]):
                memory = []

                print(Colors.BLUE + "\nIteration no: " , iteration + 1, Colors.RESET)

                print(Colors.YELLOW + "Self Play" + Colors.RESET)
                self.model.eval()
                for _ in trange(self.args["SELF_PLAY_ITERATIONS"] // self.args["PARALLEL_PROCESS"]):
                    memory += self.self_play()

                print(Colors.YELLOW + "Training..." + Colors.RESET)
                self.model.train()
                for _ in trange(self.args["EPOCHS"]):
                    self.train(memory)

                print(Colors.YELLOW + "Testing..." + Colors.RESET)
                self.model.eval()
                initial_model.eval()
                wins, draws, defeats = Arena(self.game, self.args, self.model, initial_model)

                print(Colors.GREEN + "Testing Completed" + Colors.WHITE + "\nTrained Model Stats:")
                print(Colors.GREEN, "Wins: ", wins, Colors.RESET, "|", Colors.RED, "Loss: ", defeats, Colors.RESET, "|", Colors.WHITE," Draw: ", draws, Colors.RESET)

                if ((wins) / self.args["MODEL_CHECK_GAMES"]) <= self.args["WIN_RATIO_FOR_SAVING"]:
                    print(Colors.RED, "Rejecting New Model", Colors.RESET)
                    self.model = initial_model
                else:
                    print(Colors.GREEN, "Accepting New Model", Colors.RESET)

                    print(Colors.YELLOW + "Saving Model...")
                    torch.save(self.model.state_dict(), os.path.join(self.args["MODEL_PATH"], "model01.pt"))
                    torch.save(self.optimizer.state_dict(), os.path.join(self.args["MODEL_PATH"], "optimizer01.pt"))
                    print("Saved!" + Colors.RESET)


class SPG:
    def __init__(self, game):
        self.state = game.initialise_state()
        self.memory = list()
        self.root = None
        self.node = None
