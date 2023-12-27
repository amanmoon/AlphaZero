import os
import random
import copy
import numpy as np

import torch
import torch.nn.functional as F

from tqdm import trange
from Alpha_MCTS import Alpha_MCTS
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

        single_game_memory = []
        player = 1
        state = self.game.initialise_state()

        while True:
            neutral_state = self.game.change_perspective(state, player) if self.args["ADVERSARIAL"] else state
            prob = self.mcts.search(neutral_state)

            single_game_memory.append((neutral_state, prob, player))

            temp_prob = prob ** (1 / self.args["TEMPERATURE"])

            temp_prob[temp_prob == 0] = - np.inf

            temp_prob = torch.softmax(torch.tensor(temp_prob), axis = 0).cpu().numpy()

            move = np.random.choice(self.game.possible_state, p = temp_prob)

            state = self.game.make_move(state, move, player)
            is_terminal, value = self.game.know_terminal_value(state, move)

            if is_terminal:
                return_memory = []
                for return_state, return_action_prob, return_player in single_game_memory:
                    if self.args["ADVERSARIAL"]:
                        return_value = value if return_player == player else self.game.get_opponent_value(value)
                    else:
                        return_value = value

                    return_memory.append((
                        self.game.get_encoded_state(return_state),
                        return_action_prob,
                        return_value
                    ))
                return return_memory

            if self.args["ADVERSARIAL"]:
                player = self.game.get_opponent(player)

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
            model_path = os.path.join(self.args["MODEL_PATH"], 'model_non_parallel.pt')
            optimizer_path = os.path.join(self.args["MODEL_PATH"], 'optimizer_non_parallel.pt')

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

                for _ in trange(self.args["SELF_PLAY_ITERATIONS"]):
                    memory += self.self_play()

                print(Colors.YELLOW + "Training..." + Colors.RESET)
                self.model.train()
                for _ in trange(self.args["EPOCHS"]):
                    self.train(memory)

                print(Colors.YELLOW + "Testing..." + Colors.RESET)
                self.model.eval()
                wins, draws, defeats = Arena(self.game, self.args, self.model, initial_model)
                print(Colors.GREEN + "Testing Completed" + Colors.WHITE + "\nTrained Model Stats:")
                print(Colors.GREEN, "Wins: ", wins, Colors.RESET, "|", Colors.RED, "Loss: ", defeats, Colors.RESET, "|", Colors.WHITE," Draw: ", draws, Colors.RESET)

            print(Colors.YELLOW + "Saving Model...")
            torch.save(self.model.state_dict(), os.path.join(self.args["MODEL_PATH"], "model_non_parallel.pt"))
            torch.save(self.optimizer.state_dict(), os.path.join(self.args["MODEL_PATH"], "optimizer_non_parallel.pt"))
            print("Saved!" + Colors.RESET)
