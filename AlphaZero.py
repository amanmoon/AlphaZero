import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import tqdm 
import copy

args = {
    "NO_OF_SEARCHES" : 1000,
    "EXPLORATION_CONSTANT" : 1.42,
    "NO_ITERATIONS" : 1000,
    "SELF_PLAY_ITERATIONS" : 1000,
    "EPOCHS" : 10,
    "BATCH_SIZE" : 5,
    "MODEL_CHECK_GAMES" : 10
}

class Alpha_Zero:
    def __init__(self, game, args, model, optimizer):
        self.game = game
        self.args = args
        self.model = model
        self.optimizer = optimizer

    def self_play(self, model):
        single_game_memory = list()
        state = self.game.initialise_state()
        player = 1
        while True:
            state = self.game.change_perspective(state)
            single_game_memory.append((state, prob, player))
            
            prob = model.search(self.game, self.args, model)
            move = np.argmax(prob)
            state = self.game.make_move(state, move, player)
            is_terminal, value = self.game.know_terminal_value(state, move)
            if is_terminal:
                return_memory = list()
                for return_state, return_action_prob, return_player in single_game_memory(): 
                    return_value = value if return_player == 1 else self.game.get_opponent_value(value)
                    return_memory.append(return_state, return_action_prob, return_value)

                return return_memory
                
            player = self.game.get_opponent(player)
    
    def train(self, memory, model):
        random.shuffle(memory)
        for batch_start in range(0, len(memory), self.args["BATCH_SIZE"]):
            batch_end = min(len(memory), batch_start + self.args["BATCH_SIZE"]) 
        
            training_memory = memory[batch_start : batch_end]
            state, action_prob, value = zip(*training_memory)
            state = torch.tensor(self.game.get_encoded_state(state), dtype = torch.float32)
            target_action_prob = torch.tensor(action_prob, dtype = torch.float32)
            target_value = value
            
            out_policy, out_value = model(state)

            policy_loss = F.cross_entropy(out_policy, target_action_prob)
            value_loss = F.mse_loss(out_value, target_value)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def compare_models(self, initial_model, trained_model):
        first_score = 0
        second_score = 0

        for _ in range(self.args["MODEL_CHECK_GAMES"]):
            state = self.game.initialise_state()
            player = 1
            model = initial_model
            while True:
                state = self.game.change_perspective(state)         
                prob = model.search(self.game, self.args, model)
                move = np.argmax(prob)
                state = self.game.make_move(state, move, player)
                is_terminal, _ = self.game.know_terminal_value(state, move)

                if is_terminal:
                    if model == initial_model:
                        first_score += 1
                    if model == trained_model:
                        second_score += 1

                model = trained_model
                player = self.game.get_opponent(player)
        
        return trained_model if second_score >= first_score else initial_model
    
    def learn(self):
        for iteration in self.args["NO_ITERATIONS"]:
            memory = list()

            model_under_training = copy.copy(self.model)
            
            for _ in tqdm.trange(self.args["SELF_PLAY_ITERATIONS"]):
                memory.append(self.self_play(model_under_training))
                
            for _ in tqdm.trange(self.args["EPOCHS"]):
                self.train(memory, model_under_training)
            
            self.model = self.compare_models(self.model, model_under_training)
            
            torch.save(self.model.state_dict(), f"model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}.pt")
