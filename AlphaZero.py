args = {
    "NO_OF_SEARCHES" : 10,
    "EXPLORATION_CONSTANT" : 1.42,
    "NO_ITERATIONS" : 10,
    "SELF_PLAY_ITERATIONS" : 10,
    "EPOCHS" : 10,
    "BATCH_SIZE" : 5,
    "MODEL_CHECK_GAMES" : 10
}

from Games.TicTacToe.TicTacToe import TicTacToe
from Games.TicTacToe.TicTacToeNN import Resnet
from Alpha_MCTS import Alpha_MCTS 
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import tqdm 
import copy



class Alpha_Zero:
    def __init__(self, game, args, model, optimizer):
        self.game = game
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.mcts = Alpha_MCTS(game, args, model)
        
    def self_play(self):
        
        single_game_memory = list()
        state = self.game.initialise_state()
        player = 1
        while True:
            neutral_state = self.game.change_perspective(state, player) # player added
            prob = self.mcts.search(neutral_state)
            single_game_memory.append((neutral_state, prob, player))
            
            move = np.argmax(prob)
            state = self.game.make_move(state, move, player)
            is_terminal, value = self.game.know_terminal_value(state, move)
            if is_terminal:
                return_memory = list()
                for return_state, return_action_prob, return_player in single_game_memory: 
                    return_value = value if return_player == 1 else self.game.get_opponent_value(value)
                    return_memory.append((self.game.get_encoded_state(return_state), return_action_prob, return_value))

                return return_memory
                
            player = self.game.get_opponent(player)
    
    def train(self, memory):
        random.shuffle(memory)
        for batch_start in range(0, len(memory), self.args["BATCH_SIZE"]):
            batch_end = min(len(memory), batch_start + self.args["BATCH_SIZE"]) 
        
            training_memory = memory[batch_start : batch_end]

            state, action_prob, value = zip(*training_memory)

            encoded_state = torch.tensor(np.array(state), dtype = torch.float32)

            target_action_prob = torch.tensor(action_prob, dtype = torch.float32)

            target_value = torch.tensor(value, dtype = torch.float32)
            
            out_policy, out_value = self.model(encoded_state)

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
            first_model = Alpha_MCTS(self.game, self.args, initial_model)
            second_model = Alpha_MCTS(self.game, self.args, trained_model)
            model = first_model
            while True:
                neutral_state = self.game.change_perspective(state, player)     # player cahnged       
                prob = self.mcts.search(neutral_state)
                move = np.argmax(prob)
                state = self.game.make_move(state, move, player)
                is_terminal, _ = self.game.know_terminal_value(state, move)

                if is_terminal:
                    if model == initial_model:
                        first_score += 1
                    if model == trained_model:
                        second_score += 1
                    break                        
                model = second_model if model == first_model else first_model
                player = self.game.get_opponent(player)
        
        return trained_model if second_score >= first_score else initial_model
    
    def learn(self):
        
        for iteration in range(self.args["NO_ITERATIONS"]):
            memory = list()

            initial_model = copy.copy(self.model)
            
            for _ in tqdm.trange(self.args["SELF_PLAY_ITERATIONS"]):
                memory += self.self_play()
                
            for _ in tqdm.trange(self.args["EPOCHS"]):
                self.train(memory)
            
            self.model = self.compare_models(initial_model, self.model)
            
        torch.save(self.model.state_dict(), f"model_{iteration}.pt")
        torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}.pt")
            

game = TicTacToe()
state = game.initialise_state()
model = Resnet(game, 4, 64)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

alpha_zero = Alpha_Zero(game, args, model, optimizer)

alpha_zero.learn()
