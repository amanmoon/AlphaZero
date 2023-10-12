from Alpha_MCTS import Alpha_MCTS 
import numpy as np
import random

import torch
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
        
        single_game_memory = []
        player = 1
        state = self.game.initialise_state()
        
        while True:
            neutral_state = self.game.change_perspective(state, player)
            prob = self.mcts.search(neutral_state)
            
            single_game_memory.append((neutral_state, prob, player))
            
            move = np.random.choice(self.game.possible_state, p = prob)
            
            state = self.game.make_move(state, move, player)
            is_terminal, value = self.game.know_terminal_value(state, move)
            
            if is_terminal:
                return_memory = []
                for return_state, return_action_prob, return_player in single_game_memory:
                    return_value = value if return_player == player else self.game.get_opponent_value(value)
                    return_memory.append((
                        self.game.get_encoded_state(return_state), 
                        return_action_prob, 
                        return_value
                    ))
                return return_memory
                
            player = self.game.get_opponent(player)
    
    def train(self, memory):
        
        random.shuffle(memory)
        total_loss = 0
        
        for batch_start in range(0, len(memory), self.args["BATCH_SIZE"]):
            batch_end = min(len(memory) - 1, batch_start + self.args["BATCH_SIZE"])    
                    
            training_memory = memory[batch_start : batch_end]

            state, action_prob, value = zip(*training_memory)

            state, action_prob, value =  np.array(state), np.array(action_prob), np.array(value).reshape(-1, 1)
            
            state = torch.tensor(state, dtype=torch.float32)
            policy_targets = torch.tensor(action_prob, dtype=torch.float32)
            value_targets = torch.tensor(value, dtype=torch.float32)
            
            out_policy, out_value = self.model(state)
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss
            
            total_loss += loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        avg_loss = total_loss / (batch_start / self.args["BATCH_SIZE"])
        
        return avg_loss

    def compare_models(self, initial_model):
        self.model.eval()
        initial_model.eval()
        trained_MCTS = Alpha_MCTS(self.game, self.args, self.model)
        untrained_MCTS = Alpha_MCTS(self.game, self.args, initial_model)
        
        trained_score = 0
        untrained_score = 0
        draw = 0
        
        for k in range(2):
            
            MCTS = trained_MCTS if k == 0 else untrained_MCTS
            
            for i in range(self.args["MODEL_CHECK_GAMES"] // 2):
                state = self.game.initialise_state()
                state = self.game.make_move(state, i // 9, 1)
                player = 1
                while True:
                    neutral_state = self.game.change_perspective(state)
                    out = MCTS.search(neutral_state)
                    move = np.argmax(out)
                    state = self.game.make_move(state, move, player)
                    is_terminal, value = self.game.know_terminal_value(state, move)

                    if is_terminal:
                        if MCTS == trained_MCTS and value == 1 :
                            trained_score += 1
                        elif MCTS == untrained_MCTS and value == 1:
                            untrained_score += 1
                        elif value == 0:
                            draw += 1
                        break
                    MCTS = trained_MCTS if MCTS == untrained_MCTS else untrained_MCTS
                    player = self.game.get_opponent(player)

        return (self.model, trained_score, draw, untrained_score) if trained_score >= untrained_score else (initial_model, trained_score, draw, untrained_score)
        
    def learn(self):
        try:
            path = self.args["PATH_FOR_SAVING"] + 'model.pt'
            self.model.load_state_dict(torch.load(path))
        
        except:
            print("UNABLE TO LOAD MODEL\nSETTING UP NEW MODEL...")
            
        else:
            print("MODEL FOUND\nLOADING MODEL...")
        finally:
            initial_model = copy.copy(self.model)
            for iteration in range(self.args["NO_ITERATIONS"]):
                memory = []
    
    
                print("\niteration no: " , iteration + 1)
                print("Self Play")
                
                self.model.eval()
                for _ in tqdm.trange(self.args["SELF_PLAY_ITERATIONS"]):
                    memory += self.self_play()

                print("Training...")
                
                self.model.train()
                for _ in tqdm.trange(self.args["EPOCHS"]):
                    loss= self.train(memory)
                print("Loss: ", loss.squeeze(0).item())
                
            print("Testing...")
            self.model.eval()
            initial_model.eval()
            self.model, wins, draws, defeats  = self.compare_models(initial_model)
            print("Testing Completed\nTrained Model Stats:")
            print("Wins: ", wins,"| Loss: ", defeats, "| Draw: ", draws)

        print("Saving Model...")
        torch.save(self.model.state_dict(), self.args["PATH_FOR_SAVING"] + "model.pt")
        torch.save(self.optimizer.state_dict(), self.args["PATH_FOR_SAVING"] + "optimizer.pt")
        print("Saved!")