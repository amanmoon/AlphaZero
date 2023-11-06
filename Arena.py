import numpy as np

import torch

from Alpha_MCTS_Parallel import Alpha_MCTS
from Games.ConnectFour.ConnectFour import ConnectFour
from Games.ConnectFour.ConnectFourNN import ResNet

from tqdm import trange   
    
def Arena(game, args, trained_model, untrained_model):

    model_1 = Alpha_MCTS(game, args, trained_model)
    model_2 = Alpha_MCTS(game, args, untrained_model)

    model_1_wins = 0
    model_2_wins = 0
    draw = 0

    for i in trange(args["MODEL_CHECK_GAMES"]):
        
        if i > args["MODEL_CHECK_GAMES"] // 2:
            temp = model_2
            model_2 = model_1
            model_1 = temp
            
        state = game.initialise_state()
        player = 1
        move = np.random.choice(game.possible_state, p = model_1.search(state))
        game.make_move(state, move, player)
        
        while True:
            neutral_state = game.change_perspective(state, player)
            prob = model_1.search(neutral_state)
            move = np.random.choice(game.possible_state, p = prob)
            
            if args["ADVERSARIAL"]:
                player = game.get_opponent(player)
                
            game.make_move(state, move, player)
            is_terminal, value = game.know_terminal_value(state, move)
            
            if is_terminal:
                if value == 0:
                    draw += 1
                    break
                
                elif i > args["MODEL_CHECK_GAMES"] // 2:
                    model_2_wins += 1
                else: 
                    model_1_wins += 1
                break
            
            if args["ADVERSARIAL"]:
                player = game.get_opponent(player)

            prob = model_2.search(state)
            move = np.argmax(prob)
            game.make_move(state, move, player)
            
            is_terminal, value = game.know_terminal_value(state, move)
            
            if is_terminal:
                if value == 0:
                    draw += 1
                    break
                elif i > args["MODEL_CHECK_GAMES"] // 2:
                    model_1_wins += 1
                else:
                    model_2_wins += 1
                break

    return  model_1_wins, draw, model_2_wins


args = {
    "MODEL_PATH" : f"/home/adrinospy/Programming/Projects/AI ML/general_alpha_zero/Games/ConnectFour/models_n_optimizers/",

    "ADVERSARIAL" : True,

    "TEMPERATURE" : 1,

    "NO_OF_SEARCHES" : 20,
    "EXPLORATION_CONSTANT" : 2,
    "MODEL_CHECK_GAMES": 100
}


game = ConnectFour()
device = torch.device("cuda" if torch.cuda.is_available else "cpu")

model1 = ResNet(game, 9, 128, device)
model1.eval()

model2 = ResNet(game, 9, 128, device)
model2.eval()

path = args["MODEL_PATH"] 
    
model1.load_state_dict(torch.load(path + "model.pt"))
# model2.load_state_dict(torch.load(path + "model2.pt"))

print(Arena(game, args, model1, model2))
