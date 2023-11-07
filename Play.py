from Games.ConnectFour.ConnectFour import ConnectFour
from Games.ConnectFour.ConnectFourNN import ResNet
from Alpha_MCTS import Alpha_MCTS

import numpy as np

import torch



class Colors:
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    
    
args = {
    "MODEL_PATH" : f"/home/adrinospy/Programming/Projects/AI ML/general_alpha_zero/Games/ConnectFour/models_n_optimizers/",

    "ADVERSARIAL" : True,

    "TEMPERATURE" : 1.25,

    "NO_OF_SEARCHES" : 1200,
    "EXPLORATION_CONSTANT" : 2,
    
    "ROOT_RANDOMNESS": True
}


game = ConnectFour()
device = torch.device("cuda" if torch.cuda.is_available else "cpu")

model = ResNet(game, 12, 124, device)
model.eval()

path = args["MODEL_PATH"] + "model.pt"

try:
    model.load_state_dict(torch.load(path))
    print(Colors.GREEN, "Model Found\n Model Successfully Loaded", Colors.RESET)

except:
    print(Colors.RED, "Model Not Found!!!", Colors.RESET)
    
finally:
    mcts = Alpha_MCTS(game, args, model)
    
    state = game.initialise_state()
    player = -1
    
    while True:
        print(state)
            
        if player == 1:
            valid_moves = game.get_valid_moves(state)
            print("valid_moves", [i for i in range(game.possible_state) if valid_moves[i] == 1])
            action = int(input(f"{player}:"))
    
            if valid_moves[action] == 0:
                print("action not valid")
                continue
                
        else:
            neutral_state = game.change_perspective(state, player)
            mcts_probs = mcts.search(neutral_state)
            print(mcts_probs)
            action = np.argmax(mcts_probs)
             
        state = game.make_move(state, action, player)
        
        is_terminal, value = game.know_terminal_value(state, action)
        
        if is_terminal:
            print(state)
            if value == 1:
                print(player, "won")
            else:
                print("draw")
            break
    
        player = game.get_opponent(player)