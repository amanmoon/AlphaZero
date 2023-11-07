from Alpha_Zero import Alpha_Zero
from Games.ConnectFour.ConnectFour import ConnectFour
from Games.ConnectFour.ConnectFourNN import ResNet

from tqdm import trange

import shelve
import os

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

def save_games(args, game, model, optimizer):
        try:
            model_path = os.path.join(args["MODEL_PATH"], 'model.pt')
            optimizer_path = os.path.join(args["MODEL_PATH"], 'optimizer.pt')

            model.load_state_dict(torch.load(model_path))
            optimizer.load_state_dict(torch.load(optimizer_path))
        except:
            print(Colors.RED + "UNABLE TO LOAD MODEL")
            print(Colors.GREEN + "SETTING UP NEW MODEL..." + Colors.RESET)
            
        else:
            print(Colors.GREEN + "MODEL FOUND\nLOADING MODEL..." + Colors.RESET)
        finally:

            for iteration in range(args["NO_ITERATIONS"]):
                memory = []
    
                print(Colors.BLUE + "\nIteration no: " , iteration + 1, Colors.RESET)
                
                print(Colors.YELLOW + "Self Play" + Colors.RESET)
                model.eval()
                alpha_zero = Alpha_Zero(game, args, model, optimizer)
                for _ in trange(args["SELF_PLAY_ITERATIONS"]):
                    memory = alpha_zero.self_play()          
                       
                    with shelve.open( os.path.join(args["SAVE_GAME_PATH"], "games_1.pkl"), writeback=True) as db:
                        if "data" in db:
                            existing_data = db["data"]
                            existing_data.extend(memory)
                        else:
                            db["data"] = memory

GAME = "ConnectFour"
args = {
    "MODEL_PATH" : os.path.join(os.getcwd(), "Games", GAME, "models_n_optimizers"),
    "SAVE_GAME_PATH" :  os.path.join(os.getcwd(), "Games", GAME, "games"),

    "EXPLORATION_CONSTANT" : 2,

    "TEMPERATURE" : 1.35,

    "DIRICHLET_EPSILON" : 0.25,
    "DIRICHLET_ALPHA" : 0.3,
    "ROOT_RANDOMNESS": True,

    "ADVERSARIAL" : True,

    "NO_OF_SEARCHES" : 20000,
    "NO_ITERATIONS" : 1,
    "SELF_PLAY_ITERATIONS" : 20,
    "PARALLEL_PROCESS" : 20,
    
}



game = ConnectFour()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, "in use")

model = ResNet(game, 12, 124, device)
model.eval()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = 0.0001)

save_games(args, game, model, optimizer)