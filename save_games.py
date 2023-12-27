import os
import shelve

import torch

from tqdm import trange
from Alpha_Zero_Parallel import Alpha_Zero
from Games.ConnectFour.ConnectFour import ConnectFour
from Games.ConnectFour.ConnectFourNN import ResNet


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
            for _ in trange(args["SELF_PLAY_ITERATIONS"] // args["PARALLEL_PROCESS"]):
                memory = alpha_zero.self_play()

            with shelve.open( os.path.join(args["SAVE_GAME_PATH"],"games_5.pkl"), writeback=True) as db:
                if "data" in db:
                    existing_data = db["data"]
                    existing_data.extend(memory)
                else:
                    db["data"] = memory

GAME = "ConnectFour"
args = {
    "MODEL_PATH" : os.path.join(os.getcwd(), "Games", GAME, "models_n_optimizers"),
    "SAVE_GAME_PATH" :  os.path.join(os.getcwd(), "Games", GAME, "games"),

    "EXPLORATION_CONSTANT" : 2.25,

    "TEMPERATURE" : 1.75,

    "DIRICHLET_EPSILON" : 0.25,
    "DIRICHLET_ALPHA" : 0.3,
    "ROOT_RANDOMNESS": True,

    "ADVERSARIAL" : True,

    "NO_OF_SEARCHES" : 12000,
    "NO_ITERATIONS" : 100,
    "SELF_PLAY_ITERATIONS" : 100,
    "PARALLEL_PROCESS" : 50,
}

game = ConnectFour()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, "in use")

model = ResNet(game, 9, 128, device)
model.eval()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = 0.0001)

save_games(args, game, model, optimizer)
