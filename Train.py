import os
import torch

from Games.ConnectFour.ConnectFour import ConnectFour
from Games.ConnectFour.ConnectFourNN import ResNet
from Alpha_Zero_Parallel import Alpha_Zero

GAME = "ConnectFour"

args = {
    "MODEL_PATH" : os.path.join(os.getcwd(), "Games", GAME, "models_n_optimizers"),
    "SAVE_GAME_PATH" :  os.path.join(os.getcwd(), "Games", GAME, "games"),

    "EXPLORATION_CONSTANT" : 2,

    "TEMPERATURE" : 2,

    "DIRICHLET_EPSILON" : 0.25,
    "DIRICHLET_ALPHA" : 0.3,
    "ROOT_RANDOMNESS": True,

    "ADVERSARIAL" : True,

    "NO_OF_SEARCHES" : 8000,
    "NO_ITERATIONS" : 3,
    "SELF_PLAY_ITERATIONS" : 500,
    "PARALLEL_PROCESS" : 100,
    "EPOCHS" : 4,
    "BATCH_SIZE" : 128,
    "MODEL_CHECK_GAMES" : 25,
    "WIN_RATIO_FOR_SAVING": 0.5,
}


game = ConnectFour()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device, "in use")

model = ResNet(game, 9, 128, device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = 0.0001)

state = game.initialise_state()

alpha_zero = Alpha_Zero(game, args, model, optimizer)

alpha_zero.learn()
