from Games.ConnectFour.ConnectFour import ConnectFour
from Games.ConnectFour.ConnectFourNN import ResNet
from Alpha_Zero_Parallel import Alpha_Zero

import os

import torch

GAME = "ConnectFour"

args = {
    "MODEL_PATH" : os.path.join(os.getcwd(), "Games", GAME, "models_n_optimizers"),
    "SAVE_GAME_PATH" :  os.path.join(os.getcwd(), "Games", GAME, "games"),

    "EXPLORATION_CONSTANT" : 2,

    "TEMPERATURE" : 1.25,

    "DIRICHLET_EPSILON" : 0.25,
    "DIRICHLET_ALPHA" : 0.03,
    "ROOT_RANDOMNESS": True,

    "ADVERSARIAL" : True,

    "NO_OF_SEARCHES" : 800,
    "NO_ITERATIONS" : 100,
    "SELF_PLAY_ITERATIONS" : 1000,
    "PARALLEL_PROCESS" : 100,
    "EPOCHS" : 6,
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