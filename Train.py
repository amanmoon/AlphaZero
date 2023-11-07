from Games.ConnectFour.ConnectFour import ConnectFour
from Games.ConnectFour.ConnectFourNN import ResNet
from Alpha_Zero_Parallel import Alpha_Zero

import os

import torch

GAME = "ConnectFour"

args = {
    "MODEL_PATH" : os.path.join(os.getcwd(), "Games", GAME, "models_n_optimizers"),

    "EXPLORATION_CONSTANT" : 2,

    "TEMPERATURE" : 1.25,

    "DIRICHLET_EPSILON" : 0.25,
    "DIRICHLET_ALPHA" : 0.3,
    "ROOT_RANDOMNESS": True,

    "ADVERSARIAL" : True,

    "NO_OF_SEARCHES" : 600,
    "NO_ITERATIONS" : 100,
    "SELF_PLAY_ITERATIONS" : 10,
    "PARALLEL_PROCESS" : 5,
    "EPOCHS" : 4,
    "BATCH_SIZE" : 2,
    "MODEL_CHECK_GAMES" : 80,
    
}


game = ConnectFour()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, "in use")

model = ResNet(game, 12, 124, device)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = 0.0001)

state = game.initialise_state()

alpha_zero = Alpha_Zero(game, args, model, optimizer)

alpha_zero.learn()