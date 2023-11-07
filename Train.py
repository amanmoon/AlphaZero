from Games.ConnectFour.ConnectFour import ConnectFour
from Games.ConnectFour.ConnectFourNN import ResNet
from Alpha_Zero_Parallel import Alpha_Zero

import torch

args = {
    "MODEL_PATH" : f"/home/adrinospy/Programming/Projects/AI ML/general_alpha_zero/Games/ConnectFour/models_n_optimizers/",

    "EXPLORATION_CONSTANT" : 2,

    "TEMPERATURE" : 1.75,

    "DIRICHLET_EPSILON" : 0.25,
    "DIRICHLET_ALPHA" : 0.3,
    "ROOT_RANDOMNESS": False,

    "ADVERSARIAL" : True,

    "NO_OF_SEARCHES" : 2,
    "NO_ITERATIONS" : 80,
    "SELF_PLAY_ITERATIONS" : 100,
    "PARALLEL_PROCESS" : 50,
    "EPOCHS" : 4,
    "BATCH_SIZE" : 10,
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