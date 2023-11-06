from Games.TicTacToe.TicTacToe import TicTacToe
from Games.TicTacToe.TicTacToeNN import ResNet
from Alpha_Zero_Parallel import Alpha_Zero

import torch

args = {
    "MODEL_PATH" : f"/home/adrinospy/Programming/Projects/AI ML/general_alpha_zero/Games/TicTacToe/models_n_optimizers/",

    "EXPLORATION_CONSTANT" : 2,
    "TEMPERATURE" : 1.25,
    "DIRICHLET_EPSILON" : 0.25,
    "DIRICHLET_ALPHA" : 0.3,

    "ADVERSARIAL" : True,

    "NO_OF_SEARCHES" : 1200,
    "NO_ITERATIONS" : 3,
    "SELF_PLAY_ITERATIONS" : 500,
    "PARALLEL_PROCESS" : 100,
    "EPOCHS" : 4,
    "BATCH_SIZE" : 100,
    "MODEL_CHECK_GAMES" : 80,
    
    "TRAIN": True
}


game = TicTacToe()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, "in use")

model = ResNet(game, 9, 128, device)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = 0.0001)

state = game.initialise_state()

alpha_zero = Alpha_Zero(game, args, model, optimizer)

alpha_zero.learn()