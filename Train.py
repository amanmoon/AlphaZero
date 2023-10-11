from Games.TicTacToe.TicTacToe import TicTacToe
from Games.TicTacToe.TicTacToeNN import ResNet
from Alpha_Zero import Alpha_Zero

import torch

game = "TicTacToe"

args = {
    "PATH_FOR_SAVING" : f"/home/adrinospy/Programming/Projects/AI ML/general_alpha_zero/Games/{game}/models_n_optimizers/",

    "EXPLORATION_CONSTANT" : 3,
    "NO_OF_SEARCHES" : 60,

    "NO_ITERATIONS" : 3,
    "SELF_PLAY_ITERATIONS" : 500,
    "EPOCHS" : 3,
    "BATCH_SIZE" : 64,
    "MODEL_CHECK_GAMES" : 20
}


game = TicTacToe()
model = ResNet(game, 4, 64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


state = game.initialise_state()

alpha_zero = Alpha_Zero(game, args, model, optimizer)

alpha_zero.learn()