from Games.TicTacToe.TicTacToe import TicTacToe
from Games.TicTacToe.TicTacToeNN import ResNet
from Alpha_Zero import Alpha_Zero
import torch

game = "TicTacToe"

args = {
    "MODEL_PATH" : f"/home/adrinospy/Programming/Projects/AI ML/general_alpha_zero/Games/{game}/models_n_optimizers/",

    "EXPLORATION_CONSTANT" : 2,
    "TEMPERATURE" : 1,
    "DIRICHLET_EPSILON" : 0.25,
    "DIRICHLET_ALPHA" : 0.3,

    "ROOT_POLICY_RANDOMNESS" : False,
    "ADVERSIRIAL" : True,

    "NO_OF_SEARCHES" : 1000,
    "NO_ITERATIONS" : 200,
    "SELF_PLAY_ITERATIONS" : 100,
    "EPOCHS" : 4,
    "BATCH_SIZE" : 64,
    "MODEL_CHECK_GAMES" : 18
}


game = TicTacToe()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, "in use")

model = ResNet(game, 4, 64, device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay = 0.00001)

state = game.initialise_state()

alpha_zero = Alpha_Zero(game, args, model, optimizer)

alpha_zero.learn()