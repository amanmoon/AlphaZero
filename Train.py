from Games.TicTacToe.TicTacToe import TicTacToe
from Games.TicTacToe.TicTacToeNN import ResNet
from Alpha_Zero import Alpha_Zero
from Alpha_MCTS import Alpha_MCTS
import torch
import numpy as np

game = "TicTacToe"

args = {
    "PATH_FOR_SAVING" : f"/home/adrinospy/Programming/Projects/AI ML/general_alpha_zero/Games/{game}/models_n_optimizers/",

    "EXPLORATION_CONSTANT" : 2,
    "NO_OF_SEARCHES" : 60,

    "NO_ITERATIONS" : 3,
    "SELF_PLAY_ITERATIONS" : 500,
    "EPOCHS" : 4,
    "BATCH_SIZE" : 64,
    "MODEL_CHECK_GAMES" : 20
}


game = TicTacToe()
model = ResNet(game, 4, 64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

state = game.initialise_state()

alpha_zero = Alpha_Zero(game, args, model, optimizer)

alpha_zero.learn()

game = "TicTacToe"

tictactoe = TicTacToe()

model.eval()
mcts = Alpha_MCTS(tictactoe, args, model)

state = tictactoe.initialise_state()

player = 1

while True:
    print(state)
        
    if player == 1:
        valid_moves = tictactoe.get_valid_moves(state)
        print("valid_moves", [i for i in range(tictactoe.possible_state) if valid_moves[i] == 1])
        action = int(input(f"{player}:"))

        if valid_moves[action] == 0:
            print("action not valid")
            continue
            
    else:
        neutral_state = tictactoe.change_perspective(state, player)
        mcts_probs = mcts.search(neutral_state)
        action = np.argmax(mcts_probs)
        
    state = tictactoe.make_move(state, action, player)
    
    is_terminal, value = tictactoe.know_terminal_value(state, action)
    
    if is_terminal:
        print(state)
        if value == 1:
            print(player, "won")
        else:
            print("draw")
        break
        
    player = tictactoe.get_opponent(player)