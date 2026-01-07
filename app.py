import streamlit as st
import os
import numpy as np
import torch
import time

from Games.ConnectFour.ConnectFour import ConnectFour
from Games.ConnectFour.ConnectFourNN import ResNet as ConnectFourResNet
from Games.TicTacToe.TicTacToe import TicTacToe
from Games.TicTacToe.TicTacToeNN import ResNet as TicTacToeResNet
from Alpha_MCTS import Alpha_MCTS

st.set_page_config(page_title="AlphaZero UI", layout="centered", page_icon="🎮")

st.markdown("""
<style>
    div.stButton > button {
        height: 80px;
        font-size: 30px;
        font-weight: bold;
        border-radius: 12px;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #FF4B4B, #FF9090);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(game_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if game_name == "ConnectFour":
        game = ConnectFour()
        model = ConnectFourResNet(game, 9, 128, device)
    else:
        game = TicTacToe()
        model = TicTacToeResNet(game, 9, 128, device)
        
    model.eval()
    model_path = os.path.join(os.getcwd(), "Games", game_name, "models_n_optimizers", "model.pt")
    
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            st.warning(f"Failed to load model from {model_path}: {e}")
    else:
        st.warning(f"Model path does not exist: {model_path}")
        
    return game, model

def init_state(game_name):
    st.session_state.game_name = game_name
    st.session_state.board_state = None
    st.session_state.player = 1
    st.session_state.game_over = False
    st.session_state.winner = None

st.sidebar.title("AlphaZero Play")
game_selection = st.sidebar.selectbox("Select Game", ["ConnectFour", "TicTacToe"])

st.sidebar.markdown("### Hyperparameters")
no_of_searches = st.sidebar.slider("Number of MCTS Searches", min_value=10, max_value=2000, value=600, step=10, help="More searches = stronger but slower AI.")
exploration_constant = st.sidebar.slider("Exploration Constant (C)", min_value=0.1, max_value=5.0, value=1.0, step=0.1, help="Higher values favor exploration.")
temperature = st.sidebar.slider("Temperature", min_value=0.1, max_value=2.0, value=1.0, step=0.1, help="Controls exploration during policy evaluation.")
adversarial = st.sidebar.checkbox("Adversarial (Zero-Sum)", value=True)
root_randomness = st.sidebar.checkbox("Root Randomness (Dirichlet Noise)", value=False)

mcts_args = {
    "ADVERSARIAL": adversarial,
    "ROOT_RANDOMNESS": root_randomness,
    "TEMPERATURE": temperature,
    "NO_OF_SEARCHES": no_of_searches,
    "EXPLORATION_CONSTANT": exploration_constant,
}

if root_randomness:
    mcts_args["DIRICHLET_EPSILON"] = st.sidebar.slider("Dirichlet Epsilon", 0.0, 1.0, 0.25)
    mcts_args["DIRICHLET_ALPHA"] = st.sidebar.slider("Dirichlet Alpha", 0.01, 1.0, 0.3)

if "game_name" not in st.session_state or st.session_state.game_name != game_selection:
    init_state(game_selection)

game, model = load_model(game_selection)
mcts = Alpha_MCTS(game, mcts_args, model)

if st.session_state.board_state is None:
    st.session_state.board_state = game.initialise_state()

st.sidebar.markdown("---")
if st.sidebar.button("Reset Game"):
    init_state(game_selection)
    st.session_state.board_state = game.initialise_state()
    st.rerun()

st.sidebar.markdown("### Rules")
st.sidebar.info(
    "You are Player 1 (playing first).\n"
    "AlphaZero is Player -1.\n\n"
    "For **Tic Tac Toe**: Click on a cell to place your X.\n"
    "\n"
    "For **Connect Four**: Click the ⬇️ button above a column to drop your piece."
)

st.markdown(f"<h1 class='main-header'>{game_selection} vs AlphaZero</h1>", unsafe_allow_html=True)

if st.session_state.game_over:
    if st.session_state.winner == 1:
        st.success("You Won! Amazing job playing against AlphaZero!")
    elif st.session_state.winner == -1:
        st.error("AlphaZero Won! Better luck next time!")
    else:
        st.info("It's a Draw! Well played.")

def trigger_rerun():
    time.sleep(0.1)
    st.rerun()

def check_ai_move():
    if not st.session_state.game_over and st.session_state.player == -1:
        with st.spinner(f"AlphaZero is thinking ({no_of_searches} searches)..."):
            neutral_state = game.change_perspective(st.session_state.board_state, st.session_state.player)
            mcts_probs = mcts.search(neutral_state)
            action = np.argmax(mcts_probs)
            
            st.session_state.board_state = game.make_move(
                st.session_state.board_state.copy(), action, st.session_state.player
            )
            
            # Check terminal
            is_terminal, value = game.know_terminal_value(st.session_state.board_state, action)
            if is_terminal:
                st.session_state.game_over = True
                st.session_state.winner = st.session_state.player if value == 1 else 0
            else:
                st.session_state.player = game.get_opponent(st.session_state.player)
            trigger_rerun()

def make_move(action):
    if not st.session_state.game_over and st.session_state.player == 1:
        valid_moves = game.get_valid_moves(st.session_state.board_state)
        
        if isinstance(valid_moves, np.ndarray) and valid_moves.ndim > 1:
             valid_moves = valid_moves.reshape(-1)

        if valid_moves[action] == 1:
            st.session_state.board_state = game.make_move(
                st.session_state.board_state.copy(), action, st.session_state.player
            )
            
            is_terminal, value = game.know_terminal_value(st.session_state.board_state, action)
            if is_terminal:
                st.session_state.game_over = True
                st.session_state.winner = st.session_state.player if value == 1 else 0
            else:
                st.session_state.player = game.get_opponent(st.session_state.player)
            trigger_rerun()

container = st.container()

with container:
    if game_selection == "TicTacToe":
        state = st.session_state.board_state
        for row in range(3):
            cols = st.columns([1, 1, 1, 1, 1])
            for col in range(3):
                val = state[row, col]
                display_str = "❌" if val == 1 else "⭕" if val == -1 else " "
                
                action = row * 3 + col
                
                with cols[col + 1]: 
                    if st.button(display_str, key=f"btn_{action}", disabled=st.session_state.game_over or val != 0 or st.session_state.player != 1, use_container_width=True):
                        make_move(action)

    elif game_selection == "ConnectFour":
        state = st.session_state.board_state
        
        cols = st.columns(7)
        valid_moves = game.get_valid_moves(state)
        for col in range(7):
            with cols[col]:
                if st.button("⬇️", key=f"drop_{col}", disabled=st.session_state.game_over or valid_moves[col] == 0 or st.session_state.player != 1, use_container_width=True):
                    make_move(col)
                    
        st.markdown("---")
        
        colors = {1: "🔴", -1: "🟡", 0: "⚫"}
        for row in range(6):
            cols = st.columns(7)
            for col in range(7):
                val = state[row, col]
                with cols[col]:
                    st.markdown(f"<div style='text-align:center; font-size:40px;'>{colors[val]}</div>", unsafe_allow_html=True)

if not st.session_state.game_over and st.session_state.player == -1:
    check_ai_move()
