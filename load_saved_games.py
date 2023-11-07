import shelve
import numpy as np
import os

GAME = "ConnectFour" 
SAVE_GAME_PATH =  os.path.join(os.getcwd(), "Games", GAME, "games", "games_0.pkl")

with shelve.open(SAVE_GAME_PATH) as db:
    if "data" in db:
        loaded_data = db["data"]

