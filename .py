import shelve
import os

GAME = "ConnectFour"
SAVE_GAME_PATH1 =  os.path.join(os.getcwd(), "Games", GAME, "games", "games_0.pkl.db")
SAVE_GAME_PATH2 =  os.path.join(os.getcwd(), "Games", GAME, "games", "games_1.pkl.db")

with shelve.open(SAVE_GAME_PATH2) as db:
    if "data" in db:
        memory = db["data"]
print(memory)
# with shelve.open( os.path.join(SAVE_GAME_PATH2), writeback=True) as db:
#     if "data" in db:
#         existing_data = db["data"]
#         existing_data.extend(memory)
        
