import shelve
import numpy as np

with shelve.open("data.pkl") as db:
    if "data" in db:
        loaded_data = db["data"]

print(loaded_data)


# with shelve.open("data.pkl", writeback=True) as db:
#     if "data" in db:
#         existing_data = db["data"]
#         existing_data.extend(loaded_data)
#     else:
#         db["data"] = loaded_data

