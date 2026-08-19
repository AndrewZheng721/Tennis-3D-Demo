import os
import pickle
from sklearn.model_selection import train_test_split

CACHE_DIR = "dataset/cache"

all_samples = []

for f in os.listdir(CACHE_DIR):
    if f.endswith(".pkl"):
        with open(os.path.join(CACHE_DIR, f), "rb") as fp:
            all_samples.extend(pickle.load(fp))

print("Total:", len(all_samples))

train, val = train_test_split(all_samples, test_size=0.2, random_state=42)

with open("dataset/train.pkl", "wb") as f:
    pickle.dump(train, f)

with open("dataset/val.pkl", "wb") as f:
    pickle.dump(val, f)