import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# =========================
# config
# =========================
POSE_PATH = "outputs/pose3d.pkl"
LABEL_PATH = "outputs/action_labels.json"
OUTPUT_CACHE = "dataset/cache"

WINDOW_BEFORE = 20
WINDOW_AFTER = 40


# =========================
# load pose3d
# =========================
def load_pose3d(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# =========================
# load labels
# =========================
def load_labels(path):
    with open(path, "r") as f:
        return json.load(f)


# =========================
# main builder
# =========================
def build_dataset(pose3d_dict, labels):

    samples = []

    for item in tqdm(labels):

        f = item["frame_id"]
        tid = item["track_id"]
        label = item["label"]

        seq = pose3d_dict[tid]

        start = f - WINDOW_BEFORE
        end = f + WINDOW_AFTER

        if start < 0 or end >= len(seq):
            continue

        clip = seq[start:end]

        if clip.shape[0] != (WINDOW_BEFORE + WINDOW_AFTER):
            continue

        samples.append({
            "pose": clip.astype(np.float32),
            "label": label,
            "center_frame": f,
            "track_id": tid
        })

    return samples


# =========================
# save dataset
# =========================
def save_cache(samples):

    os.makedirs(OUTPUT_CACHE, exist_ok=True)

    # train / val split
    # train, val = train_test_split(samples, test_size=0.2, random_state=42)

    # with open(os.path.join(OUTPUT_DIR, "train.pkl"), "wb") as f:
        #pickle.dump(train, f)

    #with open(os.path.join(OUTPUT_DIR, "val.pkl"), "wb") as f:
        # pickle.dump(val, f)

    with open(os.path.join(OUTPUT_CACHE, "tennis01_samples.pkl"), "wb") as f:
        pickle.dump(samples, f)

    # optional json for debug
    def to_json(data):
        return [
            {
                "label": d["label"],
                "center_frame": int(d["center_frame"]),
                "track_id": int(d["track_id"])
            }
            for d in data
        ]

    with open(os.path.join(OUTPUT_CACHE, "tennis01_samples.json"), "w") as f:
        json.dump(to_json(samples), f, indent=4)

    print("\nSaved to:", OUTPUT_CACHE)


# =========================
# run
# =========================
if __name__ == "__main__":

    pose3d_dict = load_pose3d(POSE_PATH)
    labels = load_labels(LABEL_PATH)

    samples = build_dataset(pose3d_dict, labels)
    save_cache(samples)