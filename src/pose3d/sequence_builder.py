import numpy as np
from collections import defaultdict


def build_sequences(tracked_frames):

    sequences = defaultdict(list)

    for frame in tracked_frames:

        for player in frame.players:

            sequences[
                player.track_id
            ].append(
                player.keypoints
            )

    for track_id in sequences:

        sequences[track_id] = np.array(
            sequences[track_id],
            dtype=np.float32
        )

    return sequences