

def frame_to_json(frame_result):

    result = {
        "frame_id": frame_result.frame_id,
        "players": []
    }

    for player in frame_result.players:

        result["players"].append(
            {
                "track_id": player.track_id,
                "bbox": player.bbox.tolist(),
                "confidence": player.confidence,
                "keypoints": player.keypoints.tolist()
            }
        )

    return result