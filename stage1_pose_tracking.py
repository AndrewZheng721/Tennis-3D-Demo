import os
import cv2
import json
import pickle
import argparse

from tqdm import tqdm
from ultralytics import YOLO

from src.common.pose_types import (
    TrackedPlayerPose,
    TrackedFramePose
)

from src.visualization.pose2d_visualizer import (
    Pose2DVisualizer
)

parser = argparse.ArgumentParser(description="第一阶段：姿态追踪脚本")
parser.add_argument("--video", type=str, help="输入视频路径")
parser.add_argument("--out", type=str, help="输出目录路径")

args = parser.parse_args()

VIDEO_PATH = args.video
MODEL_PATH = "weights/yolo26m-pose.pt"
OUTPUT_DIR = args.out


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


def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    model = YOLO(
        MODEL_PATH
    )

    visualizer = Pose2DVisualizer()

    cap = cv2.VideoCapture(
        VIDEO_PATH
    )

    fps = int(
        cap.get(cv2.CAP_PROP_FPS)
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    writer = cv2.VideoWriter(
        f"{OUTPUT_DIR}/tracked_vis.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    tracked_results = []
    json_results = []

    frame_id = 0

    with tqdm(total=total_frames) as pbar:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            result = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False
            )[0]

            players = []

            if (
                result.boxes is not None
                and result.boxes.id is not None
                and result.keypoints is not None
            ):

                boxes = result.boxes.xyxy.cpu().numpy()

                confs = result.boxes.conf.cpu().numpy()

                ids = result.boxes.id.cpu().numpy()

                kpts = result.keypoints.xy.cpu().numpy()

                for box, conf, tid, kp in zip(
                    boxes,
                    confs,
                    ids,
                    kpts
                ):

                    players.append(
                        TrackedPlayerPose(
                            track_id=int(tid),
                            bbox=box,
                            keypoints=kp,
                            confidence=float(conf)
                        )
                    )

            tracked_frame = TrackedFramePose(
                frame_id=frame_id,
                players=players
            )

            tracked_results.append(
                tracked_frame
            )

            json_results.append(
                frame_to_json(
                    tracked_frame
                )
            )

            vis = visualizer.draw(
                frame,
                tracked_frame
            )

            writer.write(
                vis
            )

            frame_id += 1

            pbar.update(1)

    cap.release()

    writer.release()

    with open(
        f"{OUTPUT_DIR}/tracked_pose.pkl",
        "wb"
    ) as f:

        pickle.dump(
            tracked_results,
            f
        )

    with open(
        f"{OUTPUT_DIR}/tracked_pose.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            json_results,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("Saved:")
    print(f"{OUTPUT_DIR}/tracked_pose.pkl")
    print(f"{OUTPUT_DIR}/tracked_pose.json")
    print(f"{OUTPUT_DIR}/tracked_vis.mp4")


if __name__ == "__main__":
    main()
