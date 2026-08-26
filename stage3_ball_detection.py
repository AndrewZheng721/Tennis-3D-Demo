"""第三阶段：网球检测 + 轨迹插值。

用法：
    python stage3_ball_detection.py --video data/xxx.mp4 --out outputs/xxx

默认 TrackNet（高位小球）：
    weights/tracknet.pth
    https://huggingface.co/vishnushenoy09/tracknet-v1-tennis/resolve/main/tracknet_weights.pth

YOLO 兜底：
    python stage3_ball_detection.py --video data/xxx.mp4 --out outputs/xxx --backend yolo --weights weights/yolo5_last.pt
"""

import argparse
import json
import os

import cv2
from tqdm import tqdm

from src.ball.tracknet_tracker import create_ball_tracker, default_ball_weights


def parse_args():
    parser = argparse.ArgumentParser(description="网球检测与轨迹追踪")
    parser.add_argument("--video", required=True, help="输入视频路径")
    parser.add_argument("--out", required=True, help="输出目录")
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "tracknet", "yolo"],
        help="auto：按权重文件名选择；高位小球用 tracknet",
    )
    parser.add_argument("--weights", default=None, help="TrackNet .pth 或 YOLO .pt")
    parser.add_argument(
        "--coco-sports-ball",
        action="store_true",
        help="使用 COCO 通用 sports ball（仅 YOLO）",
    )
    parser.add_argument("--conf", type=float, default=0.15, help="YOLO 置信度门槛")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO 推理分辨率")
    parser.add_argument("--max-frames", type=int, default=None, help="只跑前 N 帧")
    parser.add_argument("--trail", type=int, default=30, help="可视化轨迹长度")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    weights = args.weights
    backend = args.backend
    if not weights:
        weights, guessed = default_ball_weights()
        if backend == "auto":
            backend = guessed

    if not os.path.isfile(weights):
        raise FileNotFoundError(
            f"找不到球检测权重: {weights}\n"
            "TrackNet V1: https://huggingface.co/vishnushenoy09/tracknet-v1-tennis/resolve/main/tracknet_weights.pth\n"
            "保存为 weights/tracknet.pth\n"
            "V4 PyTorch 权重保存为 weights/tracknet_v4.pth"
        )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames is not None:
        total = min(total, args.max_frames)

    print("video:", args.video)
    print("size:", width, height, "fps:", fps, "frames:", total)
    print("backend:", backend, "weights:", weights)

    tracker = create_ball_tracker(
        weights,
        backend=backend,
        conf=args.conf,
        imgsz=args.imgsz,
        coco_sports_ball=args.coco_sports_ball,
    )
    tracker.reset()

    raw = []
    with tqdm(total=total, desc="detect ball") as pbar:
        frame_id = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames is not None and frame_id >= args.max_frames:
                break
            raw.append(tracker.detect_frame(frame))
            frame_id += 1
            pbar.update(1)

    cap.release()

    filled = tracker.interpolate_ball_positions(raw)
    shot_frames = tracker.get_ball_shot_frames(filled, fps=fps)
    payload = tracker.to_json(filled, shot_frames, fps=fps)
    payload["video"] = args.video
    payload["backend"] = getattr(tracker, "kind", backend)
    payload["raw_detect_count"] = sum(1 for d in raw if 1 in d)
    payload["filled_count"] = sum(1 for d in filled if 1 in d)

    json_path = os.path.join(args.out, "ball_tracking.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    vis_path = os.path.join(args.out, "ball_vis.mp4")
    cap = cv2.VideoCapture(args.video)
    writer = cv2.VideoWriter(
        vis_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    trail = []
    shot_set = set(shot_frames)
    frame_id = 0
    with tqdm(total=len(filled), desc="write vis") as pbar:
        while True:
            ok, frame = cap.read()
            if not ok or frame_id >= len(filled):
                break
            box = filled[frame_id].get(1)
            if box:
                cx = int((box[0] + box[2]) / 2)
                cy = int((box[1] + box[3]) / 2)
                trail.append((cx, cy))
                if len(trail) > args.trail:
                    trail = trail[-args.trail :]
            vis = tracker.draw_frame(
                frame,
                box,
                trail,
                frame_id,
                is_shot=frame_id in shot_set,
            )
            writer.write(vis)
            frame_id += 1
            pbar.update(1)

    cap.release()
    writer.release()

    print("raw detections:", payload["raw_detect_count"], "/", len(raw))
    print("shot_frames:", shot_frames)
    print("json:", json_path)
    print("video:", vis_path)


if __name__ == "__main__":
    main()
