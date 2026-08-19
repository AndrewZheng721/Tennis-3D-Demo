"""第三阶段：网球检测 + 轨迹插值。

用法：
    python stage3_ball_detection.py --video data/xxx.mp4 --out outputs/xxx

优先使用网球微调权重：
    weights/yolo5_last.pt
下载地址：
    https://drive.google.com/file/d/1UZwiG1jkWgce9lNhxJ2L0NVjX1vGM05U/view

如果没有专用权重，可以退回通用模型（效果会差一截）：
    python stage3_ball_detection.py --video data/xxx.mp4 --out outputs/xxx \\
        --weights weights/yolo26m.pt --coco-sports-ball
"""

import argparse
import json
import os

import cv2
from tqdm import tqdm

from src.ball.ball_tracker import BallTracker


def parse_args():
    parser = argparse.ArgumentParser(description="网球检测与轨迹追踪")
    parser.add_argument("--video", required=True, help="输入视频路径")
    parser.add_argument("--out", required=True, help="输出目录")
    parser.add_argument(
        "--weights",
        default="weights/yolo5_last.pt",
        help="网球 YOLO 权重，或通用 YOLO 权重",
    )
    parser.add_argument(
        "--coco-sports-ball",
        action="store_true",
        help="使用 COCO 通用 sports ball（类别 32），没有网球专用权重时才开",
    )
    parser.add_argument("--conf", type=float, default=0.15, help="置信度门槛")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="推理分辨率，高位小球建议 1280",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="只跑前 N 帧，调试用",
    )
    parser.add_argument(
        "--trail",
        type=int,
        default=30,
        help="可视化时保留最近多少帧轨迹",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    if not os.path.isfile(args.weights):
        raise FileNotFoundError(
            f"找不到球检测权重: {args.weights}\n"
            "网球专用权重: https://drive.google.com/file/d/1UZwiG1jkWgce9lNhxJ2L0NVjX1vGM05U/view\n"
            "保存为 weights/yolo5_last.pt"
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
    print("weights:", args.weights, "coco_sports_ball:", args.coco_sports_ball)

    tracker = BallTracker(
        args.weights,
        conf=args.conf,
        imgsz=args.imgsz,
        coco_sports_ball=args.coco_sports_ball,
    )

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
    if payload["raw_detect_count"] < 0.2 * max(1, len(raw)):
        print(
            "提示：有效检测不足 20%。高位远景更适合 TrackNetV3，"
            "不要继续死磕通用 YOLO。详见 test_court_ball.py 说明。"
        )


if __name__ == "__main__":
    main()
