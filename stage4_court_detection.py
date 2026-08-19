"""第四阶段：球场 14 关键点检测。

用法：
    python stage4_court_detection.py --video data/xxx.mp4 --out outputs/xxx

权重请放到：
    weights/keypoints_model.pth
下载地址（tennis_analysis 官方预训练）：
    https://drive.google.com/file/d/1QrTOF1ToQ4plsSZbkBs3zOLkVt3MBlta/view
"""

import argparse
import json
import os

import cv2

from src.court.court_line_detector import (
    CourtLineDetector,
    choose_sample_ids,
    read_frame_at,
)


def parse_args():
    parser = argparse.ArgumentParser(description="球场关键点检测")
    parser.add_argument("--video", required=True, help="输入视频路径")
    parser.add_argument("--out", required=True, help="输出目录")
    parser.add_argument(
        "--weights",
        default="weights/keypoints_model.pth",
        help="ResNet50 球场关键点权重",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="抽几帧做中位数，机位固定时 5 帧足够",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="cuda 或 cpu，默认自动选",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    if not os.path.isfile(args.weights):
        raise FileNotFoundError(
            f"找不到球场权重: {args.weights}\n"
            "请从 https://drive.google.com/file/d/1QrTOF1ToQ4plsSZbkBs3zOLkVt3MBlta/view 下载，\n"
            "保存为 weights/keypoints_model.pth"
        )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    detector = CourtLineDetector(args.weights, device=args.device)
    sample_ids = choose_sample_ids(total, args.samples)
    print("video:", args.video)
    print("size:", width, height, "fps:", fps, "frames:", total)
    print("sample frames:", sample_ids)

    detection = detector.predict_video_sampled(
        read_frame_fn=lambda fid: read_frame_at(cap, fid),
        sample_ids=sample_ids,
    )

    payload = {
        "video": args.video,
        "width": width,
        "height": height,
        "fps": float(fps),
        "sample_frame_ids": sample_ids,
        "detection": detection.to_json(),
    }
    json_path = os.path.join(args.out, "court_keypoints.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    vis_path = os.path.join(args.out, "court_vis.mp4")
    writer = cv2.VideoWriter(
        vis_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        vis = detector.draw(frame, detection)
        cv2.putText(
            vis,
            f"Frame: {frame_id}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        writer.write(vis)
        frame_id += 1
        if frame_id % 200 == 0:
            print("wrote", frame_id)

    cap.release()
    writer.release()

    print("quality_ok:", detection.quality_ok, detection.quality_reason)
    print("json:", json_path)
    print("video:", vis_path)
    if not detection.quality_ok:
        print(
            "提示：当前画面不太像高位全场机位。"
            "水平机位不要微调这个 ResNet，应换模型。详见 test_court_ball.py 说明。"
        )


if __name__ == "__main__":
    main()
