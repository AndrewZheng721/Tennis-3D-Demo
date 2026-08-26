"""第四阶段：球场 14 关键点检测。

用法：
    python stage4_court_detection.py --video data/xxx.mp4 --out outputs/xxx

权重请放到：
    weights/court_heatmap.pth
下载地址（TennisCourtDetector 热力图，优先）：
    https://drive.google.com/file/d/1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG/view
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


HEATMAP_URL = "https://drive.google.com/file/d/1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG/view"


def parse_args():
    parser = argparse.ArgumentParser(description="球场关键点检测")
    parser.add_argument("--video", required=True, help="输入视频路径")
    parser.add_argument("--out", required=True, help="输出目录")
    parser.add_argument(
        "--heatmap-weights",
        default="weights/court_heatmap.pth",
        help="TennisCourtDetector 热力图权重（优先）",
    )
    parser.add_argument(
        "--weights",
        default="weights/keypoints_model.pth",
        help="ResNet50 回归权重（仅兜底）",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="抽几帧做中位数，机位固定时 5 帧足够",
    )
    parser.add_argument(
        "--redetect",
        type=int,
        default=30,
        help="每隔多少帧重新跑热力图，中间帧跟随镜头",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    if not os.path.isfile(args.heatmap_weights):
        raise FileNotFoundError(
            f"找不到热力图权重: {args.heatmap_weights}\n"
            f"请下载 {HEATMAP_URL}\n"
            "保存为 weights/court_heatmap.pth"
        )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    detector = CourtLineDetector(
        model_path=args.weights if os.path.isfile(args.weights) else None,
        heatmap_path=args.heatmap_weights,
        device=args.device,
    )
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
    prev_gray = None
    live = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        prev_gray, live = detector.track_frame(
            frame, frame_id, prev_gray, live, redetect_every=args.redetect
        )
        vis = detector.draw(frame, live)
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

    print("quality_ok:", detection.quality_ok, detection.method, detection.quality_reason)
    print("json:", json_path)
    print("video:", vis_path)
    if not detection.quality_ok:
        print(
            "提示：当前画面不太像高位全场机位。"
            "水平机位不要微调这个 ResNet，应换模型。详见 test_court_ball.py 说明。"
        )


if __name__ == "__main__":
    main()
