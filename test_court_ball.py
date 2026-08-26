"""联合测试：球场关键点 + 网球轨迹。

用法：
    python test_court_ball.py --video 你的视频.mp4 --out outputs/试跑1

会生成：
    court_keypoints.json
    ball_tracking.json
    overlay.mp4          球场线 + 球轨迹叠在同一段视频上
    overlay_first.jpg    第一帧截图，方便快速看球场点准不准

权重（放到 Tennis-3D-Demo/weights/）：
    court_heatmap.pth     球场热力图
        https://drive.google.com/file/d/1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG/view
    tracknet.pth          网球 TrackNet（默认，高位小球）
        https://huggingface.co/vishnushenoy09/tracknet-v1-tennis/resolve/main/tracknet_weights.pth
    tracknet_v4.pth       可选，V4 Type A 的 PyTorch 权重
    yolo5_last.pt         YOLO 兜底
"""

import argparse
import json
import os

import cv2
from tqdm import tqdm

from src.ball.tracknet_tracker import create_ball_tracker, default_ball_weights
from src.court.court_line_detector import (
    CourtLineDetector,
    choose_sample_ids,
    read_frame_at,
)


def parse_args():
    parser = argparse.ArgumentParser(description="球场+网球联合测试")
    parser.add_argument("--video", required=True, help="测试视频路径")
    parser.add_argument("--out", required=True, help="输出目录")
    parser.add_argument(
        "--heatmap-weights",
        default="weights/court_heatmap.pth",
        help="TennisCourtDetector 热力图权重",
    )
    parser.add_argument(
        "--court-weights",
        default="weights/keypoints_model.pth",
    )
    parser.add_argument("--ball-weights", default=None)
    parser.add_argument(
        "--ball-backend",
        default="auto",
        choices=["auto", "tracknet", "yolo"],
    )
    parser.add_argument(
        "--coco-sports-ball",
        action="store_true",
        help="没有网球专用权重时，用通用 YOLO 的 sports ball",
    )
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--court-samples", type=int, default=5)
    parser.add_argument(
        "--court-redetect",
        type=int,
        default=30,
        help="每隔多少帧重新跑一次球场热力图，中间帧跟镜头",
    )
    parser.add_argument("--skip-ball", action="store_true", help="只测球场")
    parser.add_argument("--skip-court", action="store_true", help="只测球")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

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
    print("size:", width, "x", height, "fps:", fps, "frames:", total)

    detection = None
    if not args.skip_court:
        if not os.path.isfile(args.heatmap_weights):
            raise FileNotFoundError(
                f"找不到 {args.heatmap_weights}\n"
                "请下载 https://drive.google.com/file/d/1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG/view\n"
                "保存为 weights/court_heatmap.pth"
            )
        court_detector = CourtLineDetector(
            model_path=args.court_weights if os.path.isfile(args.court_weights) else None,
            heatmap_path=args.heatmap_weights,
        )
        sample_ids = choose_sample_ids(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), args.court_samples)
        print("court sample frames:", sample_ids)
        detection = court_detector.predict_video_sampled(
            read_frame_fn=lambda fid: read_frame_at(cap, fid),
            sample_ids=sample_ids,
        )
        court_json = {
            "video": args.video,
            "width": width,
            "height": height,
            "fps": float(fps),
            "sample_frame_ids": sample_ids,
            "detection": detection.to_json(),
        }
        with open(os.path.join(args.out, "court_keypoints.json"), "w", encoding="utf-8") as f:
            json.dump(court_json, f, ensure_ascii=False, indent=2)
        print("court:", detection.method, detection.quality_ok, detection.quality_reason)
        if not detection.quality_ok:
            print("球场质量告警：水平/半场机位不要微调当前 ResNet，应换模型。")
    else:
        court_detector = None

    filled = None
    shot_frames = []
    if not args.skip_ball:
        ball_weights = args.ball_weights
        ball_backend = args.ball_backend
        if not ball_weights:
            ball_weights, guessed = default_ball_weights()
            if ball_backend == "auto":
                ball_backend = guessed
        if not os.path.isfile(ball_weights):
            raise FileNotFoundError(
                f"找不到 {ball_weights}\n"
                "TrackNet: https://huggingface.co/vishnushenoy09/tracknet-v1-tennis/resolve/main/tracknet_weights.pth\n"
                "保存为 weights/tracknet.pth"
            )
        print("ball:", ball_backend, ball_weights)
        ball_tracker = create_ball_tracker(
            ball_weights,
            backend=ball_backend,
            conf=args.conf,
            imgsz=args.imgsz,
            coco_sports_ball=args.coco_sports_ball,
        )
        ball_tracker.reset()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        raw = []
        with tqdm(total=total, desc="detect ball") as pbar:
            n = 0
            while n < total:
                ok, frame = cap.read()
                if not ok:
                    break
                raw.append(ball_tracker.detect_frame(frame))
                n += 1
                pbar.update(1)
        filled = ball_tracker.interpolate_ball_positions(raw)
        shot_frames = ball_tracker.get_ball_shot_frames(filled, fps=fps)
        payload = ball_tracker.to_json(filled, shot_frames, fps=fps)
        payload["video"] = args.video
        payload["backend"] = getattr(ball_tracker, "kind", ball_backend)
        payload["raw_detect_count"] = sum(1 for d in raw if 1 in d)
        with open(os.path.join(args.out, "ball_tracking.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print("ball raw detections:", payload["raw_detect_count"], "/", len(raw))
        print("shot_frames:", shot_frames)
    else:
        ball_tracker = None

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    overlay_path = os.path.join(args.out, "overlay.mp4")
    writer = cv2.VideoWriter(
        overlay_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    trail = []
    shot_set = set(shot_frames)
    first_saved = False
    frame_id = 0
    prev_gray = None
    live = None
    with tqdm(total=total, desc="write overlay") as pbar:
        while frame_id < total:
            ok, frame = cap.read()
            if not ok:
                break
            vis = frame
            if court_detector is not None:
                prev_gray, live = court_detector.track_frame(
                    frame,
                    frame_id,
                    prev_gray,
                    live,
                    redetect_every=args.court_redetect,
                )
                vis = court_detector.draw(vis, live)
            box = None
            if filled is not None and frame_id < len(filled):
                box = filled[frame_id].get(1)
                if box:
                    trail.append(
                        (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
                    )
                    trail = trail[-30:]
                vis = ball_tracker.draw_frame(
                    vis, box, trail, frame_id, is_shot=frame_id in shot_set
                )
            else:
                cv2.putText(
                    vis,
                    f"Frame: {frame_id}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
            if not first_saved:
                cv2.imwrite(os.path.join(args.out, "overlay_first.jpg"), vis)
                first_saved = True
            writer.write(vis)
            frame_id += 1
            pbar.update(1)

    cap.release()
    writer.release()
    print("overlay:", overlay_path)
    print("preview:", os.path.join(args.out, "overlay_first.jpg"))


if __name__ == "__main__":
    main()
