"""联合测试：球场关键点 + 网球轨迹。

用法：
    python test_court_ball.py --video 你的视频.mp4 --out outputs/试跑1

会生成：
    court_keypoints.json
    ball_tracking.json
    overlay.mp4          球场线 + 球轨迹叠在同一段视频上
    overlay_first.jpg    第一帧截图，方便快速看球场点准不准

权重（放到 Tennis-3D-Demo/weights/）：
    keypoints_model.pth
        https://drive.google.com/file/d/1QrTOF1ToQ4plsSZbkBs3zOLkVt3MBlta/view
    yolo5_last.pt
        https://drive.google.com/file/d/1UZwiG1jkWgce9lNhxJ2L0NVjX1vGM05U/view

关于机位和要不要换模型（先看测试结果再决定）
----------------------------------------
球场：
    当前方案 = tennis_analysis 的 ResNet50 回归 + 白线吸附 + 单应性。
    适合：高位、从底线后方能看到整片场的转播机位。
    不适合：水平机位、只拍近端半场、大幅摇镜。
    不要微调这个 ResNet：训练数据全是高位全场，224×224 回归精度也到头了。
    高位如果角点仍飘：下一步换 yastrebksv/TennisCourtDetector（热力图，640×360）。
    水平机位：必须另训或另找模型，和现在这套不是同一任务。

球：
    当前方案 = 网球微调 YOLO + 时间插值。
    比本仓库原来的 yolo26m sports ball 更针对网球。
    高位远景、运动模糊仍会漏。若 raw 检测率 < 20%，不要继续微调 YOLOv5，
    应换 TrackNet / TrackNetV3（多帧热力图，网球轨迹更稳）。
    水平近景球很大，YOLO 通常够用。
"""

import argparse
import json
import os

import cv2
from tqdm import tqdm

from src.ball.ball_tracker import BallTracker
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
        "--court-weights",
        default="weights/keypoints_model.pth",
    )
    parser.add_argument(
        "--ball-weights",
        default="weights/yolo5_last.pt",
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
        if not os.path.isfile(args.court_weights):
            raise FileNotFoundError(
                f"找不到 {args.court_weights}，请先下载 keypoints_model.pth"
            )
        court_detector = CourtLineDetector(args.court_weights)
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
        print("court quality_ok:", detection.quality_ok, detection.quality_reason)
        if not detection.quality_ok:
            print("球场质量告警：水平/半场机位不要微调当前 ResNet，应换模型。")
    else:
        court_detector = None

    filled = None
    shot_frames = []
    if not args.skip_ball:
        if not os.path.isfile(args.ball_weights):
            raise FileNotFoundError(
                f"找不到 {args.ball_weights}"
            )
        ball_tracker = BallTracker(
            args.ball_weights,
            conf=args.conf,
            imgsz=args.imgsz,
            coco_sports_ball=args.coco_sports_ball,
        )
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
        payload["raw_detect_count"] = sum(1 for d in raw if 1 in d)
        with open(os.path.join(args.out, "ball_tracking.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print("ball raw detections:", payload["raw_detect_count"], "/", len(raw))
        print("shot_frames:", shot_frames)
        if payload["raw_detect_count"] < 0.2 * max(1, len(raw)):
            print("球检测率偏低：高位远景优先换 TrackNetV3，而不是继续微调 YOLOv5。")
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
    with tqdm(total=total, desc="write overlay") as pbar:
        while frame_id < total:
            ok, frame = cap.read()
            if not ok:
                break
            vis = frame
            if detection is not None:
                vis = court_detector.draw(vis, detection)
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
