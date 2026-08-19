import os
import subprocess

VIDEO_DIR = "data/videos"


def run(cmd):
    print(" ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main():
    if not os.path.isdir(VIDEO_DIR):
        raise SystemExit(f"找不到 {VIDEO_DIR}")

    for video in os.listdir(VIDEO_DIR):
        if not video.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            continue

        video_path = os.path.join(VIDEO_DIR, video)
        video_name = os.path.splitext(video)[0]
        out_dir = os.path.join("outputs", video_name)

        print("\n====================")
        print("Processing:", video_name)
        print("====================")

        run([
            "python", "stage1_pose_tracking.py",
            "--video", video_path,
            "--out", out_dir,
        ])

        run([
            "python", "stage2_pose3d_pro.py",
            "--input", f"{out_dir}/tracked_pose.pkl",
            "--out", out_dir,
        ])

        run([
            "python", "stage4_court_detection.py",
            "--video", video_path,
            "--out", out_dir,
        ])

        run([
            "python", "stage3_ball_detection.py",
            "--video", video_path,
            "--out", out_dir,
        ])

        print(f"\nStage3 动作标注仍需人工：{video_name}")


if __name__ == "__main__":
    main()
