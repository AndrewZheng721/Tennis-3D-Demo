import os
import subprocess

VIDEO_DIR = "data/videos"

for video in os.listdir(VIDEO_DIR):

    if not video.endswith(".mp4"):
        continue

    video_path = os.path.join(VIDEO_DIR, video)
    video_name = video.replace(".mp4", "")

    print("\n====================")
    print("Processing:", video_name)
    print("====================")

    # stage1
    subprocess.run([
        "python", "stage1_pose_tracking.py",
        "--video", video_path,
        "--out", f"outputs/{video_name}"
    ])

    # stage2
    subprocess.run([
        "python", "stage2_pose3d_pro.py",
        "--input", f"outputs/{video_name}/tracked_pose.pkl",
        "--out", f"outputs/{video_name}"
    ])

    # stage3（人工，不自动跑）
    print(f"\n👉 Run Stage3 manually for: {video_name}")
    