import os
import cv2
import json
from collections import defaultdict
import argparse


parser = argparse.ArgumentParser(description="第三阶段：动作标注工具")
parser.add_argument("--video", type=str, help="输入视频文件路径")
parser.add_argument("--out", type=str, help="输出目录路径")
args = parser.parse_args()

VIDEO_PATH = os.path.join(args.video)
OUTPUT_JSON = os.path.join(args.out, "action_labels.json")


class LabelTool:

    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)

        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.frame_id = 0
        self.paused = False

        self.video_name = os.path.basename(video_path).split(".")[0]
        self.current_track_id = 1

        # 存标注
        self.labels = []

        print("FPS:", self.fps)
        print("Total frames:", self.total)

    def draw_info(self, frame):

        info = f"Frame: {self.frame_id}"
        cv2.putText(frame, info, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame

    def run(self):

        while True:

            if not self.paused:
                ret, frame = self.cap.read()

                if not ret:
                    break

                self.frame_id += 1

            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_id)
                ret, frame = self.cap.read()

                if not ret:
                    break

            frame = self.draw_info(frame)

            cv2.imshow("Label Tool", frame)

            key = cv2.waitKey(30) & 0xFF

            # =========================
            # controls
            # =========================

            if key == ord('q'):
                break

            elif key == ord(' '):
                self.paused = not self.paused

            elif key == ord('s'):
                self.labels.append({
                    "frame_id": self.frame_id,
                    "track_id": self.current_track_id,
                    "label": "serve"
                })
                print("serve:", self.frame_id, "track_id:", self.current_track_id)
            
            elif key == ord('f'):
                self.labels.append({
                    "frame_id": self.frame_id,
                    "track_id": self.current_track_id,
                    "label": "forehand"
                })
                print("forehand:", self.frame_id, "track_id:", self.current_track_id)

            elif key == ord('b'):
                self.labels.append({
                    "frame_id": self.frame_id,
                    "track_id": self.current_track_id,
                    "label": "backhand"
                })
                print("backhand:", self.frame_id, "track_id:", self.current_track_id)

            elif key == ord('v'):
                self.labels.append({
                    "frame_id": self.frame_id,
                    "track_id": self.current_track_id,
                    "label": "volley"
                })
                print("volley:", self.frame_id, "track_id:", self.current_track_id)

            elif key == ord('m'):
                self.labels.append({
                    "frame_id": self.frame_id,
                    "track_id": self.current_track_id,
                    "label": "smash"
                })
                print("smash:", self.frame_id, "track_id:", self.current_track_id)

            elif key == 83:  # →
                self.frame_id += 1
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_id)

            elif key == 81:  # ←
                self.frame_id = max(0, self.frame_id - 2)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_id)

        self.cap.release()
        cv2.destroyAllWindows()

        self.save()

    def save(self):

        with open(OUTPUT_JSON, "w") as f:
            json.dump(self.labels, f, indent=4)
        print("Saved to:", OUTPUT_JSON)


if __name__ == "__main__":
    tool = LabelTool(VIDEO_PATH)
    tool.run()
