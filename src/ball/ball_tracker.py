"""网球检测与轨迹补全。

这个文件要解决的问题
--------------------
网球在画面里非常小（高位 1080p 里往往只有几个像素），还飞得很快，
经常模糊、被球拍挡住、或者某几帧完全看不见。
我们需要每一帧尽量给出球心坐标，才能画出轨迹、判断击球时刻。

YOLO 是什么？
------------
YOLO（You Only Look Once）是目标检测模型：给它一张图，它输出若干个框。
每个框包含：左上角、右下角、类别、置信度。

这里用的不是「通用 COCO 模型里的 sports ball」，
而是 tennis_analysis 在网球数据上微调过的 YOLO。
通用模型把网球、足球、篮球当成同一类，而且对「几个像素的小球」很弱。
微调后的模型只认网球，召回率会高很多。

为什么检测完还要插值？
--------------------
即使微调过，仍会有漏检。
只在「很短的空缺」里插值，例如球飞过网被挡了 3～5 帧。
如果两帧之间球突然跳到画面另一侧，那是误检，必须断开，不能连成折线。

水平机位轨迹为什么会乱
--------------------
旧逻辑每帧都选「置信度最高的框」。灯光、网、鞋子都可能比真球分更高，
于是点在画面里乱跳，再被无限制插值连成蜘蛛网。
现在改为：优先选离上一帧球心近的框；跳太远就丢掉；漏检超过几帧才重新搜索。

击球帧怎么找？
------------
球被打出去后，它在画面里的上下方向通常会换一次（从变高变成变低，或反过来）。
我们看球心 y 坐标的滚动平均，找「斜率变号且能持续一段时间」的位置。
这只是粗检，水平机位、侧旋很高时会不准，动作识别阶段可以再校正。
"""

from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


class BallTracker:
    """逐帧检测网球，并对漏检做时间插值。"""

    def __init__(
        self,
        model_path: str,
        conf: float = 0.15,
        imgsz: int = 1280,
        coco_sports_ball: bool = False,
    ):
        """加载 YOLO 权重。

        conf：置信度门槛。太高会漏检，太低会把鞋、线、远景噪点当成球。
        imgsz：推理时把图像缩到的边长。小球必须用比较大的尺寸（1280），
              用默认 640 时，高位机位里球经常只剩 1～2 个像素，直接消失。
        coco_sports_ball：如果没有网球专用权重，可退回通用 YOLO，只保留 32 号类。
        """
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.coco_sports_ball = coco_sports_ball
        self._prev_center = None
        self._misses = 0

    def reset(self):
        """换一段视频前调用，清掉上一支球的位置记忆。"""
        self._prev_center = None
        self._misses = 0

    def _candidates(self, frame: np.ndarray):
        kwargs = {
            "conf": self.conf,
            "imgsz": self.imgsz,
            "verbose": False,
        }
        if self.coco_sports_ball:
            kwargs["classes"] = [32]

        result = self.model.predict(frame, **kwargs)[0]
        found = []
        if result.boxes is None:
            return found
        names = result.names
        for box in result.boxes:
            cls_id = int(box.cls[0])
            name = names.get(cls_id, "")
            if self.coco_sports_ball and name != "sports ball":
                continue
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            found.append(xyxy + [conf])
        return found

    def detect_frame(self, frame: np.ndarray) -> Dict[int, List[float]]:
        """检测单帧。有上一帧位置时，优先跟近处的框，而不是全局最高分。"""
        h, w = frame.shape[:2]
        max_jump = 0.12 * float(np.hypot(w, h))
        cands = self._candidates(frame)
        if not cands:
            self._misses += 1
            if self._misses > 8:
                self._prev_center = None
            return {}

        def center_of(box):
            return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

        if self._prev_center is None:
            best = max(cands, key=lambda b: b[4])
        else:
            scored = []
            for box in cands:
                cx, cy = center_of(box)
                dist = float(np.hypot(cx - self._prev_center[0], cy - self._prev_center[1]))
                if dist > max_jump:
                    continue
                scored.append((box[4] - 0.6 * (dist / max_jump), box))
            if scored:
                best = max(scored, key=lambda x: x[0])[1]
            else:
                best = max(cands, key=lambda b: b[4])
                if best[4] < 0.4:
                    self._misses += 1
                    if self._misses > 8:
                        self._prev_center = None
                    return {}
                self._prev_center = None

        self._prev_center = center_of(best)
        self._misses = 0
        return {1: best}

    def detect_video(
        self,
        cap: cv2.VideoCapture,
        max_frames: Optional[int] = None,
        progress=None,
    ):
        """按帧读取视频并检测，不把所有图像一次性装进内存。

        长视频如果整段读进 RAM 会爆内存。这里只保留每帧的 4～5 个数字。
        """
        detections = []
        frame_id = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and frame_id >= max_frames:
                break
            det = self.detect_frame(frame)
            detections.append(det)
            if progress is not None:
                progress(frame_id)
            frame_id += 1
        return detections

    def interpolate_ball_positions(
        self, ball_positions: List[Dict[int, List[float]]]
    ) -> List[Dict[int, List[float]]]:
        """用前后帧把漏检补上。

        pandas.interpolate 默认按时间下标做线性插值。
        开头连续缺失时用 bfill（用后面第一个有效值往前填），
        避免轨迹一开始就是空的。
        """
        rows = []
        confs = []
        for item in ball_positions:
            box = item.get(1, [])
            if box and len(box) >= 4 and np.isfinite(box[0]):
                rows.append(box[:4])
                confs.append(box[4] if len(box) > 4 else 1.0)
            else:
                rows.append([np.nan, np.nan, np.nan, np.nan])
                confs.append(np.nan)

        df = pd.DataFrame(rows, columns=["x1", "y1", "x2", "y2"])
        df = df.interpolate(limit=8)
        conf_s = pd.Series(confs).interpolate(limit=8)

        filled = []
        prev_c = None
        for i, row in df.iterrows():
            if row.isna().any():
                filled.append({})
                continue
            box = [
                float(row["x1"]),
                float(row["y1"]),
                float(row["x2"]),
                float(row["y2"]),
                float(conf_s.iloc[i]) if not np.isnan(conf_s.iloc[i]) else 0.0,
            ]
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            if prev_c is not None:
                jump = float(np.hypot(cx - prev_c[0], cy - prev_c[1]))
                if jump > 220:
                    filled.append({})
                    prev_c = None
                    continue
            filled.append({1: box})
            prev_c = (cx, cy)
        return filled

    def get_ball_shot_frames(
        self, ball_positions: List[Dict[int, List[float]]], fps: float = 30.0
    ) -> List[int]:
        """根据球心 y 方向换向，粗略找出击球帧。

        原仓库按 24fps、窗口 25 帧来写。
        你的视频大多是 30fps，这里按时间换算：大约 1 秒内方向要保持改变，
        才认为是一次真正击球，而不是噪声抖一下。
        """
        rows = []
        for item in ball_positions:
            box = item.get(1, [])
            rows.append(box[:4] if box and len(box) >= 4 else [np.nan] * 4)
        df = pd.DataFrame(rows, columns=["x1", "y1", "x2", "y2"])
        df["mid_y"] = (df["y1"] + df["y2"]) / 2
        df["mid_y_rolling_mean"] = df["mid_y"].rolling(
            window=5, min_periods=1, center=False
        ).mean()
        df["delta_y"] = df["mid_y_rolling_mean"].diff()
        df["ball_hit"] = 0

        min_change = max(15, int(round(fps * 0.83)))
        lookahead = int(min_change * 1.2)
        for i in range(1, len(df) - lookahead):
            d0 = df["delta_y"].iloc[i]
            d1 = df["delta_y"].iloc[i + 1]
            if not np.isfinite(d0) or not np.isfinite(d1):
                continue
            neg = d0 > 0 and d1 < 0
            pos = d0 < 0 and d1 > 0
            if not (neg or pos):
                continue
            change_count = 0
            for j in range(i + 1, i + lookahead + 1):
                dj = df["delta_y"].iloc[j]
                if not np.isfinite(dj):
                    continue
                if neg and d0 > 0 and dj < 0:
                    change_count += 1
                elif pos and d0 < 0 and dj > 0:
                    change_count += 1
            if change_count > min_change - 1:
                df.loc[i, "ball_hit"] = 1

        return df.index[df["ball_hit"] == 1].tolist()

    def to_json(
        self,
        ball_positions: List[Dict[int, List[float]]],
        shot_frames: List[int],
        fps: float,
    ):
        """转成和本项目其它阶段类似的 JSON 结构。"""
        frames = []
        for i, item in enumerate(ball_positions):
            box = item.get(1)
            if not box:
                frames.append(
                    {
                        "frame_id": i,
                        "time_sec": round(i / fps, 4) if fps else i,
                        "bbox": None,
                        "center": None,
                        "confidence": None,
                    }
                )
                continue
            x1, y1, x2, y2 = box[:4]
            conf = box[4] if len(box) > 4 else None
            frames.append(
                {
                    "frame_id": i,
                    "time_sec": round(i / fps, 4) if fps else i,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "center": [float((x1 + x2) / 2), float((y1 + y2) / 2)],
                    "confidence": None if conf is None else float(conf),
                }
            )
        return {
            "fps": float(fps),
            "num_frames": len(frames),
            "shot_frames": [int(x) for x in shot_frames],
            "frames": frames,
        }

    def draw_frame(
        self,
        frame: np.ndarray,
        bbox_with_conf: Optional[List[float]],
        trail: List[tuple],
        frame_id: int,
        is_shot: bool = False,
    ) -> np.ndarray:
        """画当前框、最近一段轨迹、击球标记。"""
        vis = frame.copy()
        h, w = vis.shape[:2]
        max_jump = 0.12 * float(np.hypot(w, h))
        if len(trail) >= 2:
            seg = []
            for p in trail:
                if not seg:
                    seg = [p]
                    continue
                if np.hypot(p[0] - seg[-1][0], p[1] - seg[-1][1]) > max_jump:
                    if len(seg) >= 2:
                        cv2.polylines(
                            vis,
                            [np.array(seg, dtype=np.int32).reshape(-1, 1, 2)],
                            False,
                            (0, 255, 255),
                            2,
                        )
                    seg = [p]
                else:
                    seg.append(p)
            if len(seg) >= 2:
                cv2.polylines(
                    vis,
                    [np.array(seg, dtype=np.int32).reshape(-1, 1, 2)],
                    False,
                    (0, 255, 255),
                    2,
                )
        if bbox_with_conf:
            x1, y1, x2, y2 = map(int, bbox_with_conf[:4])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cv2.circle(vis, (cx, cy), 4, (0, 255, 255), -1)
            conf = bbox_with_conf[4] if len(bbox_with_conf) > 4 else 0
            cv2.putText(
                vis,
                f"ball {conf:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
        if is_shot:
            cv2.putText(
                vis,
                "SHOT",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
            )
        cv2.putText(
            vis,
            f"Frame: {frame_id}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        return vis
