import cv2
import numpy as np
import os
import json


# ============================================================
# 配置
# ============================================================

VIDEO_PATH = "data/tennis_demo1.mp4"
OUTPUT_VIDEO = "outputs/court_detection.mp4"
OUTPUT_JSON = "outputs/court_lines.json"

# 每隔多少帧重新检测一次
DETECT_INTERVAL = 30

# Canny 参数
CANNY_LOW = 50
CANNY_HIGH = 150

# Hough 参数
HOUGH_THRESHOLD = 80
MIN_LINE_LENGTH = 100
MAX_LINE_GAP = 30


# ============================================================
# 工具函数
# ============================================================

def line_length(line):
    x1, y1, x2, y2 = line
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def line_angle(line):
    x1, y1, x2, y2 = line

    angle = np.degrees(
        np.arctan2(
            y2 - y1,
            x2 - x1
        )
    )

    # 统一到 [-90, 90]
    if angle > 90:
        angle -= 180

    if angle < -90:
        angle += 180

    return angle


def draw_line(img, line, color, thickness=3):
    x1, y1, x2, y2 = map(int, line)

    cv2.line(
        img,
        (x1, y1),
        (x2, y2),
        color,
        thickness
    )


# ============================================================
# 白线检测
# ============================================================

def detect_white_lines(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 网球场白线通常亮度高
    lower = np.array([0, 0, 150])
    upper = np.array([180, 100, 255])

    mask = cv2.inRange(
        hsv,
        lower,
        upper
    )

    # 去除小噪声
    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    edges = cv2.Canny(
        mask,
        CANNY_LOW,
        CANNY_HIGH
    )

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=MIN_LINE_LENGTH,
        maxLineGap=MAX_LINE_GAP
    )

    if lines is None:
        return [], mask, edges

    result = []

    for l in lines:

        line = l[0]

        length = line_length(line)

        if length < MIN_LINE_LENGTH:
            continue

        angle = line_angle(line)

        result.append({
            "line": line.tolist(),
            "length": float(length),
            "angle": float(angle)
        })

    return result, mask, edges


# ============================================================
# 将线分成水平 / 垂直方向
# ============================================================

def classify_lines(lines):

    horizontal = []
    vertical = []

    for item in lines:

        angle = item["angle"]

        # 透视情况下，球场线不一定严格水平/垂直
        if abs(angle) < 25:
            horizontal.append(item)

        elif abs(abs(angle) - 90) < 25:
            vertical.append(item)

    return horizontal, vertical


# ============================================================
# 根据线段估计球场区域
# ============================================================

def estimate_court(lines, frame):

    h_lines, v_lines = classify_lines(lines)

    height, width = frame.shape[:2]

    # ========================================================
    # 水平线
    # ========================================================

    h_lines = sorted(
        h_lines,
        key=lambda x: x["length"],
        reverse=True
    )

    # ========================================================
    # 垂直/斜线
    # ========================================================

    v_lines = sorted(
        v_lines,
        key=lambda x: x["length"],
        reverse=True
    )

    # --------------------------------------------------------
    # 这里先采用最长线段作为候选
    # 后续可以进一步优化成真正的球场线聚类
    # --------------------------------------------------------

    candidate_lines = (
        h_lines[:10] +
        v_lines[:10]
    )

    # ========================================================
    # 找到所有候选线的端点
    # ========================================================

    points = []

    for item in candidate_lines:

        x1, y1, x2, y2 = item["line"]

        points.append((x1, y1))
        points.append((x2, y2))

    if len(points) < 4:
        return None

    points = np.array(points, dtype=np.float32)

    # ========================================================
    # 用凸包得到大致球场区域
    # ========================================================

    hull = cv2.convexHull(points)

    hull = hull.reshape(-1, 2)

    if len(hull) < 4:
        return None

    # ========================================================
    # 找最小外接矩形
    # ========================================================

    rect = cv2.minAreaRect(hull)

    box = cv2.boxPoints(rect)

    box = np.array(box, dtype=np.float32)

    return {
        "box": box.tolist(),
        "horizontal_lines": h_lines,
        "vertical_lines": v_lines
    }


# ============================================================
# 对球场四角排序
# ============================================================

def order_points(points):

    points = np.array(points, dtype=np.float32)

    result = np.zeros((4, 2), dtype=np.float32)

    s = points.sum(axis=1)

    result[0] = points[np.argmin(s)]  # 左上
    result[2] = points[np.argmax(s)]  # 右下

    diff = np.diff(points, axis=1)

    result[1] = points[np.argmin(diff)]  # 右上
    result[3] = points[np.argmax(diff)]  # 左下

    return result


# ============================================================
# 绘制结果
# ============================================================

def draw_court(frame, court):

    output = frame.copy()

    if court is None:
        return output

    box = np.array(
        court["box"],
        dtype=np.float32
    )

    box = order_points(box)

    # ========================================================
    # 球场外框
    # ========================================================

    pts = box.astype(np.int32)

    cv2.polylines(
        output,
        [pts],
        True,
        (0, 255, 0),
        4
    )

    # ========================================================
    # 四个角点
    # ========================================================

    names = [
        "TL",
        "TR",
        "BR",
        "BL"
    ]

    for point, name in zip(box, names):

        x, y = map(int, point)

        cv2.circle(
            output,
            (x, y),
            8,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            output,
            name,
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    # ========================================================
    # 画检测到的水平线
    # ========================================================

    for item in court["horizontal_lines"][:10]:

        draw_line(
            output,
            item["line"],
            (255, 0, 0),
            2
        )

    # ========================================================
    # 画检测到的垂直线
    # ========================================================

    for item in court["vertical_lines"][:10]:

        draw_line(
            output,
            item["line"],
            (0, 255, 255),
            2
        )

    return output


# ============================================================
# 主程序
# ============================================================

def main():

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    cap = cv2.VideoCapture(
        VIDEO_PATH
    )

    if not cap.isOpened():

        print(
            "无法打开视频:",
            VIDEO_PATH
        )

        return

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    print("FPS:", fps)
    print("Resolution:", width, height)
    print("Frame count:", frame_count)

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    court = None

    frame_id = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # ====================================================
        # 定期重新检测
        # ====================================================

        if (
            court is None
            or frame_id % DETECT_INTERVAL == 0
        ):

            lines, mask, edges = detect_white_lines(
                frame
            )

            print(
                f"Frame {frame_id}: "
                f"{len(lines)} lines"
            )

            new_court = estimate_court(
                lines,
                frame
            )

            if new_court is not None:
                court = new_court

        # ====================================================
        # 绘制
        # ====================================================

        output = draw_court(
            frame,
            court
        )

        cv2.putText(
            output,
            f"Frame: {frame_id}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        writer.write(output)

        frame_id += 1

    cap.release()
    writer.release()

    # ========================================================
    # 保存球场信息
    # ========================================================

    if court is not None:

        with open(
            OUTPUT_JSON,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                court,
                f,
                ensure_ascii=False,
                indent=2
            )

        print()
        print("================================")
        print("Court detected")
        print("================================")
        print(
            "Court JSON:",
            OUTPUT_JSON
        )
        print(
            "Output video:",
            OUTPUT_VIDEO
        )

    else:

        print(
            "没有检测到球场"
        )


if __name__ == "__main__":
    main()