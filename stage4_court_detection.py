import cv2
import numpy as np
import json
import os


VIDEO_PATH = "data/tennis_high_angle_01.mp4"

OUTPUT_DIR = "outputs"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


cap = cv2.VideoCapture(
    VIDEO_PATH
)


# 读取第一帧
ret, frame = cap.read()

if not ret:
    raise Exception("video open failed")


h, w = frame.shape[:2]


print(
    "video size:",
    w,
    h
)


# ==========================
# 球场检测
# ==========================

def detect_court(frame):

    img = frame.copy()


    # HSV
    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2HSV
    )


    # 网球场绿色区域
    lower = np.array(
        [30,40,40]
    )

    upper = np.array(
        [90,255,255]
    )


    mask = cv2.inRange(
        hsv,
        lower,
        upper
    )


    # 去噪
    kernel = np.ones(
        (5,5),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )


    # 边缘
    edges = cv2.Canny(
        mask,
        50,
        150
    )


    # Hough线
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi/180,
        threshold=80,
        minLineLength=80,
        maxLineGap=20
    )


    result = img.copy()


    points=[]


    if lines is not None:

        for line in lines:

            x1,y1,x2,y2 = line[0]


            cv2.line(
                result,
                (x1,y1),
                (x2,y2),
                (0,0,255),
                2
            )


    return result, lines, mask



result, lines, mask = detect_court(frame)


cv2.imwrite(
    "outputs/court_debug.jpg",
    result
)


print(
    "saved outputs/court_debug.jpg"
)
