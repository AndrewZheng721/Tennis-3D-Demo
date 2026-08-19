from ultralytics import YOLO
import cv2
import json
import os
import pandas as pd


# =====================
# 配置
# =====================

MODEL_PATH = "weights/yolo26m.pt"

VIDEO_PATH = "data/tennis_high_angle_01.mp4"

OUTPUT_DIR = "outputs"

VIDEO_OUT = os.path.join(
    OUTPUT_DIR,
    "ball_tracking_bytetrack.mp4"
)

JSON_OUT = os.path.join(
    OUTPUT_DIR,
    "ball_tracking_bytetrack.json"
)

EXCEL_OUT = os.path.join(
    OUTPUT_DIR,
    "ball_tracking_bytetrack.xlsx"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)



# =====================
# 加载模型
# =====================

model = YOLO(MODEL_PATH)



# =====================
# 打开视频
# =====================

cap = cv2.VideoCapture(
    VIDEO_PATH
)


if not cap.isOpened():
    raise RuntimeError(
        "Video open failed"
    )


fps = cap.get(
    cv2.CAP_PROP_FPS
)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)



print(
    "FPS:",
    fps
)

print(
    "Resolution:",
    width,
    height
)



# =====================
# writer
# =====================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)


writer = cv2.VideoWriter(
    VIDEO_OUT,
    fourcc,
    fps,
    (width,height)
)



# =====================
# 保存
# =====================

ball_results = []

excel_results = []


frame_id = 0



while True:


    ret, frame = cap.read()


    if not ret:
        break



    # =====================
    # YOLO + ByteTrack
    # =====================

    results = model.track(

        frame,

        imgsz=1280,

        conf=0.05,

        classes=[32],

        persist=True,

        tracker="bytetrack.yaml",

        verbose=False

    )



    frame_info = {

        "frame_id": frame_id,

        "balls":[]

    }



    vis_frame = frame.copy()



    for r in results:


        boxes = r.boxes


        if boxes is None:
            continue



        for box in boxes:


            conf = float(
                box.conf[0]
            )


            cls = int(
                box.cls[0]
            )


            name = model.names[cls]


            if name != "sports ball":
                continue



            x1,y1,x2,y2 = map(
                int,
                box.xyxy[0]
            )


            cx = (
                x1+x2
            ) / 2


            cy = (
                y1+y2
            ) / 2



            # =====================
            # Track ID
            # =====================

            track_id = None


            if box.id is not None:

                track_id = int(
                    box.id[0]
                )



            frame_info["balls"].append(

                {

                    "track_id":track_id,

                    "bbox":[
                        x1,
                        y1,
                        x2,
                        y2
                    ],

                    "center":[
                        cx,
                        cy
                    ],

                    "confidence":conf

                }

            )



            # =====================
            # Excel
            # =====================

            excel_results.append(

                {

                    "frame_id":frame_id,

                    "time_sec":round(
                        frame_id/fps,
                        3
                    ),

                    "track_id":track_id,

                    "x1":x1,

                    "y1":y1,

                    "x2":x2,

                    "y2":y2,

                    "center_x":round(
                        cx,
                        2
                    ),

                    "center_y":round(
                        cy,
                        2
                    ),

                    "confidence":round(
                        conf,
                        4
                    )

                }

            )



            # =====================
            # 可视化
            # =====================


            cv2.rectangle(

                vis_frame,

                (x1,y1),

                (x2,y2),

                (0,255,0),

                2

            )


            cv2.circle(

                vis_frame,

                (
                    int(cx),
                    int(cy)
                ),

                5,

                (0,0,255),

                -1

            )


            cv2.putText(

                vis_frame,

                f"id:{track_id} {conf:.2f}",

                (x1,y1-10),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.6,

                (0,255,0),

                2

            )



    ball_results.append(
        frame_info
    )


    writer.write(
        vis_frame
    )



    if frame_id % 100 == 0:

        print(
            "processed:",
            frame_id
        )



    frame_id += 1



cap.release()

writer.release()



# =====================
# JSON
# =====================

with open(

    JSON_OUT,

    "w",

    encoding="utf-8"

) as f:


    json.dump(

        ball_results,

        f,

        indent=4,

        ensure_ascii=False

    )



# =====================
# Excel
# =====================


df = pd.DataFrame(
    excel_results
)


df.to_excel(

    EXCEL_OUT,

    index=False

)



print("================")

print(
    "frames:",
    frame_id
)

print(
    "video:",
    VIDEO_OUT
)

print(
    "json:",
    JSON_OUT
)

print(
    "excel:",
    EXCEL_OUT
)

print(
    "detections:",
    len(excel_results)
)