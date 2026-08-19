import json
import os
import argparse


# ==========================
# 参数
# ==========================

parser = argparse.ArgumentParser(
    description="Fusion pose2d + pose3d + ball"
)


parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="output directory"
)


args = parser.parse_args()


INPUT_DIR = args.input


POSE2D_JSON = os.path.join(
    INPUT_DIR,
    "tracked_pose.json"
)


POSE3D_JSON = os.path.join(
    INPUT_DIR,
    "pose3d.json"
)


BALL_JSON = os.path.join(
    INPUT_DIR,
    "ball_detections.json"
)


OUTPUT_JSON = os.path.join(
    INPUT_DIR,
    "fusion.json"
)



# ==========================
# 加载数据
# ==========================

print("Loading files...")


with open(
    POSE2D_JSON,
    "r",
    encoding="utf-8"
) as f:
    pose2d_data = json.load(f)



with open(
    POSE3D_JSON,
    "r",
    encoding="utf-8"
) as f:
    pose3d_data = json.load(f)



with open(
    BALL_JSON,
    "r",
    encoding="utf-8"
) as f:
    ball_data = json.load(f)



print(
    "pose2d frames:",
    len(pose2d_data)
)

print(
    "pose3d tracks:",
    len(pose3d_data)
)

print(
    "ball frames:",
    len(ball_data)
)



# ==========================
# Pose3D建立索引
# 
# key:
# (track_id, frame_id)
#
# value:
# joints3d
# ==========================


pose3d_index = {}


for person in pose3d_data:


    track_id = person["track_id"]


    for frame in person["frames"]:


        frame_id = frame["frame_id"]


        pose3d_index[
            (
                track_id,
                frame_id
            )
        ] = frame["joints3d"]



print(
    "pose3d index:",
    len(pose3d_index)
)



# ==========================
# Ball建立索引
#
# key:
# frame_id
#
# value:
# balls
# ==========================


ball_index = {}


for frame in ball_data:


    fid = frame["frame_id"]


    ball_index[fid] = frame["balls"]



# ==========================
# 融合
# ==========================


fusion_result = []



for frame in pose2d_data:


    frame_id = frame["frame_id"]


    frame_output = {

        "frame_id": frame_id,

        "players": [],

        "balls": ball_index.get(
            frame_id,
            []
        )

    }



    # ----------------------
    # 添加人物
    # ----------------------

    for player in frame["players"]:


        track_id = player["track_id"]


        player_output = {

            "track_id": track_id,


            "pose2d": player["keypoints"],


            "pose3d": pose3d_index.get(
                (
                    track_id,
                    frame_id
                ),
                None
            )

        }


        frame_output["players"].append(
            player_output
        )



    fusion_result.append(
        frame_output
    )



# ==========================
# 保存
# ==========================


with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as f:


    json.dump(
        fusion_result,
        f,
        indent=4,
        ensure_ascii=False
    )



print("===================")

print(
    "Fusion completed"
)

print(
    "Output:",
    OUTPUT_JSON
)

print(
    "Frames:",
    len(fusion_result)
)