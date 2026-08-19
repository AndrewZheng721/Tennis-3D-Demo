import cv2

COCO_SKELETON = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


class Pose2DVisualizer:

    def draw(self, frame, tracked_frame):

        img = frame.copy()

        for player in tracked_frame.players:

            bbox = player.bbox.astype(int)

            # bbox
            cv2.rectangle(
                img,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                (0, 255, 0),
                2
            )

            # track id
            cv2.putText(
                img,
                f"ID:{player.track_id}",
                (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # keypoints
            for x, y in player.keypoints:

                cv2.circle(
                    img,
                    (int(x), int(y)),
                    4,
                    (0, 0, 255),
                    -1
                )

            # skeleton
            for s, e in COCO_SKELETON:

                if s >= len(player.keypoints):
                    continue

                if e >= len(player.keypoints):
                    continue

                p1 = player.keypoints[s]
                p2 = player.keypoints[e]

                cv2.line(
                    img,
                    (int(p1[0]), int(p1[1])),
                    (int(p2[0]), int(p2[1])),
                    (255, 0, 0),
                    2
                )

        return img