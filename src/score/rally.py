from typing import List, Optional


def run_rally(bounces: List[dict]) -> List[dict]:
    events = []
    phase = "serve"
    last_side = None
    point_id = 0

    def close_point(frame_id, reason, err_side):
        nonlocal phase, last_side, point_id
        winner = None
        if err_side in ("near", "far"):
            winner = "far" if err_side == "near" else "near"
        events.append(
            {
                "type": "point",
                "frame_id": int(frame_id),
                "reason": reason,
                "error_side": err_side,
                "winner_side": winner,
                "point_id": point_id,
            }
        )
        point_id += 1
        phase = "serve"
        last_side = None

    for b in bounces:
        fid = int(b["frame_id"])
        side = b.get("side") or "unknown"
        if phase == "serve":
            if b.get("service"):
                events.append(
                    {
                        "type": "serve_in",
                        "frame_id": fid,
                        "side": side,
                        "service": b["service"],
                        "point_id": point_id,
                    }
                )
                phase = "rally"
                last_side = side
            elif b.get("in_singles"):
                events.append(
                    {
                        "type": "in_play",
                        "frame_id": fid,
                        "side": side,
                        "point_id": point_id,
                    }
                )
                phase = "rally"
                last_side = side
            else:
                events.append(
                    {
                        "type": "fault",
                        "frame_id": fid,
                        "side": side,
                        "call": b.get("call"),
                        "point_id": point_id,
                    }
                )
            continue
        if last_side and side == last_side and side != "unknown":
            close_point(fid, "double_bounce", side)
            continue
        if not b.get("in_singles"):
            close_point(fid, "out", side)
            continue
        events.append(
            {
                "type": "in_play",
                "frame_id": fid,
                "side": side,
                "point_id": point_id,
            }
        )
        last_side = side
    return events
