import glob
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_ZONE = {
    "far_left": "远端左区",
    "far_right": "远端右区",
    "near_left": "近端左区",
    "near_right": "近端右区",
    "far_in": "远端界内",
    "near_in": "近端界内",
    "out": "界外",
    "far": "远端",
    "near": "近端",
}
_font_file = None
_fonts = {}


def zone_cn(name):
    if not name or name == "-":
        return "无"
    return _ZONE.get(name, str(name))


def font_file():
    global _font_file
    if _font_file is not None:
        return _font_file or None
    cands = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]
    for pat in (
        "/usr/share/fonts/**/NotoSansCJK*.ttc",
        "/usr/share/fonts/**/NotoSansSC*.otf",
        "/usr/share/fonts/**/wqy*.ttc",
    ):
        cands.extend(glob.glob(pat, recursive=True))
    for p in cands:
        if p and os.path.isfile(p):
            _font_file = p
            return p
    _font_file = ""
    return None


def _font(size):
    path = font_file()
    key = (path, size)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(path, size) if path else ImageFont.load_default()
    return _fonts[key]


def put_cn(img, text, org, color, scale=0.7, thickness=2):
    size = max(16, int(round(34 * scale)))
    x, y = int(org[0]), max(0, int(org[1]) - size)
    rgb = (int(color[2]), int(color[1]), int(color[0]))
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil).text((x, y), text, font=_font(size), fill=rgb, stroke_width=max(0, thickness // 2), stroke_fill=rgb)
    img[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    return img
