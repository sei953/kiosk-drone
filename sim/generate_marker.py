#!/usr/bin/env python3
"""Generate kiosk wall textures: four 1m x 1m panels (north/east/south/west),
each with four 150mm DICT_4X4_50 ArUco markers, one per corner, 50mm inset
from the panel edge.

ID layout per wall (as authored in the PNG, viewer facing the wall):
    a --- a+1
    |       |
    a+3 --- a+2
where `a` is the wall's ID offset (north=0, east=4, south=8, west=12).

Object points for solvePnP
---------------------------
Wall-local frame: origin at panel center, X to the right / Z up, as seen by
a camera facing the wall. This is the SAME canonical corner layout for every
wall -- only the ID numbers differ per wall (see kiosk_vision/aruco_pnp_node.py
WALLS table). The world placement/yaw of each wall (so its marked face points
at the origin) is handled in sim/worlds/kiosk_4walls.sdf, not here.

Marker center = MARGIN_MM + MARKER_MM/2 = 50 + 75 = 125mm in from each edge.
Panel half-size = 500mm, so center offset from panel center = 500-125 = 375mm.

Per-corner object points for marker `i` (axis-aligned, no in-plane rotation):
    top-left     = (cx - MARKER_MM/2, cy + MARKER_MM/2, 0)
    top-right    = (cx + MARKER_MM/2, cy + MARKER_MM/2, 0)
    bottom-right = (cx + MARKER_MM/2, cy - MARKER_MM/2, 0)
    bottom-left  = (cx - MARKER_MM/2, cy - MARKER_MM/2, 0)

UV mapping direction (gz-sim PBR box albedo) is not visually re-verified for
the new walls -- confirmed only for the original north wall (id 0 top-left,
see docs/PROGRESS.md 2026-09-08). Re-check the first captured frame per wall.
"""

import os

import cv2
import numpy as np

PANEL_M = 1.0
MARKER_MM = 150
MARGIN_MM = 50  # inset from panel edge to marker edge
PX_PER_MM = 2  # 2000x2000 canvas for a 1m panel

CORNER_OFFSETS = {  # id offset within a wall's 4-marker set (0..3)
    "top_left": 0,
    "top_right": 1,
    "bottom_right": 2,
    "bottom_left": 3,
}

PX4_MODELS_DIR = "/home/joheeho/PX4-Autopilot/Tools/simulation/gz/models"
REPO_SIM_DIR = os.path.dirname(os.path.abspath(__file__))

# wall name -> (ID offset, model dir name). North reuses the original
# kiosk_wall model (ids 0-3, unchanged from the single-wall setup).
WALLS = {
    "north": (0, "kiosk_wall"),
    "east": (4, "kiosk_wall_east"),
    "south": (8, "kiosk_wall_south"),
    "west": (12, "kiosk_wall_west"),
}

canvas_px = int(PANEL_M * 1000 * PX_PER_MM)
marker_px = MARKER_MM * PX_PER_MM
margin_px = MARGIN_MM * PX_PER_MM

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# (row_off, col_off) of each marker's top-left pixel in the canvas
PLACEMENTS = {
    "top_left": (margin_px, margin_px),
    "top_right": (margin_px, canvas_px - margin_px - marker_px),
    "bottom_right": (canvas_px - margin_px - marker_px, canvas_px - margin_px - marker_px),
    "bottom_left": (canvas_px - margin_px - marker_px, margin_px),
}


def draw_marker(marker_id, side_px):
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(dictionary, marker_id, side_px)
    return cv2.aruco.drawMarker(dictionary, marker_id, side_px)  # OpenCV 4.6 fallback


def generate_wall_png(id_offset, out_path):
    canvas = np.full((canvas_px, canvas_px), 255, dtype=np.uint8)
    ids_used = {}
    for name, corner_off in CORNER_OFFSETS.items():
        marker_id = id_offset + corner_off
        marker = draw_marker(marker_id, marker_px)
        row_off, col_off = PLACEMENTS[name]
        canvas[row_off:row_off + marker_px, col_off:col_off + marker_px] = marker
        ids_used[name] = marker_id
    out = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, out)
    return ids_used


def main():
    for wall_name, (id_offset, model_dir) in WALLS.items():
        for base_dir in (REPO_SIM_DIR, PX4_MODELS_DIR):
            out_path = os.path.join(base_dir, model_dir, f"{model_dir}_marker.png")
            ids_used = generate_wall_png(id_offset, out_path)
        print(f"saved {wall_name} wall ({model_dir}, ids {id_offset}-{id_offset + 3}) "
              f"-> {REPO_SIM_DIR}/{model_dir}/ and {PX4_MODELS_DIR}/{model_dir}/")
        for name, marker_id in ids_used.items():
            print(f"  id={marker_id} ({name})")


if __name__ == "__main__":
    main()
