#!/usr/bin/env python3
"""Generates the markerless kiosk wall texture for sim/kiosk_wall_mockup/ --
a synthetic "kiosk screen" look (title bar, button grid, icons) with NO
ArUco markers, same panel size as kiosk_wall (sim/generate_marker.py):
1m x 1m, 2000x2000px canvas (PX_PER_MM=2).

This exists to test the SCRUM-43 markerless claim for real: aruco_pnp_node
must find zero markers here (no ids to detect), while mono_multiview_node's
ORB matching should still work since the screen mockup has plenty of
edges/corners to match -- a plain solid-color wall would fail for mono too,
which wouldn't actually prove anything about being "markerless", just about
having no texture at all.

Deterministic (fixed seed) so the generated PNG is reproducible across runs
instead of silently changing every time this script is re-run.
"""
import os

import cv2
import numpy as np

PANEL_M = 1.0
PX_PER_MM = 2
CANVAS_PX = int(PANEL_M * 1000 * PX_PER_MM)  # 2000

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "kiosk_wall_mockup", "kiosk_wall_mockup_texture.png")

rng = np.random.default_rng(42)


def main():
    canvas = np.full((CANVAS_PX, CANVAS_PX, 3), (235, 235, 235), dtype=np.uint8)

    # Title bar.
    cv2.rectangle(canvas, (0, 0), (CANVAS_PX, 260), (60, 110, 180), -1)
    cv2.putText(canvas, "WELCOME", (120, 175), cv2.FONT_HERSHEY_SIMPLEX, 3.2,
                (255, 255, 255), 10, cv2.LINE_AA)

    # A grid of "buttons", each visually UNIQUE (distinct color + icon shape)
    # -- a repeating pattern would give ORB descriptors that look identical
    # at multiple cells, risking false matches/aliasing during RANSAC. Real
    # keypoint matching needs locally-unique texture, not just texture.
    palette = [(200, 120, 80), (90, 180, 140), (210, 170, 60), (150, 90, 190),
               (80, 160, 210), (190, 90, 110), (120, 200, 90), (200, 90, 150),
               (90, 140, 200), (210, 140, 40), (140, 90, 200), (90, 190, 190)]
    margin, gap = 140, 60
    cols, rows = 3, 4
    cell_w = (CANVAS_PX - 2 * margin - (cols - 1) * gap) // cols
    cell_h = (CANVAS_PX - 400 - 2 * margin - (rows - 1) * gap) // rows

    def draw_icon(shape, cx, cy, s):
        if shape == 0:
            cv2.circle(canvas, (cx, cy), s, (255, 255, 255), 8)
        elif shape == 1:
            cv2.rectangle(canvas, (cx - s, cy - s), (cx + s, cy + s), (255, 255, 255), 8)
        elif shape == 2:
            pts = np.array([[cx, cy - s], [cx - s, cy + s], [cx + s, cy + s]], np.int32)
            cv2.polylines(canvas, [pts], True, (255, 255, 255), 8)
        else:
            pts = np.array([[cx, cy - s], [cx + s, cy], [cx, cy + s], [cx - s, cy]], np.int32)
            cv2.polylines(canvas, [pts], True, (255, 255, 255), 8)

    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            x0 = margin + c * (cell_w + gap)
            y0 = 400 + margin + r * (cell_h + gap)
            x1, y1 = x0 + cell_w, y0 + cell_h
            color = palette[idx % len(palette)]
            cv2.rectangle(canvas, (x0, y0), (x1, y1), color, -1)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (30, 30, 30), 6)
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            draw_icon(idx % 4, cx, cy, min(cell_w, cell_h) // 5)
            cv2.putText(canvas, f"ITEM {idx + 1}", (x0 + 30, y1 - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 3, cv2.LINE_AA)

    # Light speckle noise over the whole panel so even background regions
    # have some texture for ORB, like a real printed/lit surface would.
    noise = rng.integers(-12, 13, size=canvas.shape, dtype=np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    cv2.imwrite(OUT_PATH, canvas)
    print(f"saved {OUT_PATH} ({CANVAS_PX}x{CANVAS_PX})")


if __name__ == "__main__":
    main()
