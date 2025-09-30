#region Documentation
'''
Four labels mark the **centres of the 4 quadrant rectangles**:
    125 is just whatever size you specify
     (-125,125)      (125,125)
          ────────────────
          │       │       │
          │  TL   │  TR   │
          │       │       │
     (-125,-125)    (125,-125)

(+X → right, +Y → up)
=========================================================r
calibration_board_generator.py
=========================================================i
This program generates a calibration board
=========================================================f
Author: Eric Osborne
Date:   Aug 15, 2025
This work is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives
4.0 International License, http://creativecommons.org/licenses/by-nc-nd/4.0/
=========================================================t
'''
#endregion

#region Libraries
from moms_apriltag import TagGenerator2
from PIL import Image, ImageDraw, ImageFont
import sys
#endregion

#region Variables
MM_PER_TAG   = 50          # tag size
MM_MARGIN    = 10          # white border
WORK_W_MM    = int(sys.argv[1])
WORK_H_MM = int(sys.argv[2])
DPI          = 300
GRID_MM      = 50
FONT_SIZE_PT = 100

# calculated dimensions
BOARD_W_MM = WORK_W_MM + 2*(MM_MARGIN + MM_PER_TAG)
BOARD_H_MM = WORK_H_MM + 2*(MM_MARGIN + MM_PER_TAG)
px_per_mm   = DPI / 25.4
mm2px       = lambda mm: int(round(mm * px_per_mm))
board_w_px  = mm2px(BOARD_W_MM)
board_h_px  = mm2px(BOARD_H_MM)
tag_px      = mm2px(MM_PER_TAG)
margin_px   = mm2px(MM_MARGIN)
WORK_ORIGIN = (margin_px + tag_px, margin_px + tag_px)   # top-left of work-area
WORK_W_PX   = mm2px(WORK_W_MM)
WORK_H_PX   = mm2px(WORK_H_MM)
CENTER_PX   = (WORK_ORIGIN[0] + WORK_W_PX // 2,
               WORK_ORIGIN[1] + WORK_H_PX // 2)
#endregion

#region Helper Functions
def world2px(x_mm, y_mm):
    """centre-origin mm  →  image px  (+Y up)."""
    return (CENTER_PX[0] + mm2px(x_mm),
            CENTER_PX[1] - mm2px(y_mm))
#endregion

#region Main
board = Image.new("L", (board_w_px, board_h_px), 255)
draw  = ImageDraw.Draw(board)

# (1) AprilTags in all four corners
tg_ids  = [0, 1, 2, 3]                   # TL, TR, BR, BL
tg      = TagGenerator2("tag36h11")
tags    = {i: Image.fromarray(tg.generate(i)).resize((tag_px, tag_px),
                                                     Image.NEAREST)
           for i in tg_ids}
tag_pos = {
    0: (margin_px,                              margin_px),             # TL
    1: (board_w_px - margin_px - tag_px,        margin_px),             # TR
    2: (board_w_px - margin_px - tag_px,
        board_h_px - margin_px - tag_px),                              # BR
    3: (margin_px,
        board_h_px - margin_px - tag_px)                                # BL
}
for tid, (x, y) in tag_pos.items():
    board.paste(tags[tid], (x, y))

# (2) outline work-area
draw.rectangle([WORK_ORIGIN,
                (WORK_ORIGIN[0] + WORK_W_PX - 1,
                 WORK_ORIGIN[1] + WORK_H_PX - 1)],
               outline=0, width=4)

# (3) grid lines
step_px = mm2px(GRID_MM)
right   = WORK_ORIGIN[0] + WORK_W_PX
bottom  = WORK_ORIGIN[1] + WORK_H_PX
for x in range(WORK_ORIGIN[0] + step_px, right, step_px):
    draw.line([(x, WORK_ORIGIN[1]), (x, bottom)], fill=200)
for y in range(WORK_ORIGIN[1] + step_px, bottom, step_px):
    draw.line([(WORK_ORIGIN[0], y), (right, y)], fill=200)

# heavy axes through the centre
draw.line([(WORK_ORIGIN[0], CENTER_PX[1]), (right, CENTER_PX[1])],
          fill=0, width=2)                    # X-axis
draw.line([(CENTER_PX[0], WORK_ORIGIN[1]), (CENTER_PX[0], bottom)],
          fill=0, width=2)                    # Y-axis

# (4) centres of the four quadrant-rectangles
quad_half = WORK_W_MM / 4                     # 125 mm
centres = {
    f"({-quad_half:.0f},{quad_half:.0f})":   (-quad_half,  quad_half),   # TL
    f"({quad_half:.0f},{quad_half:.0f})":    ( quad_half,  quad_half),   # TR
    f"({-quad_half:.0f},{-quad_half:.0f})":  (-quad_half, -quad_half),   # BL
    f"({quad_half:.0f},{-quad_half:.0f})":   ( quad_half, -quad_half),   # BR
}

# load a TrueType font (falls back if unavailable)
try:
    font = ImageFont.truetype("arial.ttf", FONT_SIZE_PT)
except OSError:
    font = ImageFont.load_default()

for label, (xm, ym) in centres.items():
    px = world2px(xm, ym)
    draw.line([(px[0]-10, px[1]), (px[0]+10, px[1])], fill=0, width=2)
    draw.line([(px[0], px[1]-10), (px[0], px[1]+10)], fill=0, width=2)
    draw.text((px[0] + 12, px[1] - 12), label, fill=0, font=font)

# ---------- “Save As…” dialog ----------
import tkinter as tk
from tkinter import filedialog

# 1) Hide the root window that Tkinter needs
root = tk.Tk()
root.withdraw()

# 2) Ask where to save – default name & .png extension
dest_path = filedialog.asksaveasfilename(
    title="Save AprilTag board as…",
    initialfile="apriltag_board"+str(WORK_W_MM)+"x"+str(WORK_H_MM)+".png",
    defaultextension=".png",
    filetypes=[("PNG image", "*.png"), ("All files", "*.*")]
)

# 3) If the user picked a location, write the file there
if dest_path:
    board.save(dest_path, dpi=(DPI, DPI))
    print(f"Saved image to '{dest_path}'. Print at 100 % scale.")
else:
    print("Save cancelled – no file written.")
#endregion