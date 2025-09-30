#region Documentation
'''
IMPORTANT NOTE: THIS ONLY WORKS WITH .PNG FILES JPG FILES WILL PROVIDE INCORRECT SCALING
Manual homography calibration by clicking four points in an image.
Saves H.npy that maps image pixels -> world mm using the WORLD dict.

Controls:
- Left-click: add a point (in order: tag IDs [0, 1, 2, 3])
- u : undo last point
- r : reset all points
- s or Enter : compute & save H.npy (when 4 points selected)
- Esc : quit

=========================================================r
manual_calibration.py
=========================================================i
This is used to generate a calibration file H.npy which is used to get the real world
coordinates of an object from an image.
=========================================================f
Steps:
1) Print out calibration sheet using calibration_sheet_generator.py
2) Edit the following variable with corner coordinates
    0) top left
    1) top right
    2) bottom right
    3) bottom left
=========================================================t'''
# World (mm) coordinates of each tag center (2D plane)
WORLD = {
    0: (-250,   250),
    1: (250, 250),
    2: (250, -250),
    3: (-250,   -250),
}
"""
3) Home robot using command, G28 to center actuator
4) Place calibration sheet underneath with origin directly under actuator
   Also make sure that you grid is rotated correctly. Do G1 X-100 to make sure your X-Axis
   axis is rotated correctly. Then do G1 Y-100 and such to make sure your Y-Axis is correct
5) Then run this script and select the points in order of
    0) top left
    1) top right
    2) bottom right
    3) bottom left
6) Press s to save, not ctrl s, just s
=========================================================t
Author: Eric Osborne
Date:   Aug 15, 2025
This work is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives
4.0 International License, http://creativecommons.org/licenses/by-nc-nd/4.0/
=========================================================t
"""
#endregion

#region Imports
import cv2
import numpy as np
#endregion







#region Early Helper
def listAllCameras():
    '''prints out a list of all cameras'''
    from cv2_enumerate_cameras import enumerate_cameras
    for cam in enumerate_cameras():           # cross-platform
        print(cam.index)  # index is what cv2 needs
#endregion

#region Variables
IMAGE_PATH = "calibration_image.png"
TAG_ORDER = [0, 1, 2, 3]   # Click points in this order

# try getting calibration image
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Can't read image at: {IMAGE_PATH}. The wrong camera may"
                            f"be selected. Try changing \"camera\" varaible to one of"
                            f"these "+str(listAllCameras()))

display = img.copy()
win = "Calibration Interface"
cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

clicked_pts = []  # [(x, y), ...] in the order of TAG_ORDER
#endregion

#region Helper Functions
def redraw():
    """Refresh the overlay with points and instructions."""
    global display
    display = img.copy()

    # Draw previously selected points
    for i, (x, y) in enumerate(clicked_pts):
        tag_id = TAG_ORDER[i]
        cv2.circle(display, (int(x), int(y)), 6, (0, 255, 0), -1)
        cv2.putText(display, f"id {tag_id}", (int(x)+8, int(y)-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    # Instruction banner
    line1 = "Left-click to add points. Order: Top Left, Top Right, Bottom Right, Bottom Left"
    next_idx = len(clicked_pts)
    if next_idx < len(TAG_ORDER):
        line2 = f"Click the pixel for tag id {TAG_ORDER[next_idx]}"
    else:
        line2 = "Press 's' or Enter to save (NOT CTRL S), 'u' undo, 'r' reset, Esc quit"

    pad = 8
    overlay = display.copy()
    bar_h = 60
    cv2.rectangle(overlay, (0, 0), (display.shape[1], bar_h), (0, 0, 0), -1)
    alpha = 0.55
    cv2.addWeighted(overlay, alpha, display, 1 - alpha, 0, display)
    cv2.putText(display, line1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(display, line2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1, cv2.LINE_AA)

def on_mouse(event, x, y, flags, userdata):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_pts) < 4:
            clicked_pts.append((float(x), float(y)))
            redraw()
#endregion

cv2.setMouseCallback(win, on_mouse)
redraw()

#region Loop
while True:
    cv2.imshow(win, display)
    key = cv2.waitKey(20) & 0xFF

    if key in (27, ):  # Esc
        break
    elif key in (ord('u'), ):  # undo
        if clicked_pts:
            clicked_pts.pop()
            redraw()
    elif key in (ord('r'), ):  # reset
        clicked_pts.clear()
        redraw()
    elif key in (ord('s'), 13):  # save / Enter
        if len(clicked_pts) == 4:
            # Build pixel and world arrays in matching order
            pix_pts = np.float32(clicked_pts)
            world_pts = np.float32([WORLD[i] for i in TAG_ORDER])

            H, mask = cv2.findHomography(pix_pts, world_pts, method=0)
            if H is None:
                print("Homography failed. Try clicking more accurately and retry.")
            else:
                np.save("H.npy", H)
                print("Saved homography → H.npy")
                try:
                    H_inv = np.linalg.inv(H)
                    np.save("H_inv.npy", H_inv)  # <-- save inverse separately
                    print("Saved inverse homography → H_inv.npy")
                except np.linalg.LinAlgError:
                    pass

                break
        else:
            print(f"Need 4 points, currently have {len(clicked_pts)}.")
#endregion

cv2.destroyAllWindows()
