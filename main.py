#region Documentation
'''
         (              *      (         )       )       (      (      (
   (     )\ )         (.  `     )\ )   ( /(    ( /(       )\ )   )\ )   )\ )    *   )
   )\   (()/(   (     )\))(   (()/(   )\())   )\())     (()/(  (()/(  (()/(  ` )  /(
 (((_)   /(_))  )\   ((_)()\   /(_)) ((_)\   ((_)\       /(_))  /(_))  /(_))  ( )(_))
 )\___  (_))   ((_)  (_()((_) (_))     ((_)   _((_)     (_))   (_))   (_))_| (_(_())
((/ __| | |    | __| |  \/  | / __|   / _ \  | \| |     | _ \  |_ _|  | |_   |_   _|
 | (__  | |__  | _|  | |\/| | \__ \  | (_) | | .` |     |   /   | |   | __|    | |
  \___| |____| |___| |_|  |_| |___/   \___/  |_|\_|     |_|_\  |___|  |_|      |_|

=========================================================r
main.py
=========================================================i
Robotic Integration in Field Tasks
This is a python based approach that uses computer vision to detect the real world
coordinates of an object. These object coordinates are then given to a robotic system,
DetlaX which picks and sorts the item.
=========================================================f
Author: Eric Osborne
Date:   Aug 15, 2025
This work is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives
4.0 International License, http://creativecommons.org/licenses/by-nc-nd/4.0/
=========================================================t
'''
#endregion

#region Libraries
import time
import cv2
import numpy as np
from ultralytics import YOLO
import serial
import threading
#endregion

#region Configurations
CAM_INDEX       = 700
YOLO_MODEL_PATH = "models/general.pt"
CONF_THRESH     = 0.30
H_FILE          = "H.npy" # homography: pixels -> world mm
ALLOWED_LABELS  = "cell phone" # or set to None to allow any
PICKUP_LABELS = "cell phone" # or set to None to allow any
PORT = "COM4" # port may be different for your device use a program such as termite
FULL_PICKUP_SEQUENCE = False

FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE  = 0.5
TEXT_COLOR  = (0, 165, 255)  # orange (BGR)
TEXT_THICK  = 1              # thinner = less bold
BG_COLOR    = (255, 255, 255)  # white background
LINE_TYPE   = cv2.LINE_AA
PAD         = 3               # padding around text inside background box
PICKUP_SEQUENCE_RUNNING = False # needed to prevent pickup loop from running endlessly
#endregion

#region Helper Functions
def draw_text_with_bg(img, text, org):
    """
    Draw text with a solid white background for readability.
    org is the bottom-left corner of the text (OpenCV convention).
    """
    (w, h), base = cv2.getTextSize(text, FONT, FONT_SCALE, TEXT_THICK)
    x, y = int(org[0]), int(org[1])
    tl = (max(x - PAD, 0), max(y - h - PAD, 0))
    br = (min(x + w + PAD, img.shape[1] - 1), min(y + base + PAD, img.shape[0] - 1))

    # Filled white rectangle behind the text
    cv2.rectangle(img, tl, br, BG_COLOR, thickness=-1)
    # Foreground text
    cv2.putText(img, text, (x, y), FONT, FONT_SCALE, TEXT_COLOR, TEXT_THICK, LINE_TYPE)

def safe_text_org_for_box(x1, y1, text_height):
    """
    Choose a text origin just above the box if there's room, else below it.
    Returns (x, y) for cv2.putText (bottom-left baseline).
    """
    above_y = y1 - 6
    if above_y - text_height - PAD >= 0:
        return (x1, above_y)
    else:
        return (x1, y1 + text_height + 6)

def pix2world(u, v, H):
    """Map image pixel (u,v) -> (X,Y) in mm using H (if available)."""
    if H is None:
        return u, v
    uv1 = np.array([u, v, 1.0], dtype=float)
    XY1 = H @ uv1
    if XY1[2] == 0:
        return u, v
    XY = XY1[:2] / XY1[2]
    return float(XY[0]), float(XY[1])

def send_serial_message(message):
    """send serial communication messages"""
    ser = serial.Serial(
        PORT,
        115200,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        xonxoff=False,
        rtscts=False,
        dsrdtr=False,
        timeout=1.0,  # read timeout (s)
        write_timeout=1.0,  # write timeout (s)
    )
    try:
        ser.write((message + "\n").encode("ascii"))  # append LF
        ser.flush()
        # Optional: read one line back (until LF) and print it
        resp = ser.readline().decode(errors="replace").strip()
        if resp:
            print(f"<< {resp}")
    finally:
        ser.close()

'''for the vacuum pump D1 ON D6 OFF.'''
def pickup(x, y):
    '''This tells the robot what to do when we are picking up an object'''
    global PICKUP_SEQUENCE_RUNNING
    try:
        print(f"G0 X{int(x)} Y{int(y)}")
        send_serial_message(f"G0 X{int(x)} Y{int(y)} Z-650") # hover above location
        time.sleep(1)
        if FULL_PICKUP_SEQUENCE:
            send_serial_message("G0 Z-700") # go down
            time.sleep(1)
            send_serial_message("M03 D6") # activate suction pump
            time.sleep(1)
            send_serial_message("M05 D6") # turn off pin (good house keeping)
            send_serial_message("G0 Z-600")
            time.sleep(1)
            send_serial_message("G0 X0 Y0 Z-600") # go to home
            time.sleep(1)
            send_serial_message("M03 D1") # deactivate suction pump
            time.sleep(.25)
            send_serial_message("M05 D1") # turn off pin (good house keping)
            send_serial_message("G28")
    finally:
        PICKUP_SEQUENCE_RUNNING = False   # guarantee reset even on error
#endregion

#region OnStart
# try loading homography
H = None
try:
    H = np.load(H_FILE)
    assert H.shape == (3, 3)
    print(f"Loaded homography from {H_FILE}")
except Exception as e:
    print(f"[WARN] Could not load H from {H_FILE}: {e}\n"
          "       Falling back to pixel coordinates. YOU NEED TO RUN calibration.py first")

model = YOLO(YOLO_MODEL_PATH)

send_serial_message("G1 F200 A5000 J1200000") # set speed parameters

# try loading camera
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError(f"Could not open camera index {CAM_INDEX}")
#endregion

#region Loop
prev_t = time.time()
while True:
    ok, frame = cap.read()
    if not ok:
        print("Frame grab failed, retrying...")
        time.sleep(0.01)
        continue

    # Run detection
    results = model.predict(frame, conf=CONF_THRESH, verbose=False)
    r = results[0]

    # Draw detections
    if r.boxes is not None:
        for box in r.boxes:
            cls_id = int(box.cls)
            label  = r.names[cls_id]
            if ALLOWED_LABELS is not None and label not in ALLOWED_LABELS:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            X, Y   = pix2world(cx, cy, H)

            # start pickup sequence on a separate thread if one is not already running
            if not PICKUP_SEQUENCE_RUNNING:
                print("launching pickup")
                pickup_sequence = threading.Thread(target=pickup, args=(X,Y), daemon=True)
                PICKUP_SEQUENCE_RUNNING = True # keeps track if one is running
                pickup_sequence.start()  # launches the thread

            conf   = float(box.conf[0]) if box.conf is not None else 0.0

            # Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Compose label text
            if H is not None:
                txt = f"{label} {conf:.2f}  ({X:.0f},{Y:.0f}) mm"
            else:
                txt = f"{label} {conf:.2f}  (u={cx}, v={cy})"

            # Compute text height once for placement logic
            (_, text_h), _ = cv2.getTextSize(txt, FONT, FONT_SCALE, TEXT_THICK)
            org = safe_text_org_for_box(x1, y1, text_h)

            # Draw label with white background
            draw_text_with_bg(frame, txt, org)

            # Mark center
            cv2.circle(frame, (cx, cy), 3, TEXT_COLOR, -1, LINE_TYPE)

    # FPS overlay (top-left)
    now = time.time()
    fps = 1.0 / (now - prev_t) if now > prev_t else 0.0
    prev_t = now
    draw_text_with_bg(frame, f"FPS: {fps:.1f}", (10, 20))

    cv2.imshow("YOLO stream", frame)
    key = cv2.waitKey(1) & 0xFF
    if key in (27, ord('q')):  # Esc or q
        break
#endregion

cap.release()
cv2.destroyAllWindows()
