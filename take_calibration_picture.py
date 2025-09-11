import cv2
import time
import os

# Match your script's camera behavior
CAM_INDEX = 700
OUTPUT    = "calibration_image.png"

def main():
    cap = cv2.VideoCapture(CAM_INDEX)

    if not cap.isOpened():
        print(f"Error: Could not open camera index {CAM_INDEX}")
        return

    # Optional: give the camera a brief warm-up so exposure/white-balance settle
    warmup_frames = 10
    for _ in range(warmup_frames):
        ok, _ = cap.read()
        if not ok:
            time.sleep(0.01)
        else:
            time.sleep(0.01)

    # Grab one frame
    ok, frame = cap.read()
    if not ok or frame is None:
        print("Error: Failed to grab a frame from the camera.")
        cap.release()
        return

    # Save immediately
    cv2.imwrite(OUTPUT, frame)
    cap.release()

    print(f"Saved: {os.path.abspath(OUTPUT)}")

if __name__ == "__main__":
    main()
