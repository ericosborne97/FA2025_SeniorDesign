'''
You can use this to get the id of all cameras on your device
'''
from cv2_enumerate_cameras import enumerate_cameras
for cam in enumerate_cameras():           # cross-platform
    print(cam.index)  # index is what cv2 needs
