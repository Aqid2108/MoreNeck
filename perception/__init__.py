"""MoreNeck perception: camera frames -> tracked people with cues."""
import os
import sys

if sys.platform == "win32":
    # Media Foundation's hardware transforms make opening a webcam take 8-14 s; without them it
    # opens in ~2 s at the same 30 fps. OpenCV reads this once, so it must be set before cv2 is
    # imported: import `perception` before `cv2`.
    os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
