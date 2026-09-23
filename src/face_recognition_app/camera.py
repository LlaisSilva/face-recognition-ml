import cv2
import os


class Camera:
    """Handle camera input using OpenCV"""

    def __init__(self, source=0):
        if isinstance(source, str) and source.startswith("rtsp://"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

            print("Connecting to RTSP camera...")
            print(f"RTSP: {source}")

            self.capture = cv2.VideoCapture(
                source,
                cv2.CAP_FFMPEG
            )
        else:
            print("Connecting to local webcam...")
            print(f"Camera index: {source}")

            self.capture = cv2.VideoCapture(source)

        if not self.capture.isOpened():
            raise RuntimeError(
                f"Could not open camera {source}"
            )

        print("Camera connected!")

    def read(self):
        ret, frame = self.capture.read()

        if not ret:
            print("ERROR: OpenCV could not receive frame")
            return None

        return frame

    def release(self):
        self.capture.release()
        cv2.destroyAllWindows()