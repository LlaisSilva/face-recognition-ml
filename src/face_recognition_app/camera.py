import cv2

class Camera:
    """Handle webcam input using OpenCV"""

    def __init__(self, camera_id):
        self.camera_id = camera_id
        self.capture = cv2.VideoCapture(camera_id)

        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open camera {camera_id}")


    def read(self):
        """Read one frame from camera"""

        ret, frame = self.capture.read()

        if not ret:
            raise RuntimeError("Could not read from camera")

        return frame

    def release(self):
        """Release camera"""
        self.capture.release()
        cv2.destroyAllWindows()