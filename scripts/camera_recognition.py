import time

import cv2

from face_recognition_app.face_engine import FaceEngine
from face_recognition_app.camera import Camera
from face_recognition_app.recognition import recognize_face
from face_recognition_app.visualization import draw_face_result
from face_recognition_app.embeddings import load_embeddings_database
from face_recognition_app.config import (
    CAMERA_ID,
    API_URL,
    CAMERA_EMBEDDINGS_FILE,
    EVENT_COOLDOWN
)
from face_recognition_app.events import build_event
from face_recognition_app.api_client import send_event


camera_url = 0


class EventManager:
    """Controls how often an event can be generated.
    Prevents the same person from generating
    an event on every camera frame.
    """

    def __init__(self, cooldown=10):
        self.cooldown = cooldown
        self.last_events = {}

    def can_send(self, event_key):
        """Returns True if the event can be sent."""

        current_time = time.monotonic()

        last_event_time = self.last_events.get(event_key)

        if last_event_time is None:
            self.last_events[event_key] = current_time
            return True

        elapsed_time = current_time - last_event_time

        if elapsed_time >= self.cooldown:
            self.last_events[event_key] = current_time
            return True

        return False


class CameraRecognition:

    def __init__( self,  camera_id, api_url,  embeddings_database):
        self.camera_id = camera_id
        self.api_url = api_url
        self.embeddings_database = embeddings_database

        self.face_engine = FaceEngine()

        self.camera = Camera(camera_id)

        self.event_manager = EventManager(
            cooldown=EVENT_COOLDOWN
        )

    def process_face(self, frame, face):
        """Process each detected face."""

        embedding = face.normed_embedding

        name, similarity = recognize_face( embedding,  self.embeddings_database )

        confidence = float(face.det_score)

        print(
            f"Face detected | "
            f"{name} | "
            f"Similarity: {similarity:.4f} | "
            f"Confidence: {confidence:.4f}"
        )

        frame = draw_face_result( frame, face, name )

        self.process_event(  name=name, similarity=similarity,  confidence=confidence)

        return frame

    def process_event( self,  name,   similarity,  confidence  ):
        """Creates and sends an event for the detected
        face if the cooldown allows it.
        """

        if name.lower() == "unknown":

            event_type = "UNKNOWN_FACE"
            person_id = None
            event_key = f"{self.camera_id}:unknown"

        else:

            event_type = "FACE_RECOGNIZED"
            person_id = name

            event_key = (
                f"{self.camera_id}:"
                f"person:{person_id}"
            )

        if not self.event_manager.can_send(event_key):
            return

        event = build_event(
            event_type=event_type,
            camera_id=self.camera_id,
            person_id=person_id,
            confidence=confidence,
            match_similarity=similarity
        )

        self.send_event(event)

    def send_event(self, event):
        """Sends an event to the backend API."""

        try:

            response = send_event(event,  self.api_url  )

            print(
                f"[EVENT] "
                f"{event['event_type']} "
                f"sent successfully"
            )

            print(
                f"[API] {response}"
            )

        except Exception as error:

            print(
                f"[API ERROR] "
                f"Could not send event: {error}"
            )

    def run(self):
        print("=" * 60)
        print("FACE RECOGNITION - CAMERA")
        print("=" * 60)

        print(f"Camera ID: {self.camera_id}")

        print(f"students loaded:"
              f"{len(self.embeddings_database)}")

        print(f"Event cooldown: "
              f"{EVENT_COOLDOWN}")

        print()
        print("Camera started.")
        print("Press Q to exit.")
        print("=" * 60)

        try:
            while True:
                frame = self.camera.read()
                if frame is None:
                    print("Warning: could not read frame")
                    continue

                faces = self.face_engine.get_face(frame)

                print(f"Faces detected: {len(faces)}")

                for face in faces:
                    frame = self.process_face(frame, face)

                cv2.imshow("Face recognition", frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
        finally:
            self.camera.release()
            cv2.destroyAllWindows()

            print("Camera stopped")

def main():
    database = load_embeddings_database(
        CAMERA_EMBEDDINGS_FILE
    )

    app = CameraRecognition(
        camera_id=camera_url,
        api_url=API_URL,
        embeddings_database=database
    )

    app.run()


if __name__ == "__main__":
    main()