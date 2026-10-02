import threading
import time

import cv2

from face_recognition_app.face_engine import FaceEngine
from face_recognition_app.camera import Camera
from face_recognition_app.recognition import recognize_face
from face_recognition_app.visualization import draw_face_result
from face_recognition_app.config import CAMERA_ID, CAMERA_ID_2, EVENT_COOLDOWN, EMBEDDINGS_FILE
from face_recognition_app.events import build_event
from face_recognition_app.api_client import send_event
from face_recognition_app.embeddings import load_embeddings_database


class EventManager:
    """Control how often an event can be generated."""

    def __init__(self, cooldown=10):
        self.cooldown = cooldown
        self.last_events = {}

    def can_send(self, event_key):
        """Return True if the event can be sent."""

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

    def __init__(self, camera_id, embeddings_database):
        self.camera_id = camera_id
        self.embeddings_database = embeddings_database
        self.face_engine = FaceEngine()
        self.camera = Camera(camera_id)
        self.event_manager = EventManager(cooldown=EVENT_COOLDOWN)

    def process_face(self, frame, face):
        """Process a detected face."""

        embedding = face.normed_embedding

        identifier, person_type, similarity = recognize_face(
            embedding,
            self.embeddings_database
        )

        confidence = float(face.det_score)

        print(
            f"Face detected | {identifier} | "
            f"Similarity: {similarity:.4f} | "
            f"Confidence: {confidence:.4f}"
        )

        frame = draw_face_result(
            frame,
            face,
            identifier
        )

        self.process_event(
            identifier=identifier,
            person_type=person_type,
            similarity=similarity,
            confidence=confidence
        )

        return frame

    def process_event(self, identifier, person_type, similarity, confidence):
        """Create and send an event to the backend."""

        if identifier.lower() == "unknown":
            event_type = "person_unrecognized"
            person_id = None
            event_key = f"{self.camera_id}:unknown"
        else:
            event_type = "person_recognized"
            person_id = identifier
            event_key = f"{self.camera_id}:person:{person_id}"

        if not self.event_manager.can_send(event_key):
            return

        event = build_event(
            event_type=event_type,
            camera_id=str(self.camera_id),
            identifier=person_id,
            person_type=person_type,
            confidence=confidence,
            match_similarity=similarity
        )

        self.send_event(event)

    def send_event(self, event):
        """Send an event to the backend API."""

        try:
            response = send_event(event)

            print(
                f"[EVENT] {event['event_type']} "
                f"sent successfully"
            )

            print(f"[API] {response}")

        except Exception as error:
            print(f"[API ERROR] Could not send event: {error}")

    def run(self):
        """Run face recognition using the configured camera."""

        print("=" * 60)
        print("FACE RECOGNITION - CAMERA")
        print("=" * 60)

        print(f"Camera ID: {self.camera_id}")
        print(f"Gallery loaded: {len(self.embeddings_database)} people")
        print(f"Event cooldown: {EVENT_COOLDOWN}")

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

            print(f"Camera stopped {self.camera_id}")


def run_camera(camera_id, embeddings_database):
    """Create and run one camera."""

    app = CameraRecognition(
        camera_id=camera_id,
        embeddings_database=embeddings_database
    )

    app.run()


def main():
    """Load the local embeddings database and start the cameras."""

    print("=" * 60)
    print("STARTING FACE RECOGNITION")
    print("=" * 60)

    print("[DATABASE] Loading local embeddings database...")

    try:
        database = load_embeddings_database(EMBEDDINGS_FILE)

    except Exception as error:
        print(
            f"[DATABASE ERROR] "
            f"Could not load embeddings database: {error}"
        )

        return

    if not database:
        print("[DATABASE ERROR] Embeddings database is empty.")
        return

    print(f"[DATABASE] {len(database)} people loaded.")

    for identifier, person in database.items():
        person_type = person.get("person_type")
        embeddings = person.get("embeddings", [])

        print(
            f"[DATABASE] {identifier} | "
            f"type={person_type} | "
            f"embeddings={len(embeddings)}"
        )

    print()
    print(f"Camera 1: {CAMERA_ID}")
    print(f"Camera 2: {CAMERA_ID_2}")

    camera_1 = threading.Thread(
        target=run_camera,
        args=(CAMERA_ID, database),
        daemon=True
    )

    camera_1.start()

    camera_2 = None

    if CAMERA_ID_2 is not None:
        camera_2 = threading.Thread(
            target=run_camera,
            args=(CAMERA_ID_2, database),
            daemon=True
        )

        camera_2.start()

    try:
        while True:
            if not camera_1.is_alive():
                print("Camera 1 stopped")
                break

            if camera_2 is not None and not camera_2.is_alive():
                print("Camera 2 stopped")
                break

            threading.Event().wait(1)

    except KeyboardInterrupt:
        print()
        print("Stopping cameras...")


if __name__ == "__main__":
    main()