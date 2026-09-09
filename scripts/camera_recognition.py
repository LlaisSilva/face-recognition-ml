import cv2

from face_recognition_app.face_engine import FaceEngine
from face_recognition_app.camera import Camera
from face_recognition_app.recognition import recognize_face
from face_recognition_app.visualization import draw_face_result
from face_recognition_app.embeddings import load_embeddings_database
from face_recognition_app.config import CAMERA_EMBEDDINGS_FILE

def main():
    print("=" * 60)
    print("FACE RECOGNITION - CAMERA")
    print("=" * 60)

    face_engine = FaceEngine()

    database = load_embeddings_database(CAMERA_EMBEDDINGS_FILE)

    print(f"Students loaded: {len(database)}")

    camera = Camera(camera_id=0)

    print("Camera started.")
    print("Press Q to exit.")
    recognized = False

    while True:
        frame = camera.read()

        if frame is None:
            print("Warning: could not read frame")
            continue

        faces = face_engine.get_face(frame)

        for face in faces:
            embedding = face.normed_embedding

            name, similarity = recognize_face(embedding, database)

            print(
                f"{name} | "
                f"Similarity: {similarity:.4f}"
            )

            frame = draw_face_result(frame, face, name)

            recognized =True

        cv2.imshow("Face Recognition", frame)
        if recognized:
            cv2.waitKey(10000)
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()

if __name__ == "__main__":
    main()
