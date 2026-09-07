import sys
from pathlib import Path
import cv2

from face_recognition_app.face_engine import FaceEngine
from face_recognition_app.config import (DEFAULT_THRESHOLD, EMBEDDINGS_FILE)
from face_recognition_app.recognition import recognize_face
from face_recognition_app.embeddings import load_embeddings_database
from face_recognition_app.visualization import draw_face_result

def recognize_image(image_path, face_engine, embeddings_database):

    """Detect all faces in an image and recognize each one"""

    image_path = Path(image_path)

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"could not read image: {image_path}")

    faces = face_engine.get_face(image)

    if len(faces) ==0:
        print("No faces found in image")
        return

    print(f"faces detected: {len(faces)}")
    print("-"*60)

    #Recognize faces

    for index, face in enumerate(faces, start=1):
        embedding = face.normed_embedding

        name, similarity = recognize_face(embedding, embeddings_database, threshold=DEFAULT_THRESHOLD)

        print(
            f"Face {index}: "
            f"{name} | "
            f"Similarity: {similarity:.4f}"
        )
        image = draw_face_result(image, face, name)
        cv2.imshow("Face recognition: ", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    print("=" * 60)
    print("FACIAL RECOGNITION TEST")
    print("=" * 60)

    """if len(sys.argv)<2:
        print(
            "Usage:"
        )

        print(
            "python scripts/test_recognition.py "
            "path/to/image.jpg"
        )

        return"""

   # image_path = Path(sys.argv[1])
    image_path = input("Enter the path of the image: ").strip()
    image_path = Path(image_path)

    if not image_path.exists():

        print(
            f"Error: Image not found:"
            f"{image_path}"

        )
        print(EMBEDDINGS_FILE)
        print("Run create_embeddings.py first")

        return
    if not EMBEDDINGS_FILE.exists():
        print(
            "\nError: Embeddings database "
            "not found:"
        )

        print(
            EMBEDDINGS_FILE
        )

        print(
            "\nRun create_embeddings.py first."
        )

        return

    print("Loading face engine...")
    face_engine = FaceEngine()

    print("Loading embeddings databse...")

    embeddings_database = load_embeddings_database(EMBEDDINGS_FILE)

    print(f"Students in database: {len(embeddings_database)}")

    recognize_image(image_path, face_engine, embeddings_database)

    print("="*60)
    print("TEST FINISHED")
    print("="*60)

if __name__ == "__main__":
    main()