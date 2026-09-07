from face_recognition_app.embeddings import build_embeddings_database, save_embeddings_database

from face_recognition_app.face_engine import FaceEngine
from face_recognition_app.config import CAMERA_EMBEDDINGS_FILE

def main():
    print("=" * 60)
    print("CREATING CAMERA EMBEDDINGS DATABASE")
    print("=" * 60)

    face_engine = FaceEngine()

    database = build_embeddings_database("", face_engine)

    save_embeddings_database(database, CAMERA_EMBEDDINGS_FILE)

    print("Students registered:")
    for student_name, embeddings in database.items():
        print(f"{student_name}: {len(embeddings)} embedding(s)")

        print("=" * 60)
        print("CAMERA DATABASE CREATED")
        print("=" * 60)

if __name__ == "__main__":
    main()