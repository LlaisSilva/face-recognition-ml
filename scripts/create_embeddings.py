from face_recognition_app.config import (STUDENTS_DIR, EMBEDDINGS_FILE)

from face_recognition_app.face_engine import (FaceEngine)

from face_recognition_app.embeddings import (build_embeddings_database, save_embeddings_database)

def main():
    print("=" * 60)
    print("CREATING EMBEDDINGS DATABASE")
    print("=" * 60)

    face_engine = FaceEngine()

    database = build_embeddings_database(STUDENTS_DIR, face_engine)

    save_embeddings_database(database, EMBEDDINGS_FILE)

    print("\n" + "=" * 60)
    print("DATABASE CREATED")
    print("=" * 60)

    print(
        f"Students registered: "
        f"{len(database)}"
    )


if __name__ == "__main__":
    main()