from pathlib import Path

from face_recognition_app.config import EMBEDDINGS_FILE
from face_recognition_app.api_client import get_next_enrollment_job
from face_recognition_app.download_faces import download_job_images
from face_recognition_app.face_engine import FaceEngine
from face_recognition_app.embeddings import build_job_embeddings, save_embeddings_database


def process_jobs():
    """Process enrollment jobs and generate a local embeddings database."""

    print("=" * 70)
    print("FACE EMBEDDING GENERATOR")
    print("=" * 70)

    print()
    print("[1] Loading InsightFace...")
    face_engine = FaceEngine()

    print()
    print("[2] Looking for enrollment jobs...")

    database = {}

    while True:
        try:
            job = get_next_enrollment_job()
        except Exception as error:
            print()
            print("[BACKEND ERROR]")
            print(error)
            break

        if job is None:
            print()
            print("[BACKEND] No more enrollment jobs.")
            break

        job_id = job.get("job_id")
        enrollment_id = job.get("enrollment_id")
        identifier = job.get("identifier", "unknown")
        person_type = job.get("person_type")

        print()
        print("=" * 70)
        print(f"[JOB] {job_id}")
        print(f"[ENROLLMENT] {enrollment_id}")
        print(f"[PERSON] {identifier}")
        print(f"[TYPE] {person_type}")
        print("=" * 70)

        try:
            downloaded_images = download_job_images(job)
        except Exception as error:
            print("[IMAGE ERROR]")
            print(error)
            continue

        if not downloaded_images:
            print("[JOB] No images downloaded.")
            continue

        print()
        print(f"[IMAGES] {len(downloaded_images)} downloaded.")

        result = build_job_embeddings(job=job, downloaded_images=downloaded_images, face_engine=face_engine)

        embeddings = result["embeddings"]
        samples = result["samples"]

        print()
        print(f"[RESULT] {len(embeddings)} embeddings generated.")

        if not embeddings:
            print("[RESULT] No valid embeddings generated.")
            continue

        database[identifier] = result

        print()
        print(f"[DATABASE] {identifier} added.")
        print(f"[DATABASE] Samples: {samples}")

        for index, embedding in enumerate(embeddings, start=1):
            print(f"[EMBEDDING {index}] dimensions={len(embedding)}")

    if not database:
        print()
        print("[DATABASE] No embeddings generated.")
        return

    save_embeddings_database(database, EMBEDDINGS_FILE)

    print()
    print("=" * 70)
    print("PROCESSING FINISHED")
    print("=" * 70)
    print(f"People processed: {len(database)}")

    total_embeddings = sum(len(person["embeddings"]) for person in database.values())

    print(f"Embeddings generated: {total_embeddings}")
    print(f"Database: {Path(EMBEDDINGS_FILE)}")
    print("=" * 70)


def main():
    """Run the enrollment job processing pipeline."""

    try:
        process_jobs()
    except KeyboardInterrupt:
        print()
        print("Process interrupted by user.")
    except Exception as error:
        print()
        print("[FATAL ERROR]")
        print(error)
        raise


if __name__ == "__main__":
    main()