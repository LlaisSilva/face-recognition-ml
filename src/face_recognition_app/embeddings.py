import pickle
from pathlib import Path


def generate_embedding(image, face_engine):
    """
    Generate a facial embedding for an image.

    """
    embedding = face_engine.generate_embedding(image)
    return embedding


def build_job_embeddings(job, downloaded_images, face_engine):
    """
    Generate facial embeddings for all images in a job.
    """
    embeddings = []
    samples = []

    for item in downloaded_images:
        sample_index = item["sample_index"]
        image = item["image"]

        print(f"[EMBEDDING] Processing sample {sample_index}")

        embedding = generate_embedding(image, face_engine)

        if embedding is None:
            print(f"[EMBEDDING] FAILED sample={sample_index}")
            continue

        embeddings.append(embedding)
        samples.append(sample_index)

        print(f"[EMBEDDING] OK sample={sample_index} dimensions={len(embedding)}")

    return {
        "identifier": job.get("identifier"),
        "person_type": job.get("person_type"),
        "enrollment_id": job.get("enrollment_id"),
        "job_id": job.get("job_id"),
        "embeddings": embeddings,
        "samples": samples
    }


def save_embeddings_database(database, output_path):
    """
    Save the embeddings database locally.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as file:
        pickle.dump(database, file)

    print(f"[DATABASE] Saved: {output_path}")


def load_embeddings_database(database_path):
    """
    Load the local embeddings database.
    """
    database_path = Path(database_path)

    with open(database_path, "rb") as file:
        return pickle.load(file)