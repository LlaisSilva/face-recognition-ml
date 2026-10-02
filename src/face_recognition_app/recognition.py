import numpy as np


def cosine_similarity(embedding_1, embedding_2):
    """Calculate cosine similarity between two normalized embeddings."""
    return float(np.dot(embedding_1, embedding_2))


def recognize_face(embedding, embeddings_database, threshold=0.45):
    """Compare a face embedding against the local embeddings database."""

    best_identifier = "unknown"
    best_person_type = None
    best_similarity = -1.0

    for identifier, person_data in embeddings_database.items():
        person_type = person_data.get("person_type")
        person_embeddings = person_data.get("embeddings", [])

        for stored_embedding in person_embeddings:
            similarity = cosine_similarity(embedding, stored_embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_identifier = identifier
                best_person_type = person_type

    if best_similarity < threshold:
        return "unknown", None, best_similarity

    return best_identifier, best_person_type, best_similarity