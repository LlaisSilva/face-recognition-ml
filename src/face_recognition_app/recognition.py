import numpy as np

from .config import DEFAULT_THRESHOLD

def calculate_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two normalized embeddings"""

    similarity = np.dot(embedding1, embedding2)

    return float(similarity)


def recognize_face(embedding, embeddings_database,   threshold=DEFAULT_THRESHOLD):
    """Compare one face embeddings against all embeddings in the database
     Return:
        name: recognized student or unknown
        best_similarity: highest similarity found"""

    best_name= "Unknown"

    best_similarity = -1.0

    for person_name, person_embeddings in embeddings_database.items():

        for person_embedding in person_embeddings:

            similarity = calculate_similarity(embedding, person_embedding)

            if similarity> best_similarity:
                best_similarity = similarity
                best_name = person_name


    if best_similarity<threshold:
        best_name="unknown"


    return best_name, best_similarity

