import cv2
import pickle
from pathlib import Path


def generate_embeddings(image_path, face_engine):
    """Detect all the faces of an image and return an embedding for each face"""

    image_path = Path(image_path)

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Error it was not possible to read"
              f"{image_path}")
        return []

    faces = face_engine.get_face(image)

    if len(faces) == 0:
        print(f"Warning: No face found "
              f"{image_path}")
        return []

    embeddings = []

    for face in faces:
        embeddings.append(face.normed_embedding)

    return embeddings


def build_embeddings_database(student_dir, face_engine):
    """Iterate through all the students folders and create the embeddings database"""

    student_dir = Path(student_dir)

    database = {}

    if not student_dir.exists():
        raise FileNotFoundError(f"Folder not found"
                                f"{student_dir}")

    for students_folder in sorted(student_dir.iterdir()):
        if not students_folder.is_dir():
            continue

        student_name = students_folder.name

        register_dir = students_folder/"register"

        if not register_dir.exists():
            print(f"Warning: Register folder not found {student_name}")
            continue

        database[student_name] = []
        print(f"student: {student_name}")


        for image_path in sorted(register_dir.iterdir()):

            if image_path.suffix.lower() not in {".jpg",".jpeg",".png"}:
                continue

            embeddings = generate_embeddings(image_path, face_engine)

            if len(embeddings) == 0:
                continue

            if len(embeddings) > 1:
                print(
                f"Warning: More than one face found " f"in {image_path.name}. " f"Image ignored.")
                continue
            embedding = embeddings[0]
            database[student_name].append(embedding)

            print(f"Ok {image_path.name}")

        #if no valid image are found remove the student from database

        if len(database[student_name])==0:
            del database[student_name]
            print("Warning: No valid embedding ")
    return database



def save_embeddings_database(database, output_path):
    """Save the embedding database as a picke file"""

    output_path= Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path,"wb")as file:
        pickle.dump(database, file)

    print(f"Banco salvo em: "
          f"{output_path}")

def load_embeddings_database(database_path):
    """Load the embeddings database"""

    database_path = Path(database_path)

    with open(database_path,"rb")as file:
        database = pickle.load(file)

    return database
