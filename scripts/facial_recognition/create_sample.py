from pathlib import Path
import random
import shutil

NUM_PEOPLE = 300
NUM_STUDENTS=150
NUM_INTRUDER=150

#Student:
#1 photo for register
#4 photos for validation
#2 photo for tests

REGISTRATION_PHOTOS = 1
VALIDATION_PHOTOS=4
TEST_PHOTOS=2

SEED= 32

# Paths

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR/"data"/"processed"
OUTPUT_DIR =PROCESSED_DIR/"sample_300"


# Functions

def find_identities(vggface2_path):
    """Find the folder of people inside VGGFace2
    Each folder represents an identity"""

    vggface2_path = Path(vggface2_path)

    identities= [
        folder
        for folder in vggface2_path.iterdir()
        if folder.is_dir()
    ]
    return identities


def select_people(identities):
    """Select 300 people randomly
    150 Students
    150 Intruders"""

    if len(identities) < NUM_PEOPLE:
        raise ValueError(
            f"the dataset has only {len(identities) } identities"

        )
    random.seed(SEED)

    sampled= random.sample(identities, NUM_PEOPLE)

    students = sampled[:NUM_STUDENTS]
    intruders=sampled[NUM_STUDENTS:]

    return students, intruders

def find_images(person):
    """Get all  images of a person"""
    extensions= {".jpg", ".jpeg", ".png"}

    images=[
        file
        for file in person.rglob("*")
        if file.is_file()
        and file.suffix.lower() in extensions

    ]
    return images

def create_folder(path):
    """Create folder if it doesn't exist yet"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

def copy_images(images, path, quantity):
    """Select 'quantity' of images and copy to the path"""
    if len(images) <quantity:
        raise ValueError(f"There is not enough images in {images[0].parent}")

    sampled = random.sample(images, quantity)
    create_folder(path)
    for num, image in enumerate(sampled, start=1):
        new_name= f"photo_{num:02d}{image.suffix.lower()}"
        file_path = path/new_name
        shutil.copy2(image, file_path)

def prepare_student(person,output_dir ):
    """Create the structure
    student/
        registered/
        validation/
        test/"""

    images = find_images(person)

    required_count = (
            REGISTRATION_PHOTOS
            + VALIDATION_PHOTOS
            + TEST_PHOTOS
    )
    if len(images) < required_count:
        print(f"Warning: {person.name} has only {len(images)} images. Skipping.")
        return False

    random.shuffle(images)
    registration = images[:REGISTRATION_PHOTOS]

    start = REGISTRATION_PHOTOS
    end= start+ VALIDATION_PHOTOS

    validation= images[start:end]

    test = images[
        end:end + TEST_PHOTOS

    ]

    person_dir = output_dir/person.name

    copy_images(registration, person_dir/"register", REGISTRATION_PHOTOS)

    copy_images(validation, person_dir/"validation", VALIDATION_PHOTOS)

    copy_images(test, person_dir /"test", TEST_PHOTOS)

    return True


def prepare_intruder(person,output_dir ):
    """Create the structure
    intruder/
        validation/
        test/"""

    images = find_images(person)

    required_count = (
            VALIDATION_PHOTOS
            + TEST_PHOTOS
    )
    if len(images) < required_count:
        print(f"Warning: {person.name} has only {len(images)} images. Skipping.")
        return False

    random.shuffle(images)


    validation= images[:VALIDATION_PHOTOS]

    test = images[
        VALIDATION_PHOTOS:
        VALIDATION_PHOTOS+ TEST_PHOTOS

    ]

    person_dir = output_dir/person.name


    copy_images(validation, person_dir/"validation", VALIDATION_PHOTOS)

    copy_images(test, person_dir /"test", TEST_PHOTOS)

    return True

# main exec

def prepare_dataset(vggface2_path):
    print("=" * 60)
    print("PREPARANDO DATASET")
    print("=" * 60)

    vggface2_path = Path(vggface2_path)

    if not vggface2_path.exists():
        raise FileNotFoundError(f"Dataset not found:{vggface2_path} ",)

    print("[1/4] Searching identities...")
    identities= find_identities(vggface2_path)
    print(
        f"Identidades encontradas: {len(identities)}"
    )

    print("[2/4] Selecting people...")

    students, intruders = select_people(identities)

    print(f"Alunos:   {len(students)}")
    print(f"Intrusos: {len(intruders)}")

    print("[3/4] Creating students dataset")
    students_dir = OUTPUT_DIR/"students"
    created_students_count =0

    for people in students:
        success= prepare_student(
            people,
             students_dir
        )
        if success:
            created_students_count+=1

    print(f"Students prepared: ")
    print(f"{created_students_count}")


    print("[4/4] Creating students dataset")
    intruders_dir = OUTPUT_DIR/"intruders"
    created_intruders_count =0

    for people in intruders:
        success= prepare_intruder(
            people,
             intruders_dir
        )
        if success:
            created_intruders_count+=1

    print(f"Students prepared: ")
    print(f"{created_intruders_count}")

    print("\n" + "=" * 60)
    print("DATASET PREPARADO!")
    print("=" * 60)

    print(f"\nResultado:")
    print(OUTPUT_DIR)

#Main

if __name__ == "__main__":
    vggface2_path = input(
        "cole o caminho do VGGFace2 baixado pelo KaggleHub"

    )
    prepare_dataset(vggface2_path)


