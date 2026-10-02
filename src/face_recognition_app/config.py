
from pathlib import Path
import os

from dotenv import load_dotenv

from face_recognition_app.api_client import BACKEND_URL

load_dotenv()
PROJECT_ROOT= Path(__file__).resolve().parents[2]

#Dataset

DATA_DIR = PROJECT_ROOT/"data"

PROCESSED_DIR = DATA_DIR/"processed"

SAMPLE_DIR = PROCESSED_DIR/"sample_300"

STUDENTS_DIR = SAMPLE_DIR/"students"

INTRUDERS_DIR = SAMPLE_DIR/"intruders"

EMBEDDINGS_DIR = PROJECT_ROOT/"models"

EMBEDDINGS_FILE = (
    EMBEDDINGS_DIR/"banco_embeddings.pkl"
)

CAMERA_DATABASE_DIR = DATA_DIR/"bd"

CAMERA_EMBEDDINGS_FILE = EMBEDDINGS_DIR/"banco_camera.pkl"


CAMERA_ID = os.getenv("CAMERA_ID")
if CAMERA_ID.isdigit():
    CAMERA_ID = int(CAMERA_ID)

CAMERA_ID_2 = os.getenv("CAMERA_ID_2")


RESULTS_DIR = DATA_DIR/ "results"

MODEL_NAME="buffalo_l"

DET_SIZE = (640,640)

PROVIDERS = [
    "CPUExecutionProvider"
]


DEFAULT_THRESHOLD = 0.40

EVENT_COOLDOWN = 10

BACKEND_URL = os.getenv("BACKEND_URL")
AIBOX_API_KEY =os.getenv("AIBOX_API_KEY")
print("CAMERA_ID:", repr(CAMERA_ID))
BACKEND_IMAGES_DIR = (
    DATA_DIR / "backend_images"
)

BACKEND_IMAGES_DIR.mkdir(
    parents=True,
    exist_ok=True
)