from pathlib import Path

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

CAMERA_EMBEDDINGS_FILE = EMBEDDINGS_DIR/"banco_camera.pkl"


RESULTS_DIR = DATA_DIR/ "results"

MODEL_NAME="buffalo_l"

DET_SIZE = (640,640)

PROVIDERS = [
    "CPUExecutionProvider"
]

DEFAULT_THRESHOLD = 0.40