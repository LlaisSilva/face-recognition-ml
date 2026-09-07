from insightface.app import FaceAnalysis
from .config import (
    MODEL_NAME,
    DET_SIZE,
    PROVIDERS
)
class FaceEngine:
    """Responsible for face detection and embeddings generation  using InsightFace"""


    def __init__(self):
        self.app = FaceAnalysis(
            name=MODEL_NAME,
            providers=PROVIDERS
        )

        self.app.prepare(
            ctx_id=-1,
            det_size=DET_SIZE
        )
    def get_face(self, image):
        """Detect all the faces presents images """

        return self.app.get(image)



