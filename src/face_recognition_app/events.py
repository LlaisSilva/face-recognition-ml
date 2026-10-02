from uuid import uuid4
from datetime import datetime, timezone

def generate_event_id():
    return f"AI-FACE-{uuid4()}"


def build_event(event_type,camera_id, confidence=None, identifier=None, person_type=None, track_id=None, match_similarity=None):
    event ={
        "external_event_id": generate_event_id(),
        "event_type": event_type,
        "camera_id": str(camera_id),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "track_id": track_id,
        "confidence":(round(float(confidence),4)
                      if confidence is not None
                      else None),
        "match_similarity": (round(float(match_similarity), 4)
                             if match_similarity is not None
                             else None),
    }

    if event_type  == "person_recognized":
        event["identifier"] = identifier
        event["person_type"] = person_type
    return event




