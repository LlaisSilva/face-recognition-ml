import requests

from .config import (
    BACKEND_URL,
    AIBOX_API_KEY
)


HEADERS = {
    "X-AIBOX-API-Key": AIBOX_API_KEY
}

TIMEOUT = 30


def get_next_enrollment_job():
    """
    Fetch the next facial registration available
    for processing from the backend

    Endpoint:

    GET /api/v1/ai/enrollment-jobs/next
    """

    url = (
        f"{BACKEND_URL}"
        f"/api/v1/ai/enrollment-jobs/next"
    )

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=TIMEOUT
    )


    if response.status_code == 204:
        return None

    if not response.ok:

        print(
            "[BACKEND ERROR]"
            f" status={response.status_code}"
        )

        print(
            response.text
        )

        response.raise_for_status()

    return response.json()



def download_enrollment_image(url):
    """
    Downloads an image provided by the backend. The URL already includes a temporary token.
    Returns the image bytes.
    """

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=TIMEOUT
    )

    if not response.ok:

        print(
            "[IMAGE ERROR]"
            f" status={response.status_code}"
        )

        print(
            response.text
        )

        response.raise_for_status()

    return response.content



def get_face_gallery():
    """


    """

    url = (
        f"{BACKEND_URL}"
        f"/api/v1/ai/face-gallery"
    )

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=TIMEOUT
    )

    if not response.ok:

        print(
            "[GALLERY ERROR]"
            f" status={response.status_code}"
        )

        print(
            response.text
        )

        response.raise_for_status()

    return response.json()



def send_event(event):
    """

    Sends recognition event to the backend
    """

    url = (
        f"{BACKEND_URL}"
        f"/api/v1/ai/events/"
    )

    response = requests.post(
        url,
        headers=HEADERS,
        json=event,
        timeout=TIMEOUT
    )

    if not response.ok:

        print(
            "[EVENT ERROR]"
            f" status={response.status_code}"
        )

        print(
            response.text
        )

        response.raise_for_status()

    return response.json()