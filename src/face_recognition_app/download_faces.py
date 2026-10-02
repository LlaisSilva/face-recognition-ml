from pathlib import Path

import cv2
import numpy as np

from .api_client import (
    download_enrollment_image
)

from .config import (
    BACKEND_IMAGES_DIR
)


def download_job_images(job):
    """ Downloading all images """

    job_id = job["job_id"]

    identifier = job.get(
        "identifier",
        "unknown"
    )

    photos = job.get(
        "photos",
        []
    )

    if not photos:

        raise ValueError(
            f"Job {job_id} "
            "does not contain photos."
        )


    job_dir = (
        BACKEND_IMAGES_DIR / job_id
    )

    job_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    downloaded_images = []


    for photo in sorted(
        photos,
        key=lambda item: item["sample_index"]
    ):

        sample_index = int(
            photo["sample_index"]
        )

        url = photo["url"]


        print(
            f"[IMAGE] Downloading "
            f"{identifier} "
            f"sample={sample_index}"
        )


        image_bytes = download_enrollment_image(
            url
        )


        if not image_bytes:

            print(
                f"[IMAGE ERROR] "
                f"Empty image "
                f"sample={sample_index}"
            )

            continue


        file_path = (
            job_dir
            / f"photo_{sample_index:02d}.jpg"
        )


        with open(
            file_path,
            "wb"
        ) as file:

            file.write(
                image_bytes
            )


        # Converte bytes para imagem OpenCV
        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )


        if image is None:

            print(
                f"[IMAGE ERROR] "
                f"Could not decode "
                f"{file_path}"
            )

            continue


        downloaded_images.append(
            {
                "sample_index": sample_index,
                "path": file_path,
                "image": image
            }
        )


        print(
            f"[IMAGE] OK "
            f"{file_path}"
        )


    return downloaded_images