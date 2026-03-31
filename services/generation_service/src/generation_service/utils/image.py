import os

from generation_service.core.config import settings


def save_image(image, task_id: str):
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)

    filename = f"{task_id}.png"
    path = os.path.join(settings.OUTPUT_DIR, filename)

    image.save(path)

    return path
