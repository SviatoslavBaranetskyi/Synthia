import os
import uuid

from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError

from generation_service.core.config import resolve_service_path, settings


class InvalidUploadedImageError(ValueError):
    pass


def save_image(image, task_id: str):
    output_dir = resolve_service_path(settings.OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{task_id}.png"
    path = os.path.join(str(output_dir), filename)

    image.save(path)

    return path


def save_uploaded_image(upload: UploadFile) -> str:
    upload_dir = resolve_service_path(settings.UPLOAD_DIR)
    os.makedirs(upload_dir, exist_ok=True)

    if upload.content_type and not upload.content_type.startswith("image/"):
        raise InvalidUploadedImageError("Uploaded file must be an image.")

    extension = os.path.splitext(upload.filename or "")[1].lower() or ".png"
    filename = f"{uuid.uuid4()}{extension}"
    path = os.path.join(str(upload_dir), filename)

    try:
        upload.file.seek(0)
        with Image.open(upload.file) as source:
            image = source.convert("RGB")
            image.save(path)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        if os.path.exists(path):
            os.remove(path)
        raise InvalidUploadedImageError(
            "Uploaded file is not a valid readable image."
        ) from exc

    return path
