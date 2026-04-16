import logging
import os

from PIL import Image

from generation_service.services.diffusion import diffusion_service
from generation_service.services.instantid import instantid_service
from generation_service.services.queue import task_queue
from generation_service.utils.image import save_image

logger = logging.getLogger(__name__)


def run_worker() -> None:
    diffusion_service.load_model()

    while True:
        task = task_queue.next_task()
        source_image_path: str | None = None

        try:
            mode = task.payload.get("mode", "txt2img")
            payload = {k: v for k, v in task.payload.items() if k != "mode"}

            if mode == "identity_edit":
                source_image_path = payload.pop("source_image_path")
                payload.pop("strength", None)
                payload.pop("model_key", None)
                with Image.open(source_image_path) as source_image:
                    image = instantid_service.generate_identity_image(
                        reference_image=source_image,
                        **payload,
                    )
            elif mode == "img2img":
                source_image_path = payload.pop("source_image_path")
                payload.pop("identity_strength", None)
                payload.pop("conditioning_scale", None)
                payload.pop("width", None)
                payload.pop("height", None)
                payload.pop("model_key", None)
                with Image.open(source_image_path) as source_image:
                    image = diffusion_service.edit_image(
                        image=source_image,
                        **payload,
                    )
            else:
                payload.pop("model_key", None)
                image = diffusion_service.generate_image(**payload)

            image_path = save_image(image, task.task_id)
            task_queue.mark_done(task.task_id, image_path)
            logger.info("Generation task %s completed", task.task_id)
        except Exception as exc:
            logger.exception("Generation task %s failed", task.task_id)
            task_queue.mark_failed(task.task_id, str(exc))
        finally:
            if source_image_path and os.path.exists(source_image_path):
                os.remove(source_image_path)
