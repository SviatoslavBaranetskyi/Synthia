import logging

from generation_service.services.diffusion import diffusion_service
from generation_service.services.queue import task_queue
from generation_service.utils.image import save_image

logger = logging.getLogger(__name__)


def run_worker() -> None:
    diffusion_service.load_model()

    while True:
        task = task_queue.next_task()

        try:
            image = diffusion_service.generate_image(**task.payload)
            image_path = save_image(image, task.task_id)
            task_queue.mark_done(task.task_id, image_path)
            logger.info("Generation task %s completed", task.task_id)
        except Exception as exc:
            logger.exception("Generation task %s failed", task.task_id)
            task_queue.mark_failed(task.task_id, str(exc))
