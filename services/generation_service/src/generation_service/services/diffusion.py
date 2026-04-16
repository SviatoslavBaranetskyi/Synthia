import logging
from threading import Lock

import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
from PIL import Image

from generation_service.core.config import settings
from generation_service.services.scheduler import build_scheduler

logger = logging.getLogger(__name__)


class DiffusionService:
    def __init__(self) -> None:
        self._pipeline: StableDiffusionPipeline | None = None
        self._img2img_pipeline: StableDiffusionImg2ImgPipeline | None = None
        self._load_lock = Lock()

    def _resolve_torch_dtype(self) -> torch.dtype:
        if settings.PRECISION == "fp32":
            return torch.float32

        if settings.PRECISION == "fp16":
            return torch.float16

        if not settings.DEVICE.startswith("cuda") or not torch.cuda.is_available():
            return torch.float32

        device_name = torch.cuda.get_device_name(0).lower()

        # GTX 16xx cards are known to produce NaNs/black images with fp16
        # diffusion inference.
        if "gtx 16" in device_name:
            logger.warning(
                "Detected %s, using float32 to avoid fp16 black-image artifacts",
                torch.cuda.get_device_name(0),
            )
            return torch.float32

        return torch.float16

    def load_model(self) -> None:
        if self._pipeline is not None:
            return

        with self._load_lock:
            if self._pipeline is not None:
                return

            logger.info("Loading diffusion model from %s", settings.MODEL_PATH)
            torch_dtype = self._resolve_torch_dtype()

            pipeline = StableDiffusionPipeline.from_single_file(
                settings.MODEL_PATH,
                torch_dtype=torch_dtype,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
            )

            pipeline.safety_checker = None
            pipeline.feature_extractor = None
            pipeline.scheduler = build_scheduler(
                sampler="dpmpp_sde_karras",
                scheduler_config=pipeline.scheduler.config,
            )
            pipeline.enable_attention_slicing()
            pipeline.vae.enable_tiling()

            if settings.DEVICE.startswith("cuda"):
                pipeline.enable_model_cpu_offload()
            else:
                pipeline.to(settings.DEVICE)

            self._pipeline = pipeline
            self._img2img_pipeline = StableDiffusionImg2ImgPipeline(
                **pipeline.components
            )
            logger.info("Diffusion model loaded with dtype=%s", torch_dtype)

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        sampler: str = "dpmpp_sde_karras",
        steps: int = 30,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: int | None = None,
    ):
        self.load_model()

        assert self._pipeline is not None

        self._pipeline.scheduler = build_scheduler(
            sampler=sampler,
            scheduler_config=self._pipeline.scheduler.config,
        )

        generator = None
        if seed is not None:
            generator = torch.Generator(device=settings.DEVICE).manual_seed(seed)

        result = self._pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
            output_type="pil",
        )

        return result.images[0]

    def edit_image(
        self,
        prompt: str,
        image: Image.Image,
        negative_prompt: str = "",
        sampler: str = "dpmpp_sde_karras",
        steps: int = 30,
        guidance_scale: float = 7.5,
        strength: float = 0.65,
        seed: int | None = None,
    ):
        self.load_model()

        assert self._img2img_pipeline is not None
        self._img2img_pipeline.scheduler = build_scheduler(
            sampler=sampler,
            scheduler_config=self._img2img_pipeline.scheduler.config,
        )

        generator = None
        if seed is not None:
            generator = torch.Generator(device=settings.DEVICE).manual_seed(seed)

        prepared_image = image.convert("RGB")

        result = self._img2img_pipeline(
            prompt=prompt,
            image=prepared_image,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            strength=strength,
            generator=generator,
            output_type="pil",
        )

        return result.images[0]


diffusion_service = DiffusionService()
