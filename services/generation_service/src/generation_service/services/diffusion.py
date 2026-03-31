import logging
from threading import Lock

import torch
from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSDEScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    StableDiffusionPipeline,
)

from generation_service.core.config import settings

logger = logging.getLogger(__name__)


class DiffusionService:
    def __init__(self) -> None:
        self._pipeline: StableDiffusionPipeline | None = None
        self._load_lock = Lock()

    def _resolve_torch_dtype(self) -> torch.dtype:
        if settings.PRECISION == "fp32":
            return torch.float32

        if settings.PRECISION == "fp16":
            return torch.float16

        if not settings.DEVICE.startswith("cuda") or not torch.cuda.is_available():
            return torch.float32

        device_name = torch.cuda.get_device_name(0).lower()

        # GTX 16xx cards are known to produce NaNs/black images with fp16 diffusion inference.
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
            pipeline.scheduler = self._build_scheduler(
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
            logger.info("Diffusion model loaded with dtype=%s", torch_dtype)

    def _build_scheduler(self, sampler: str, scheduler_config):
        if sampler == "dpmpp_sde_karras":
            return DPMSolverSDEScheduler.from_config(
                scheduler_config,
                use_karras_sigmas=True,
            )

        if sampler == "dpmpp_2m_karras":
            return DPMSolverMultistepScheduler.from_config(
                scheduler_config,
                use_karras_sigmas=True,
                algorithm_type="dpmsolver++",
            )

        if sampler == "euler_a":
            return EulerAncestralDiscreteScheduler.from_config(scheduler_config)

        if sampler == "euler":
            return EulerDiscreteScheduler.from_config(scheduler_config)

        if sampler == "ddim":
            return DDIMScheduler.from_config(scheduler_config)

        raise ValueError(f"Unsupported sampler: {sampler}")

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

        self._pipeline.scheduler = self._build_scheduler(
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


diffusion_service = DiffusionService()
