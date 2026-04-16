import logging
from pathlib import Path
from threading import Lock

import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps

from generation_service.core.config import resolve_service_path, settings
from generation_service.services.scheduler import build_scheduler

logger = logging.getLogger(__name__)


class InstantIDService:
    def __init__(self) -> None:
        self._pipeline = None
        self._face_app = None
        self._draw_kps = None
        self._cv2 = None
        self._np = None
        self._load_lock = Lock()

    def _round_to_multiple(self, value: int, multiple: int = 64) -> int:
        return max(multiple, int(round(value / multiple) * multiple))

    def _resolve_torch_dtype(self) -> torch.dtype:
        if settings.PRECISION == "fp32":
            return torch.float32

        if settings.PRECISION == "fp16":
            return torch.float16

        if not settings.DEVICE.startswith("cuda") or not torch.cuda.is_available():
            return torch.float32

        device_name = torch.cuda.get_device_name(0).lower()
        if "gtx 16" in device_name:
            logger.warning(
                (
                    "Detected %s, using float32 for InstantID "
                    "to avoid fp16 black-image artifacts"
                ),
                torch.cuda.get_device_name(0),
            )
            return torch.float32

        return torch.float16

    def _resolve_runtime_provider_list(self) -> list[str]:
        requested = [item.strip() for item in settings.INSTANTID_PROVIDERS.split(",")]

        try:
            import onnxruntime as ort
        except ImportError:
            return ["CPUExecutionProvider"]

        available = set(ort.get_available_providers())
        providers = [provider for provider in requested if provider in available]
        return providers or ["CPUExecutionProvider"]

    def _ctx_id(self) -> int:
        if settings.DEVICE.startswith("cuda") and torch.cuda.is_available():
            return 0
        return -1

    def _ensure_instantid_assets(self) -> tuple[str, str]:
        cache_dir = resolve_service_path(settings.INSTANTID_CACHE_DIR)
        cache_dir.mkdir(parents=True, exist_ok=True)

        adapter_path = hf_hub_download(
            repo_id=settings.INSTANTID_REPO_ID,
            filename="ip-adapter.bin",
            local_dir=str(cache_dir),
        )

        controlnet_config = hf_hub_download(
            repo_id=settings.INSTANTID_REPO_ID,
            filename="ControlNetModel/config.json",
            local_dir=str(cache_dir),
        )
        hf_hub_download(
            repo_id=settings.INSTANTID_REPO_ID,
            filename="ControlNetModel/diffusion_pytorch_model.safetensors",
            local_dir=str(cache_dir),
        )

        return adapter_path, str(Path(controlnet_config).parent)

    def load_model(self) -> None:
        if self._pipeline is not None and self._face_app is not None:
            return

        with self._load_lock:
            if self._pipeline is not None and self._face_app is not None:
                return

            try:
                import cv2
                import numpy as np
                from diffusers import ControlNetModel
                from diffusers.models.controlnets.multicontrolnet import (
                    MultiControlNetModel,
                )
                from insightface.app import FaceAnalysis
                from generation_service.vendor.instantid import (
                    pipeline_stable_diffusion_xl_instantid as instantid_pipeline,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "InstantID requires optional packages: "
                    "cv2, onnxruntime, and insightface. "
                    "Install them before using model_key=sdxl_instantid."
                ) from exc

            logger.info(
                "Loading InstantID pipeline with base model %s",
                settings.SDXL_MODEL_ID,
            )
            torch_dtype = self._resolve_torch_dtype()
            adapter_path, controlnet_path = self._ensure_instantid_assets()

            identity_controlnet = ControlNetModel.from_pretrained(
                controlnet_path,
                torch_dtype=torch_dtype,
            )
            structure_controlnet = ControlNetModel.from_pretrained(
                settings.INSTANTID_STRUCTURE_CONTROLNET_ID,
                torch_dtype=torch_dtype,
            )

            pipeline_cls = instantid_pipeline.StableDiffusionXLInstantIDPipeline
            pipeline = pipeline_cls.from_pretrained(
                settings.SDXL_MODEL_ID,
                controlnet=MultiControlNetModel(
                    [identity_controlnet, structure_controlnet]
                ),
                torch_dtype=torch_dtype,
            )

            if settings.DEVICE.startswith("cuda"):
                pipeline.enable_model_cpu_offload()
            else:
                pipeline.to(settings.DEVICE)

            pipeline.enable_attention_slicing()
            pipeline.vae.enable_tiling()
            pipeline.load_ip_adapter_instantid(adapter_path)

            providers = self._resolve_runtime_provider_list()
            face_app = FaceAnalysis(
                name="antelopev2",
                root=str(resolve_service_path(settings.INSTANTID_FACE_ANALYSIS_ROOT)),
                providers=providers,
            )

            self._pipeline = pipeline
            self._face_app = face_app
            self._draw_kps = instantid_pipeline.draw_kps

            # Keep imports local to avoid hard dependency during route import.
            self._cv2 = cv2
            self._np = np
            logger.info(
                "InstantID pipeline loaded with dtype=%s providers=%s",
                torch_dtype,
                providers,
            )

    def _iter_detection_candidates(self, image: Image.Image):
        base_image = ImageOps.exif_transpose(image).convert("RGB")
        min_side_targets = [None, 768, 1024]
        rotations = [0, 90, 270]

        for rotation in rotations:
            rotated = (
                base_image
                if rotation == 0
                else base_image.rotate(rotation, expand=True)
            )

            for target in min_side_targets:
                candidate = rotated
                if target is not None:
                    min_side = min(candidate.size)
                    if min_side < target:
                        scale = target / min_side
                        candidate = candidate.resize(
                            (
                                round(candidate.width * scale),
                                round(candidate.height * scale),
                            ),
                            Image.Resampling.LANCZOS,
                        )

                yield candidate, rotation

    def _extract_face_features(self, image: Image.Image):
        assert self._face_app is not None

        original_size = image.size
        det_sizes = [(640, 640), (1024, 1024), (1280, 1280)]
        best_face = None
        best_image = None
        best_area = -1.0

        for det_size in det_sizes:
            self._face_app.prepare(ctx_id=self._ctx_id(), det_size=det_size)

            for candidate_image, rotation in self._iter_detection_candidates(image):
                faces = self._face_app.get(
                    self._cv2.cvtColor(
                        self._np.array(candidate_image),
                        self._cv2.COLOR_RGB2BGR,
                    )
                )
                if not faces:
                    continue

                face = max(
                    faces,
                    key=lambda item: (item["bbox"][2] - item["bbox"][0])
                    * (item["bbox"][3] - item["bbox"][1]),
                )
                area = (face["bbox"][2] - face["bbox"][0]) * (
                    face["bbox"][3] - face["bbox"][1]
                )
                if area > best_area:
                    best_face = face
                    best_image = candidate_image
                    best_area = area
                    logger.info(
                        (
                            "InstantID face detected with det_size=%s "
                            "rotation=%s candidate_size=%s"
                        ),
                        det_size,
                        rotation,
                        candidate_image.size,
                    )

            if best_face is not None:
                break

        if best_face is None or best_image is None:
            raise ValueError(
                "No face detected in the uploaded image for InstantID. "
                f"Input image size was {original_size[0]}x{original_size[1]}. "
                "Use a photo with one clear frontal face, "
                "larger face area, and less blur."
            )

        return best_face, best_image

    def _resolve_target_size(
        self,
        image: Image.Image,
        width: int | None,
        height: int | None,
        preserve_source_aspect: bool,
    ) -> tuple[int, int]:
        if width is not None and height is not None and not preserve_source_aspect:
            return width, height

        source_width, source_height = image.size
        if width is None and height is None:
            target_long_side = 1024
        else:
            target_long_side = max(width or 0, height or 0, 1024)

        if preserve_source_aspect:
            if source_width >= source_height:
                scale = target_long_side / source_width
            else:
                scale = target_long_side / source_height
            target_width = self._round_to_multiple(round(source_width * scale))
            target_height = self._round_to_multiple(round(source_height * scale))
            return target_width, target_height

        return width or 1024, height or 1024

    def _build_canny_image(
        self,
        image: Image.Image,
        width: int,
        height: int,
    ) -> Image.Image:
        resized = image.resize((width, height), Image.Resampling.LANCZOS).convert("RGB")
        image_array = self._np.array(resized)
        gray = self._cv2.cvtColor(image_array, self._cv2.COLOR_RGB2GRAY)
        edges = self._cv2.Canny(gray, 100, 200)
        edge_image = self._np.stack([edges] * 3, axis=2)
        return Image.fromarray(edge_image)

    def generate_identity_image(
        self,
        prompt: str,
        reference_image: Image.Image,
        negative_prompt: str = "",
        sampler: str = "dpmpp_sde_karras",
        steps: int = 30,
        guidance_scale: float = 5.0,
        width: int | None = None,
        height: int | None = None,
        seed: int | None = None,
        identity_strength: float = 0.8,
        conditioning_scale: float = 0.8,
        structure_strength: float = 0.35,
        preserve_source_aspect: bool = True,
    ):
        self.load_model()

        assert self._pipeline is not None
        assert self._draw_kps is not None
        self._pipeline.scheduler = build_scheduler(
            sampler=sampler,
            scheduler_config=self._pipeline.scheduler.config,
        )

        face_info, prepared_reference_image = self._extract_face_features(
            reference_image
        )
        target_width, target_height = self._resolve_target_size(
            prepared_reference_image,
            width=width,
            height=height,
            preserve_source_aspect=preserve_source_aspect,
        )
        face_emb = face_info["embedding"]
        face_kps = self._draw_kps(
            prepared_reference_image,
            face_info["kps"],
        ).resize((target_width, target_height), Image.Resampling.LANCZOS)
        canny_image = self._build_canny_image(
            prepared_reference_image,
            width=target_width,
            height=target_height,
        )

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        result = self._pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_embeds=face_emb,
            image=[face_kps, canny_image],
            controlnet_conditioning_scale=[
                conditioning_scale,
                structure_strength,
            ],
            # Let diffusers expand scalar guidance values for all attached
            # controlnets. Passing explicit lists here is brittle across
            # versions and caused length-mismatch validation errors.
            control_guidance_start=0.0,
            control_guidance_end=1.0,
            ip_adapter_scale=identity_strength,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=target_width,
            height=target_height,
            generator=generator,
            output_type="pil",
        )

        return result.images[0]


instantid_service = InstantIDService()
