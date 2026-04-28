from __future__ import annotations

import asyncio
import base64
import copy
import logging
import random
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote

import aiohttp

from generation_service.core.config import resolve_service_path, settings
from schemas.generation import GenerateImageRequest, GenerateImageResponse

logger = logging.getLogger(__name__)

SUPPORTED_LORAS: dict[str, str] = {
    "NiceGirls UltraReal": "nicegirls_Zimage.safetensors",
    "TANNED TO PALE SKIN": "zimage_paletannedskin_000000100.safetensors",
    "Breast Slider": "Z-Breast-Slider.safetensors",
    "Realistic Snapshot": "RealisticSnapshot-Zimage-Turbov5.safetensors",
    "begotten": "begott3n_zImage_base.safetensors",
}


class ImageGenerationError(RuntimeError):
    """Base error for the ComfyUI/Z-Image generation flow."""


class ImageGenerationConfigurationError(ImageGenerationError):
    """Raised when ComfyUI or model files are not available on disk."""


class ImageGenerationTemporaryError(ImageGenerationError):
    """Raised for transient problems such as OOM or queue overload."""


class ImageGenerationTimeoutError(ImageGenerationTemporaryError):
    """Raised when a ComfyUI run exceeds the configured timeout."""


class ImageGenerationQueueFullError(ImageGenerationTemporaryError):
    """Raised when the async image queue is full."""


@dataclass(slots=True)
class _QueuedGeneration:
    prompt_id: str
    request: GenerateImageRequest
    future: asyncio.Future[GenerateImageResponse]


class ComfyUIProcessManager:
    def __init__(self) -> None:
        self._process: asyncio.subprocess.Process | None = None
        self._stdout_task: asyncio.Task[None] | None = None
        self._startup_lock = asyncio.Lock()

    @property
    def api_base(self) -> str:
        return f"http://{settings.COMFYUI_HOST}:{settings.COMFYUI_PORT}"

    @property
    def root(self) -> Path:
        configured_root: Path = resolve_service_path(settings.COMFYUI_ROOT)
        if (configured_root / "main.py").exists():
            return configured_root

        nested_root: Path = configured_root / "ComfyUI"
        if (nested_root / "main.py").exists():
            return nested_root

        return configured_root

    @property
    def output_dir(self) -> Path:
        return self.root / "output"

    def ensure_layout(self) -> None:
        root = self.root
        root.mkdir(parents=True, exist_ok=True)
        (root / "input").mkdir(parents=True, exist_ok=True)
        (root / "output").mkdir(parents=True, exist_ok=True)
        (root / "models" / "diffusion_models").mkdir(parents=True, exist_ok=True)
        (root / "models" / "text_encoders").mkdir(parents=True, exist_ok=True)
        (root / "models" / "vae").mkdir(parents=True, exist_ok=True)
        (root / "models" / "loras").mkdir(parents=True, exist_ok=True)

    def validate_required_assets(self) -> None:
        required_files = {
            "ComfyUI entrypoint": self.root / "main.py",
            "Z-Image GGUF model": self.root
            / "models"
            / "diffusion_models"
            / settings.ZIMAGE_BASE_MODEL_FILE,
            "Z-Image text encoder": self.root
            / "models"
            / "text_encoders"
            / settings.ZIMAGE_TEXT_ENCODER_FILE,
            "Z-Image VAE": self.root / "models" / "vae" / settings.ZIMAGE_VAE_FILE,
        }
        missing = [
            f"{label}: {path}"
            for label, path in required_files.items()
            if not path.exists()
        ]
        if missing:
            raise ImageGenerationConfigurationError(
                "ComfyUI/Z-Image environment is not ready. Missing files:\n"
                + "\n".join(missing)
            )

    async def start(self, session: aiohttp.ClientSession) -> None:
        if self._process is not None and self._process.returncode is None:
            return

        async with self._startup_lock:
            if self._process is not None and self._process.returncode is None:
                return

            self.ensure_layout()
            self.validate_required_assets()

            command = [
                sys.executable,
                "main.py",
                "--lowvram",
                "--listen",
                settings.COMFYUI_LISTEN_HOST,
                "--port",
                str(settings.COMFYUI_PORT),
            ]

            logger.info(
                "Starting ComfyUI at %s on port %s with --lowvram",
                self.root,
                settings.COMFYUI_PORT,
            )
            try:
                self._process = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=str(self.root),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                self._stdout_task = asyncio.create_task(self._stream_stdout())
                await self._wait_until_ready(session)
            except Exception:
                await self.stop()
                raise
            logger.info("ComfyUI is ready")

    async def stop(self) -> None:
        process = self._process
        self._process = None

        if process is None:
            return

        if process.returncode is None:
            logger.info("Stopping ComfyUI process")
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=10)
            except asyncio.TimeoutError:
                logger.warning("ComfyUI did not terminate gracefully, killing it")
                process.kill()
                await process.wait()

        if self._stdout_task is not None:
            self._stdout_task.cancel()
            try:
                await self._stdout_task
            except asyncio.CancelledError:
                pass
            self._stdout_task = None

    async def _stream_stdout(self) -> None:
        if self._process is None or self._process.stdout is None:
            return

        try:
            while True:
                line = await self._process.stdout.readline()
                if not line:
                    break
                decoded = line.decode("utf-8", errors="replace").rstrip()
                logger.info("ComfyUI | %s", decoded)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Failed to read ComfyUI output")

    async def _wait_until_ready(self, session: aiohttp.ClientSession) -> None:
        deadline = time.monotonic() + settings.COMFYUI_STARTUP_TIMEOUT_SECONDS
        last_error: Exception | None = None

        while time.monotonic() < deadline:
            try:
                for endpoint in ("/health", "/system_stats"):
                    async with session.get(
                        f"{self.api_base}{endpoint}",
                        timeout=aiohttp.ClientTimeout(total=5),
                    ) as response:
                        if response.status == 200:
                            return
            except Exception as exc:
                last_error = exc

            await asyncio.sleep(1)

        raise ImageGenerationConfigurationError(
            "ComfyUI did not become healthy within the startup timeout."
        ) from last_error


class ZImageWorkflowFactory:
    """Builds and caches static workflow fragments for specific LoRA chains."""

    def __init__(self, comfy_root: Path) -> None:
        self._comfy_root = comfy_root
        self._template_cache: dict[tuple[tuple[str, float], ...], dict[str, Any]] = {}

    def build(
        self,
        prompt_id: str,
        request: GenerateImageRequest,
        seed: int,
    ) -> dict[str, Any]:
        signature = tuple((item.name, round(item.weight, 3)) for item in request.loras)
        template = self._template_cache.get(signature)
        if template is None:
            template = self._build_template(signature)
            self._template_cache[signature] = template

        workflow = copy.deepcopy(template)
        workflow["100"]["inputs"]["text"] = request.prompt
        if request.negative_prompt:
            workflow["101"] = {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": request.negative_prompt,
                    "clip": workflow["100"]["inputs"]["clip"],
                },
            }
        workflow["102"]["inputs"]["width"] = request.width
        workflow["102"]["inputs"]["height"] = request.height
        workflow["104"]["inputs"]["seed"] = seed
        workflow["104"]["inputs"]["steps"] = request.steps
        workflow["106"]["inputs"]["filename_prefix"] = f"zimage_{prompt_id}"
        return workflow

    def validate_requested_loras(self, request: GenerateImageRequest) -> None:
        lora_dir = self._comfy_root / "models" / "loras"
        missing_loras = []
        for item in request.loras:
            filename = self._lora_filename(item.name)
            if not (lora_dir / filename).exists():
                missing_loras.append(f"{item.name}: {lora_dir / filename}")

        if missing_loras:
            raise ImageGenerationConfigurationError(
                "Requested LoRA files are missing:\n" + "\n".join(missing_loras)
            )

    def _build_template(
        self,
        signature: tuple[tuple[str, float], ...],
    ) -> dict[str, Any]:
        workflow: dict[str, Any] = {
            "1": {
                "class_type": "UnetLoaderGGUF",
                "inputs": {
                    "unet_name": settings.ZIMAGE_BASE_MODEL_FILE,
                },
            },
            "2": {
                "class_type": "CLIPLoader",
                "inputs": {
                    "clip_name": settings.ZIMAGE_TEXT_ENCODER_FILE,
                    "type": "lumina2",
                    "device": "default",
                },
            },
            "3": {
                "class_type": "VAELoader",
                "inputs": {
                    "vae_name": settings.ZIMAGE_VAE_FILE,
                },
            },
        }

        current_model: list[Any] = ["1", 0]
        current_clip: list[Any] = ["2", 0]

        for index, (name, weight) in enumerate(signature, start=10):
            node_id = str(index)
            workflow[node_id] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "model": current_model,
                    "clip": current_clip,
                    "lora_name": self._lora_filename(name),
                    "strength_model": weight,
                    "strength_clip": weight,
                },
            }
            current_model = [node_id, 0]
            current_clip = [node_id, 1]

        workflow["100"] = {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "",
                "clip": current_clip,
            },
        }
        workflow["101"] = {
            "class_type": "ConditioningZeroOut",
            "inputs": {
                "conditioning": ["100", 0],
            },
        }
        workflow["102"] = {
            "class_type": "EmptySD3LatentImage",
            "inputs": {
                "width": 768,
                "height": 768,
                "batch_size": 1,
            },
        }
        workflow["103"] = {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {
                "model": current_model,
                "shift": 3.0,
            },
        }
        workflow["104"] = {
            "class_type": "KSampler",
            "inputs": {
                "seed": 0,
                "steps": 10,
                "cfg": 1.0,
                "sampler_name": "res_multistep",
                "scheduler": "simple",
                "denoise": 1.0,
                "model": ["103", 0],
                "positive": ["100", 0],
                "negative": ["101", 0],
                "latent_image": ["102", 0],
            },
        }
        workflow["105"] = {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["104", 0],
                "vae": ["3", 0],
            },
        }
        workflow["106"] = {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "zimage",
                "images": ["105", 0],
            },
        }
        return workflow

    def _lora_filename(self, name: str) -> str:
        try:
            return SUPPORTED_LORAS[name]
        except KeyError as exc:
            supported = ", ".join(SUPPORTED_LORAS)
            raise ImageGenerationConfigurationError(
                f"Unsupported LoRA '{name}'. Supported values: {supported}"
            ) from exc


class ImageGenerationService:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[_QueuedGeneration] = asyncio.Queue(
            maxsize=settings.ZIMAGE_QUEUE_SIZE
        )
        self._worker_task: asyncio.Task[None] | None = None
        self._session: aiohttp.ClientSession | None = None
        self._comfy = ComfyUIProcessManager()
        self._workflow_factory = ZImageWorkflowFactory(self._comfy.root)

    async def startup(self) -> None:
        if not settings.ZIMAGE_ENABLED:
            logger.info("Z-Image generation is disabled by config")
            return

        if self._session is None:
            generation_timeout_seconds = settings.COMFYUI_GENERATION_TIMEOUT_SECONDS
            timeout_total: float | None = None
            if generation_timeout_seconds > 0:
                timeout_total = float(generation_timeout_seconds)
            timeout = aiohttp.ClientTimeout(total=timeout_total)
            self._session = aiohttp.ClientSession(timeout=timeout)

        await self._comfy.start(self._session)

        if self._worker_task is None:
            self._worker_task = asyncio.create_task(
                self._worker_loop(),
                name="zimage-worker",
            )
            logger.info("Async Z-Image worker started")

    async def shutdown(self) -> None:
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

        while True:
            try:
                pending_job = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if not pending_job.future.done():
                pending_job.future.set_exception(
                    ImageGenerationTemporaryError(
                        "Image generation service is shutting down."
                    )
                )
            self._queue.task_done()

        if self._session is not None:
            await self._session.close()
            self._session = None

        await self._comfy.stop()

    async def generate(self, request: GenerateImageRequest) -> GenerateImageResponse:
        if not settings.ZIMAGE_ENABLED:
            raise ImageGenerationConfigurationError("Z-Image generation is disabled.")

        if self._worker_task is None:
            raise ImageGenerationConfigurationError(
                "Z-Image worker is not running. Service startup is incomplete."
            )

        loop = asyncio.get_running_loop()
        prompt_id = str(uuid.uuid4())
        future: asyncio.Future[GenerateImageResponse] = loop.create_future()

        try:
            self._queue.put_nowait(
                _QueuedGeneration(
                    prompt_id=prompt_id,
                    request=request,
                    future=future,
                )
            )
        except asyncio.QueueFull as exc:
            raise ImageGenerationQueueFullError(
                "Image generation queue is full. Try again later."
            ) from exc
        logger.info(
            "Enqueued Z-Image request %s (%sx%s, %s LoRAs)",
            prompt_id,
            request.width,
            request.height,
            len(request.loras),
        )
        return await future

    async def _worker_loop(self) -> None:
        while True:
            job = await self._queue.get()
            try:
                result = await self._run_generation(job.prompt_id, job.request)
                if not job.future.done():
                    job.future.set_result(result)
            except Exception as exc:
                logger.exception("Z-Image request %s failed", job.prompt_id)
                if not job.future.done():
                    job.future.set_exception(exc)
            finally:
                self._queue.task_done()

    async def _run_generation(
        self,
        prompt_id: str,
        request: GenerateImageRequest,
    ) -> GenerateImageResponse:
        assert self._session is not None

        try:
            self._workflow_factory.validate_requested_loras(request)

            seed = (
                request.seed
                if request.seed is not None
                else random.randint(0, 2_147_483_647)
            )
            started_at = time.perf_counter()
            workflow = self._workflow_factory.build(
                prompt_id=prompt_id,
                request=request,
                seed=seed,
            )

            logger.info(
                "Submitting workflow for %s (seed=%s, steps=%s, loras=%s)",
                prompt_id,
                seed,
                request.steps,
                [item.model_dump() for item in request.loras],
            )

            comfy_prompt_id = await self._submit_workflow(prompt_id, workflow)

            timeout_seconds = settings.COMFYUI_GENERATION_TIMEOUT_SECONDS
            history: dict[str, Any] | None
            if timeout_seconds <= 0:
                history = await self._wait_for_history(comfy_prompt_id)
            else:
                try:
                    history = await asyncio.wait_for(
                        self._wait_for_history(comfy_prompt_id),
                        timeout=(
                            timeout_seconds
                            + settings.COMFYUI_POLL_INTERVAL_SECONDS
                            + 2
                        ),
                    )
                except asyncio.TimeoutError as exc:
                    history = await self._try_get_completed_history(comfy_prompt_id)
                    if history is None:
                        await self._interrupt_current_run()
                        message = (
                            f"Generation exceeded {timeout_seconds} seconds "
                            "and was cancelled."
                        )
                        raise ImageGenerationTimeoutError(
                            message
                        ) from exc

            if history is None:
                raise ImageGenerationError("ComfyUI history is unavailable.")

            error_message = self._extract_history_error(history)
            if error_message is not None:
                raise self._map_runtime_error(error_message)

            image_meta = self._extract_first_image(history)
            image_bytes, _ = await self._read_generated_image(image_meta)
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)

            image_base64: str | None = None
            image_url: str | None = None
            image_relative_path = self._output_relative_path(image_meta).as_posix()

            if request.response_format in {"base64", "both"}:
                image_base64 = base64.b64encode(image_bytes).decode("ascii")

            if request.response_format in {"url", "both"}:
                image_url = self._build_image_url(image_meta)

            logger.info("Z-Image request %s completed in %sms", prompt_id, elapsed_ms)
            return GenerateImageResponse(
                prompt_id=prompt_id,
                image_path=image_relative_path,
                image_url=image_url,
                image_base64=image_base64,
                seed=seed,
                elapsed_ms=elapsed_ms,
                width=request.width,
                height=request.height,
                loras=request.loras,
            )
        except aiohttp.ClientError as exc:
            raise ImageGenerationTemporaryError(
                "Failed to communicate with ComfyUI."
            ) from exc

    async def _submit_workflow(
        self,
        client_prompt_id: str,
        workflow: dict[str, Any],
    ) -> str:
        assert self._session is not None
        payload = {
            "prompt": workflow,
            "client_id": client_prompt_id,
        }
        async with self._session.post(
            f"{self._comfy.api_base}/prompt",
            json=payload,
        ) as response:
            body = await response.text()
            if response.status >= 400:
                raise self._map_runtime_error(
                    f"ComfyUI rejected workflow with status {response.status}: {body}"
                )
            data = await response.json(content_type=None)
        if not isinstance(data, dict):
            raise ImageGenerationError("ComfyUI returned an invalid prompt payload.")
        prompt_id = data.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ImageGenerationError("ComfyUI returned an invalid prompt_id.")
        return prompt_id

    async def _wait_for_history(self, comfy_prompt_id: str) -> dict[str, Any]:
        while True:
            history_payload = await self._fetch_history_payload(comfy_prompt_id)
            history = history_payload.get(comfy_prompt_id)
            if isinstance(history, dict):
                return history

            await asyncio.sleep(settings.COMFYUI_POLL_INTERVAL_SECONDS)

    async def _try_get_completed_history(
        self,
        comfy_prompt_id: str,
    ) -> dict[str, Any] | None:
        """One final history check before interrupting a near-complete run."""
        try:
            history_payload = await self._fetch_history_payload(comfy_prompt_id)
        except Exception:
            logger.exception(
                "Failed final history check for timed out prompt %s",
                comfy_prompt_id,
            )
            return None
        return history_payload.get(comfy_prompt_id)

    async def _fetch_history_payload(self, comfy_prompt_id: str) -> dict[str, Any]:
        assert self._session is not None

        async with self._session.get(
            f"{self._comfy.api_base}/history/{comfy_prompt_id}"
        ) as response:
            body = await response.text()
            if response.status >= 400:
                raise ImageGenerationError(
                    f"Failed to fetch ComfyUI history: {response.status} {body}"
                )
            payload = await response.json(content_type=None)
            if not isinstance(payload, dict):
                raise ImageGenerationError(
                    "ComfyUI returned an invalid history payload."
                )
            return cast(dict[str, Any], payload)

    async def _interrupt_current_run(self) -> None:
        if self._session is None:
            return

        try:
            async with self._session.post(
                f"{self._comfy.api_base}/interrupt"
            ) as response:
                await response.read()
                logger.warning(
                    "Sent interrupt request to ComfyUI (status=%s)",
                    response.status,
                )
        except Exception:
            logger.exception("Failed to interrupt timed out ComfyUI run")

    def _extract_history_error(self, history: dict[str, Any]) -> str | None:
        status = history.get("status")
        if isinstance(status, dict):
            status_text = status.get("status_str")
            if isinstance(status_text, str) and status_text.lower() == "error":
                messages = status.get("messages")
                if isinstance(messages, list):
                    return " | ".join(str(item) for item in messages)
                return "ComfyUI returned an error status."

        for key in ("error", "execution_error"):
            error_value = history.get(key)
            if error_value:
                return str(error_value)

        return None

    def _extract_first_image(self, history: dict[str, Any]) -> dict[str, Any]:
        outputs = history.get("outputs")
        if not isinstance(outputs, dict):
            raise ImageGenerationError("ComfyUI history does not contain outputs.")

        for node_output in outputs.values():
            if not isinstance(node_output, dict):
                continue
            images = node_output.get("images")
            if isinstance(images, list) and images:
                image_meta = images[0]
                if isinstance(image_meta, dict):
                    return image_meta

        raise ImageGenerationError("ComfyUI finished without producing an image.")

    async def _read_generated_image(
        self,
        image_meta: dict[str, Any],
    ) -> tuple[bytes, Path]:
        base_dir = self._comfy.output_dir.resolve()
        image_path = (base_dir / self._output_relative_path(image_meta)).resolve()
        if not image_path.is_relative_to(base_dir):
            raise ImageGenerationError(
                "ComfyUI returned an image path outside the output directory."
            )

        if image_path.exists():
            image_bytes = await asyncio.to_thread(image_path.read_bytes)
            return image_bytes, image_path

        image_bytes = await self._download_image_from_api(image_meta)
        await asyncio.to_thread(image_path.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(image_path.write_bytes, image_bytes)
        return image_bytes, image_path

    async def _download_image_from_api(self, image_meta: dict[str, Any]) -> bytes:
        assert self._session is not None

        filename = quote(str(image_meta["filename"]))
        subfolder = quote(str(image_meta.get("subfolder", "")))
        file_type = quote(str(image_meta.get("type", "output")))
        url = (
            f"{self._comfy.api_base}/view?filename={filename}"
            f"&subfolder={subfolder}&type={file_type}"
        )
        async with self._session.get(url) as response:
            if response.status >= 400:
                body = await response.text()
                raise ImageGenerationError(
                    "Failed to download generated image from ComfyUI: "
                    f"{response.status} {body}"
                )
            return await response.read()

    def _build_image_url(self, image_meta: dict[str, Any]) -> str | None:
        if not settings.ZIMAGE_OUTPUT_URL_PREFIX:
            return None

        encoded_path = quote(self._output_relative_path(image_meta).as_posix())
        return f"{settings.ZIMAGE_OUTPUT_URL_PREFIX.rstrip('/')}/{encoded_path}"

    def _output_relative_path(self, image_meta: dict[str, Any]) -> Path:
        file_type = str(image_meta.get("type", "output"))
        if file_type != "output":
            raise ImageGenerationError(
                f"Unsupported ComfyUI output type: {file_type!r}."
            )

        filename = Path(str(image_meta.get("filename", ""))).name
        if not filename:
            raise ImageGenerationError(
                "ComfyUI image metadata does not contain filename."
            )

        raw_subfolder = str(image_meta.get("subfolder", "")).strip("/\\")
        subfolder = Path()
        if raw_subfolder:
            subfolder = Path(raw_subfolder.replace("\\", "/"))
            if subfolder.is_absolute() or any(
                part in {".", ".."} for part in subfolder.parts
            ):
                raise ImageGenerationError(
                    "ComfyUI returned an invalid image subfolder."
                )

        return subfolder / filename

    def _map_runtime_error(self, message: str) -> ImageGenerationError:
        lowered = message.lower()
        if any(
            marker in lowered
            for marker in ("out of memory", "cuda error", "cudnn_status_not_supported")
        ):
            return ImageGenerationTemporaryError(
                "GPU memory is insufficient for this request. "
                "Try lowering resolution or reducing the number of LoRAs."
            )

        return ImageGenerationError(message)


image_generation_service = ImageGenerationService()
