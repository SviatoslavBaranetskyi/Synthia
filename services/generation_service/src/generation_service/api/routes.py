from typing import Annotated, cast

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from generation_service.api.dependencies import (
    get_image_generation_service,
    get_task_queue,
)
from generation_service.services.model_catalog import AVAILABLE_MODELS
from generation_service.services.image_generation import (
    ImageGenerationConfigurationError,
    ImageGenerationError,
    ImageGenerationQueueFullError,
    ImageGenerationService,
    ImageGenerationTemporaryError,
)
from generation_service.services.queue import GenerationTaskQueue, QueueFullError
from generation_service.utils.image import (
    InvalidUploadedImageError,
    save_uploaded_image,
)
from schemas.generation import (
    GenerateAcceptedResponse,
    GenerateImageRequest,
    GenerateImageResponse,
    GenerateRequest,
    GenerationModelKey,
    GenerationSampler,
    GenerationModelInfo,
    GenerationResultResponse,
)

router = APIRouter(prefix="/generation", tags=["generation"])
image_router = APIRouter(prefix="/generate", tags=["generation"])


@router.post("/", response_model=GenerateAcceptedResponse)
def generate(
    request: GenerateRequest,
    queue: GenerationTaskQueue = Depends(get_task_queue),
) -> GenerateAcceptedResponse:
    if request.model_key != "sd15_realisticvision":
        raise HTTPException(
            status_code=400,
            detail="Only sd15_realisticvision is currently supported for /generation/",
        )

    payload = request.model_dump()
    payload["mode"] = "txt2img"
    try:
        return queue.submit(payload)
    except QueueFullError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/models", response_model=list[GenerationModelInfo])
def list_generation_models() -> list[GenerationModelInfo]:
    return cast(list[GenerationModelInfo], AVAILABLE_MODELS)


@router.post("/edit", response_model=GenerateAcceptedResponse)
async def edit_image(
    prompt: Annotated[str, Form(min_length=1, max_length=5000)],
    image: UploadFile = File(...),
    negative_prompt: Annotated[str, Form()] = "",
    model_key: Annotated[GenerationModelKey, Form()] = "sdxl_instantid",
    sampler: Annotated[GenerationSampler, Form()] = "dpmpp_sde_karras",
    steps: Annotated[int, Form(ge=1, le=150)] = 30,
    guidance_scale: Annotated[float, Form(ge=1.0, le=30.0)] = 7.5,
    strength: Annotated[float, Form(ge=0.0, le=1.0)] = 0.65,
    identity_strength: Annotated[float, Form(ge=0.0, le=1.5)] = 0.8,
    conditioning_scale: Annotated[float, Form(ge=0.0, le=2.0)] = 0.8,
    structure_strength: Annotated[float, Form(ge=0.0, le=2.0)] = 0.35,
    preserve_source_aspect: Annotated[bool, Form()] = True,
    width: Annotated[int | None, Form(ge=512, le=2048, multiple_of=8)] = None,
    height: Annotated[int | None, Form(ge=512, le=2048, multiple_of=8)] = None,
    seed: Annotated[int | None, Form()] = None,
    queue: GenerationTaskQueue = Depends(get_task_queue),
) -> GenerateAcceptedResponse:
    try:
        source_image_path = save_uploaded_image(image)
    except InvalidUploadedImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    payload = {
        "mode": "identity_edit" if model_key == "sdxl_instantid" else "img2img",
        "model_key": model_key,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "sampler": sampler,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "strength": strength,
        "identity_strength": identity_strength,
        "conditioning_scale": conditioning_scale,
        "structure_strength": structure_strength,
        "preserve_source_aspect": preserve_source_aspect,
        "width": width,
        "height": height,
        "seed": seed,
        "source_image_path": source_image_path,
    }
    try:
        return queue.submit(payload)
    except QueueFullError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/tasks/{task_id}", response_model=GenerationResultResponse)
def get_generation_result(
    task_id: str,
    queue: GenerationTaskQueue = Depends(get_task_queue),
) -> GenerationResultResponse:
    result = queue.get_result(task_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Generation task not found")

    return result


@image_router.post("/image", response_model=GenerateImageResponse)
async def generate_image(
    request: GenerateImageRequest,
    service: ImageGenerationService = Depends(get_image_generation_service),
) -> GenerateImageResponse:
    try:
        return await service.generate(request)
    except ImageGenerationQueueFullError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ImageGenerationTemporaryError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ImageGenerationConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ImageGenerationError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
