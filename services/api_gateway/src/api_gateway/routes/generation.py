from typing import Annotated, Any

from api_gateway.clients.generation_client import GenerationClient
from api_gateway.dependencies import get_generation_client
from fastapi import APIRouter, Depends, File, Form, UploadFile
from schemas.generation import (
    GenerateAcceptedResponse,
    GenerateImageRequest,
    GenerateImageResponse,
    GenerateRequest,
    GenerationModelInfo,
    GenerationModelKey,
    GenerationSampler,
    GenerationResultResponse,
)

router = APIRouter(prefix="/generation", tags=["generation"])
image_router = APIRouter(prefix="/generate", tags=["generation"])


@router.post("/", response_model=GenerateAcceptedResponse)
async def generate(
    request: GenerateRequest,
    client: GenerationClient = Depends(get_generation_client),
) -> GenerateAcceptedResponse:
    return await client.generate(request)


@router.get("/models", response_model=list[GenerationModelInfo])
async def list_generation_models(
    client: GenerationClient = Depends(get_generation_client),
) -> Any:
    return await client.list_models()


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
    client: GenerationClient = Depends(get_generation_client),
) -> GenerateAcceptedResponse:
    return await client.edit_image(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        model_key=model_key,
        sampler=sampler,
        steps=steps,
        guidance_scale=guidance_scale,
        strength=strength,
        identity_strength=identity_strength,
        conditioning_scale=conditioning_scale,
        structure_strength=structure_strength,
        preserve_source_aspect=preserve_source_aspect,
        width=width,
        height=height,
        seed=seed,
    )


@router.get("/tasks/{task_id}", response_model=GenerationResultResponse)
async def get_generation_result(
    task_id: str,
    client: GenerationClient = Depends(get_generation_client),
) -> GenerationResultResponse:
    return await client.get_result(task_id)


@image_router.post("/image", response_model=GenerateImageResponse)
async def generate_image(
    request: GenerateImageRequest,
    client: GenerationClient = Depends(get_generation_client),
) -> GenerateImageResponse:
    return await client.generate_image(request)
