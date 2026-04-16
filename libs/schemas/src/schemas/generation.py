from typing import Literal

from pydantic import BaseModel, Field


GenerationStatus = Literal["pending", "processing", "done", "error"]
GenerationSampler = Literal[
    "dpmpp_sde_karras",
    "dpmpp_2m_karras",
    "euler_a",
    "euler",
    "ddim",
]
GenerationModelKey = Literal[
    "sd15_realisticvision",
    "sdxl_instantid",
]


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=5000)
    negative_prompt: str = ""
    model_key: GenerationModelKey = "sd15_realisticvision"
    sampler: GenerationSampler = "dpmpp_sde_karras"
    steps: int = Field(30, ge=1, le=150)
    guidance_scale: float = Field(7.5, ge=1.0, le=30.0)
    width: int = Field(512, ge=64, le=2048, multiple_of=8)
    height: int = Field(512, ge=64, le=2048, multiple_of=8)
    seed: int | None = None


class GenerateAcceptedResponse(BaseModel):
    task_id: str
    status: GenerationStatus = "pending"


class GenerationResultResponse(BaseModel):
    task_id: str
    status: GenerationStatus
    image_path: str | None = None
    error: str | None = None


class GenerationModelInfo(BaseModel):
    key: GenerationModelKey
    label: str
    task: Literal["txt2img", "identity_edit"]
    description: str
    default_width: int
    default_height: int
