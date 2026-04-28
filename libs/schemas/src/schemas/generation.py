from typing import Literal

from pydantic import BaseModel, Field, field_validator


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
ImageGenerationEncoding = Literal["base64", "url", "both"]
SupportedImageLoraName = Literal[
    "NiceGirls UltraReal",
    "TANNED TO PALE SKIN",
    "Breast Slider",
    "Realistic Snapshot",
    "begotten",
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


class ImageLoraConfig(BaseModel):
    name: SupportedImageLoraName
    weight: float = Field(..., ge=0.0, le=1.0)


class GenerateImageRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=5000)
    negative_prompt: str = ""
    loras: list[ImageLoraConfig] = Field(default_factory=list, max_length=5)
    seed: int | None = None
    steps: int = Field(10, ge=1, le=30)
    width: int = Field(768, ge=256, le=1024, multiple_of=8)
    height: int = Field(768, ge=256, le=1024, multiple_of=8)
    response_format: ImageGenerationEncoding = "both"

    @field_validator("loras")
    @classmethod
    def validate_loras_unique(
        cls,
        value: list[ImageLoraConfig],
    ) -> list[ImageLoraConfig]:
        names = [item.name for item in value]
        if len(names) != len(set(names)):
            raise ValueError("Each LoRA can only be specified once per request.")
        return value


class GenerateImageResponse(BaseModel):
    prompt_id: str
    image_path: str
    image_url: str | None = None
    image_base64: str | None = None
    seed: int
    elapsed_ms: int
    width: int
    height: int
    loras: list[ImageLoraConfig]
