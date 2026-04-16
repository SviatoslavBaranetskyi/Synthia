from schemas.generation import GenerationModelInfo


AVAILABLE_MODELS = [
    GenerationModelInfo(
        key="sd15_realisticvision",
        label="Realistic Vision V6.0 (SD 1.5)",
        task="txt2img",
        description=(
            "Current local SD 1.5 checkpoint. "
            "Best for existing text-to-image and basic img2img."
        ),
        default_width=512,
        default_height=512,
    ),
    GenerationModelInfo(
        key="sdxl_instantid",
        label="SDXL + InstantID",
        task="identity_edit",
        description="Identity-preserving SDXL editing using a reference face image.",
        default_width=1024,
        default_height=1024,
    ),
]
