from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSDEScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
)


def build_scheduler(sampler: str, scheduler_config):
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
