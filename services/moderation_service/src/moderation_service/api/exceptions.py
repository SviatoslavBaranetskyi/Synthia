from fastapi import Request
from fastapi.responses import JSONResponse


class ServiceError(Exception):
    pass


async def service_exception_handler(request: Request, exc: ServiceError):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )
