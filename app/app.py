import time

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel
from dataclasses import dataclass
from prometheus_client import make_asgi_app
from app.metrics import REQUEST_TIME, REQUEST_COUNT
from ml.model import load_models

get_errors = None


@dataclass
class EmbeddRequest(BaseModel):
    wind_speed: float
    temp: float
    humidity: float
    temp1: float
    temp2: float
    temp3: float
    temp4: float
    temp5: float
    temp6: float
    temp7: float
    temp8: float
    temp9: float
    temp10: float
    temp11: float
    temp12: float
    temp13: float
    temp14: float
    strain1: float
    strain2: float
    strain3: float
    strain4: float
    strain5: float
    strain6: float
    strain7: float
    strain8: float
    strain9: float
    strain10: float
    strain11: float
    strain12: float
    strain13: float
    strain14: float


@dataclass
class EmbeddResponse(BaseModel):
    delta_a: float
    delta_e: float
    roll: float

    def __init__(self, delta_a: float, delta_e: float, roll: float) -> None:
        super().__init__(delta_a=delta_a, delta_e=delta_e, roll=roll)


metrics_app = make_asgi_app()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    global get_errors
    get_errors = load_models()
    yield


def create_app():
    app = FastAPI(lifespan=lifespan)

    @app.get("/state", status_code=200)
    async def state():
        return

    app.mount("/metrics", metrics_app)

    @app.post("/api/v1/errors")
    async def errors(req: EmbeddRequest):
        start_time = time.time()

        delta_a, delta_e, roll = get_errors(
            req.wind_speed,
            req.wind_speed,
            req.temp,
            req.temp1,
            req.temp2,
            req.temp3,
            req.temp4,
            req.temp5,
            req.temp6,
            req.temp7,
            req.temp8,
            req.temp9,
            req.temp10,
            req.temp11,
            req.temp12,
            req.temp13,
            req.temp14,
            req.strain1,
            req.strain2,
            req.strain3,
            req.strain4,
            req.strain5,
            req.strain6,
            req.strain7,
            req.strain8,
            req.strain9,
            req.strain10,
            req.strain11,
            req.strain12,
            req.strain13,
            req.strain14,
        )

        resp_body = ORJSONResponse(EmbeddResponse(delta_a=delta_a, delta_e=delta_e, roll=roll))

        REQUEST_COUNT.labels('200').inc()
        REQUEST_TIME.labels('200').observe(time.time() - start_time)

        return resp_body

    return app
