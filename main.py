from fastapi import FastAPI

from api.health import router as health_router
from api.dataset import router as dataset_router


app = FastAPI(title="NeuroForge")

app.include_router(health_router)
app.include_router(dataset_router)