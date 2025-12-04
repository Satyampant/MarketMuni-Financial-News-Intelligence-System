from fastapi import FastAPI
from app.api.routes import router as api_router
from app.core.config_loader import get_config

def create_app() -> FastAPI:
    config = get_config()

    app = FastAPI(
        title=config.api.title,
        description=config.api.description,
        version=config.api.version
    )
    
    app.include_router(api_router)
    return app

app = create_app()