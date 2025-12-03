from fastapi import FastAPI
from app.api.routes import router as api_router

def create_app() -> FastAPI:
    app = FastAPI(
        title="Financial News Intelligence API",
        description="Multi-agent AI system for processing and querying financial news with sentiment analysis",
        version="1.0.0"
    )
    
    app.include_router(api_router)
    return app

app = create_app()