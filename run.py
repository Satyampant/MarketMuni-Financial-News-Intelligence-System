import uvicorn
import os
import sys
from app.core.config_loader import get_config

if __name__ == "__main__":
    config = get_config()

    os.makedirs("data/chroma_db", exist_ok=True)
    
    print("Starting Financial News Intelligence API...")
    print(f"Configuration: {config.sentiment_analysis.method} sentiment analysis")
    
    uvicorn.run(
        "app.main:app", 
        host=config.api.host, 
        port=config.api.port, 
        reload=config.api.reload
    )