import uvicorn
import os
import sys

if __name__ == "__main__":
    # Ensure data directory exists
    os.makedirs("data/chroma_db", exist_ok=True)
    
    print("Starting Financial News Intelligence API...")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)