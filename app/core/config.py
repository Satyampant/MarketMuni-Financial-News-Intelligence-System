from pathlib import Path

# Base directories
CORE_DIR = Path(__file__).resolve().parent
APP_DIR = CORE_DIR.parent
ROOT_DIR = APP_DIR.parent
RESOURCES_DIR = APP_DIR / "resources"

class Paths:
    """Centralized path configuration for resources"""
    # Persistence
    CHROMA_DB = ROOT_DIR / "data" / "chroma_db"