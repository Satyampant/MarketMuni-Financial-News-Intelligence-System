from pathlib import Path

# Base directories
CORE_DIR = Path(__file__).resolve().parent
APP_DIR = CORE_DIR.parent
ROOT_DIR = APP_DIR.parent
RESOURCES_DIR = APP_DIR / "resources"

class Paths:
    """Centralized path configuration for resources"""
    COMPANY_ALIASES = RESOURCES_DIR / "company_aliases.json"
    SECTOR_TICKERS = RESOURCES_DIR / "sector_tickers.json"
    REGULATORS = RESOURCES_DIR / "regulators.json"
    REGULATOR_IMPACT = RESOURCES_DIR / "regulator_sector_impact.json"
    SUPPLY_CHAIN_GRAPH = RESOURCES_DIR / "supply_chain_graph.json"
    
    # Persistence
    CHROMA_DB = ROOT_DIR / "data" / "chroma_db"