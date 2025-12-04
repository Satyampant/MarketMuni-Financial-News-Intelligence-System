"""
Configuration Loader for MarketMuni
Loads and validates configuration from YAML file.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field

# Resolve project root relative to this file
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_FILE = ROOT_DIR / "config.yaml"


@dataclass
class DeduplicationConfig:
    bi_encoder_threshold: float = 0.50
    cross_encoder_threshold: float = 0.70
    bi_encoder_model: str = "all-mpnet-base-v2"
    cross_encoder_model: str = "cross-encoder/stsb-distilroberta-base"


@dataclass
class EntityExtractionConfig:
    spacy_model: str = "en_core_web_sm"
    use_spacy: bool = True
    event_keywords: list = field(default_factory=list)


@dataclass
class HybridWeights:
    finbert_weight: float = 0.7
    rule_weight: float = 0.3


@dataclass
class FinBERTConfig:
    model_name: str = "ProsusAI/finbert"
    device: Optional[str] = None


@dataclass
class RuleBasedConfig:
    spacy_model: str = "en_core_web_sm"
    use_spacy: bool = True


@dataclass
class SentimentAnalysisConfig:
    method: str = "hybrid"
    hybrid_weights: HybridWeights = field(default_factory=HybridWeights)
    finbert: FinBERTConfig = field(default_factory=FinBERTConfig)
    rule_based: RuleBasedConfig = field(default_factory=RuleBasedConfig)
    entity_weights: Dict[str, float] = field(default_factory=dict)
    event_modifiers: Dict[str, float] = field(default_factory=dict)


@dataclass
class VectorStoreConfig:
    collection_name: str = "financial_news"
    persist_directory: str = "data/chroma_db"
    embedding_model: str = "all-mpnet-base-v2"
    distance_metric: str = "cosine"


@dataclass
class MultiQueryConfig:
    max_context_queries: int = 3
    initial_retrieval_multiplier: int = 2


@dataclass
class RerankingWeights:
    strategy_weight: float = 0.5
    semantic_weight: float = 0.5


@dataclass
class SentimentBoostConfig:
    enabled: bool = True
    max_multiplier: float = 1.5


@dataclass
class QueryProcessingConfig:
    default_top_k: int = 10
    min_similarity: float = 0.3
    multi_query: MultiQueryConfig = field(default_factory=MultiQueryConfig)
    reranking_weights: Dict[str, RerankingWeights] = field(default_factory=dict)
    sentiment_boost: SentimentBoostConfig = field(default_factory=SentimentBoostConfig)


@dataclass
class StockImpactConfig:
    confidence_thresholds: Dict[str, float] = field(default_factory=dict)
    fuzzy_match_threshold: float = 0.80


@dataclass
class SupplyChainConfig:
    traversal_depth: int = 1
    min_impact_score: float = 25.0
    weight_decay: float = 0.8


@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    title: str = "Financial News Intelligence API"
    description: str = "Multi-agent AI system for processing and querying financial news"
    version: str = "1.0.0"


@dataclass
class ResourcesConfig:
    company_aliases: str = "company_aliases.json"
    sector_tickers: str = "sector_tickers.json"
    regulators: str = "regulators.json"
    regulator_impact: str = "regulator_sector_impact.json"
    supply_chain_graph: str = "supply_chain_graph.json"


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class PerformanceConfig:
    cache_embeddings: bool = True
    batch_size: int = 32
    num_workers: int = 4


@dataclass
class DevelopmentConfig:
    debug: bool = False
    use_mock_data: bool = False
    mock_data_path: str = "mock_news_data.json"
    enable_profiling: bool = False


@dataclass
class Config:
    """Main configuration class containing all settings."""
    deduplication: DeduplicationConfig = field(default_factory=DeduplicationConfig)
    entity_extraction: EntityExtractionConfig = field(default_factory=EntityExtractionConfig)
    sentiment_analysis: SentimentAnalysisConfig = field(default_factory=SentimentAnalysisConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    query_processing: QueryProcessingConfig = field(default_factory=QueryProcessingConfig)
    stock_impact: StockImpactConfig = field(default_factory=StockImpactConfig)
    supply_chain: SupplyChainConfig = field(default_factory=SupplyChainConfig)
    api: APIConfig = field(default_factory=APIConfig)
    resources: ResourcesConfig = field(default_factory=ResourcesConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    development: DevelopmentConfig = field(default_factory=DevelopmentConfig)


def load_config(config_path: Optional[Path] = None) -> Config:
    """Load configuration from YAML file, falling back to defaults if missing."""
    if config_path is None:
        config_path = CONFIG_FILE
    
    if not config_path.exists():
        print(f"⚠ Warning: Config file not found at {config_path}")
        print("  Using default configuration values")
        return Config()
    
    try:
        with open(config_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        if yaml_data is None:
            print("⚠ Warning: Empty config file, using defaults")
            return Config()
        
        config = Config()
        
        # --- Deduplication ---
        if 'deduplication' in yaml_data:
            dedup = yaml_data['deduplication']
            config.deduplication = DeduplicationConfig(
                bi_encoder_threshold=dedup.get('bi_encoder_threshold', 0.50),
                cross_encoder_threshold=dedup.get('cross_encoder_threshold', 0.70),
                bi_encoder_model=dedup.get('bi_encoder_model', 'all-mpnet-base-v2'),
                cross_encoder_model=dedup.get('cross_encoder_model', 'cross-encoder/stsb-distilroberta-base')
            )
        
        # --- Entity Extraction ---
        if 'entity_extraction' in yaml_data:
            entity = yaml_data['entity_extraction']
            config.entity_extraction = EntityExtractionConfig(
                spacy_model=entity.get('spacy_model', 'en_core_web_sm'),
                use_spacy=entity.get('use_spacy', True),
                event_keywords=entity.get('event_keywords', [])
            )
        
        # --- Sentiment Analysis ---
        if 'sentiment_analysis' in yaml_data:
            sentiment = yaml_data['sentiment_analysis']
            
            hybrid_weights = HybridWeights()
            if 'hybrid_weights' in sentiment:
                hw = sentiment['hybrid_weights']
                hybrid_weights = HybridWeights(
                    finbert_weight=hw.get('finbert_weight', 0.7),
                    rule_weight=hw.get('rule_weight', 0.3)
                )
            
            finbert_config = FinBERTConfig()
            if 'finbert' in sentiment:
                fb = sentiment['finbert']
                finbert_config = FinBERTConfig(
                    model_name=fb.get('model_name', 'ProsusAI/finbert'),
                    device=fb.get('device')
                )
            
            rule_config = RuleBasedConfig()
            if 'rule_based' in sentiment:
                rb = sentiment['rule_based']
                rule_config = RuleBasedConfig(
                    spacy_model=rb.get('spacy_model', 'en_core_web_sm'),
                    use_spacy=rb.get('use_spacy', True)
                )
            
            config.sentiment_analysis = SentimentAnalysisConfig(
                method=sentiment.get('method', 'hybrid'),
                hybrid_weights=hybrid_weights,
                finbert=finbert_config,
                rule_based=rule_config,
                entity_weights=sentiment.get('entity_weights', {}),
                event_modifiers=sentiment.get('event_modifiers', {})
            )
        
        # --- Vector Store ---
        if 'vector_store' in yaml_data:
            vs = yaml_data['vector_store']
            config.vector_store = VectorStoreConfig(
                collection_name=vs.get('collection_name', 'financial_news'),
                persist_directory=vs.get('persist_directory', 'data/chroma_db'),
                embedding_model=vs.get('embedding_model', 'all-mpnet-base-v2'),
                distance_metric=vs.get('distance_metric', 'cosine')
            )
        
        # --- Query Processing ---
        if 'query_processing' in yaml_data:
            qp = yaml_data['query_processing']
            
            multi_query = MultiQueryConfig()
            if 'multi_query' in qp:
                mq = qp['multi_query']
                multi_query = MultiQueryConfig(
                    max_context_queries=mq.get('max_context_queries', 3),
                    initial_retrieval_multiplier=mq.get('initial_retrieval_multiplier', 2)
                )
            
            reranking_weights = {}
            if 'reranking_weights' in qp:
                for strategy, weights in qp['reranking_weights'].items():
                    reranking_weights[strategy] = RerankingWeights(
                        strategy_weight=weights.get('strategy_weight', 0.5),
                        semantic_weight=weights.get('semantic_weight', 0.5)
                    )
            
            sentiment_boost = SentimentBoostConfig()
            if 'sentiment_boost' in qp:
                sb = qp['sentiment_boost']
                sentiment_boost = SentimentBoostConfig(
                    enabled=sb.get('enabled', True),
                    max_multiplier=sb.get('max_multiplier', 1.5)
                )
            
            config.query_processing = QueryProcessingConfig(
                default_top_k=qp.get('default_top_k', 10),
                min_similarity=qp.get('min_similarity', 0.3),
                multi_query=multi_query,
                reranking_weights=reranking_weights,
                sentiment_boost=sentiment_boost
            )
        
        # --- Stock Impact ---
        if 'stock_impact' in yaml_data:
            si = yaml_data['stock_impact']
            config.stock_impact = StockImpactConfig(
                confidence_thresholds=si.get('confidence_thresholds', {}),
                fuzzy_match_threshold=si.get('fuzzy_match_threshold', 0.80)
            )
        
        # --- Supply Chain ---
        if 'supply_chain' in yaml_data:
            sc = yaml_data['supply_chain']
            config.supply_chain = SupplyChainConfig(
                traversal_depth=sc.get('traversal_depth', 1),
                min_impact_score=sc.get('min_impact_score', 25.0),
                weight_decay=sc.get('weight_decay', 0.8)
            )
        
        # --- API ---
        if 'api' in yaml_data:
            api = yaml_data['api']
            config.api = APIConfig(
                host=api.get('host', '0.0.0.0'),
                port=api.get('port', 8000),
                reload=api.get('reload', True),
                title=api.get('title', 'Financial News Intelligence API'),
                description=api.get('description', 'Multi-agent AI system'),
                version=api.get('version', '1.0.0')
            )
        
        # --- Resources ---
        if 'resources' in yaml_data:
            res = yaml_data['resources']
            config.resources = ResourcesConfig(
                company_aliases=res.get('company_aliases', 'company_aliases.json'),
                sector_tickers=res.get('sector_tickers', 'sector_tickers.json'),
                regulators=res.get('regulators', 'regulators.json'),
                regulator_impact=res.get('regulator_impact', 'regulator_sector_impact.json'),
                supply_chain_graph=res.get('supply_chain_graph', 'supply_chain_graph.json')
            )
        
        # --- Logging ---
        if 'logging' in yaml_data:
            log = yaml_data['logging']
            config.logging = LoggingConfig(
                level=log.get('level', 'INFO'),
                format=log.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
        
        # --- Performance ---
        if 'performance' in yaml_data:
            perf = yaml_data['performance']
            config.performance = PerformanceConfig(
                cache_embeddings=perf.get('cache_embeddings', True),
                batch_size=perf.get('batch_size', 32),
                num_workers=perf.get('num_workers', 4)
            )
        
        # --- Development ---
        if 'development' in yaml_data:
            dev = yaml_data['development']
            config.development = DevelopmentConfig(
                debug=dev.get('debug', False),
                use_mock_data=dev.get('use_mock_data', False),
                mock_data_path=dev.get('mock_data_path', 'mock_news_data.json'),
                enable_profiling=dev.get('enable_profiling', False)
            )
        
        print(f"✓ Configuration loaded from {config_path}")
        return config
        
    except Exception as e:
        print(f"⚠ Error loading config file: {e}")
        print("  Using default configuration values")
        return Config()


# Singleton instance
_config_instance: Optional[Config] = None


def get_config(reload: bool = False) -> Config:
    """Get the global configuration instance."""
    global _config_instance
    
    if _config_instance is None or reload:
        _config_instance = load_config()
    
    return _config_instance


# Auto-load on module import
config = get_config()