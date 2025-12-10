"""
Query Router Agent with MongoDB Filter Generation
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from app.core.llm_schemas import QueryRouterSchema, QueryIntent
from app.core.models import QueryRouting
from app.services.llm_client import GroqLLMClient, LLMServiceError
from app.core.config_loader import get_config



class QueryRouter:
    """
    LLM-based query router that generates MongoDB-compatible filters.
    Replaces LLMQueryRouter with MongoDB filter generation capability.
    """
    
    def __init__(self, llm_client: Optional[GroqLLMClient] = None):
        self.config = get_config()
        
        if llm_client is None:
            self.llm_client = GroqLLMClient(
                model=self.config.llm.models.fast,
                temperature=0.1,
                max_tokens=2048
            )
        else:
            self.llm_client = llm_client
        
        print(f"✓ QueryRouter initialized (MongoDB filter generation)")
        print(f"  - Model: {self.llm_client.model}")
        print(f"  - Filter generation: MongoDB query syntax")
    
    def _build_few_shot_examples(self) -> str:
        """Get few-shot examples from config."""
        return self.config.prompts.query_routing.get('few_shot_examples', '')
    
    def _build_routing_prompt(self, query: str) -> str:
        """Build the routing prompt with query."""
        prompt_template = self.config.prompts.query_routing.get('task_prompt', '')
        return prompt_template.format(query=query)
    
    def _validate_and_enrich(self, raw_result: QueryRouterSchema) -> QueryRouterSchema:
        """Post-process routing result to normalize and deduplicate extracted data."""
        
        # Deduplicate entities (case-insensitive)
        seen_entities = set()
        unique_entities = []
        for entity in raw_result.entities:
            entity_lower = entity.lower().strip()
            if entity_lower and entity_lower not in seen_entities:
                unique_entities.append(entity.strip())
                seen_entities.add(entity_lower)
        raw_result.entities = unique_entities
        
        # Deduplicate and uppercase stock symbols
        raw_result.stock_symbols = list(set(s.upper().strip() for s in raw_result.stock_symbols if s))
        
        # Deduplicate sectors
        seen_sectors = set()
        unique_sectors = []
        for sector in raw_result.sectors:
            sector_lower = sector.lower().strip()
            if sector_lower and sector_lower not in seen_sectors:
                unique_sectors.append(sector.strip())
                seen_sectors.add(sector_lower)
        raw_result.sectors = unique_sectors
        
        # Deduplicate regulators
        seen_regulators = set()
        unique_regulators = []
        for regulator in raw_result.regulators:
            regulator_lower = regulator.lower().strip()
            if regulator_lower and regulator_lower not in seen_regulators:
                unique_regulators.append(regulator.strip())
                seen_regulators.add(regulator_lower)
        raw_result.regulators = unique_regulators
        
        # Ensure refined query is not empty
        if not raw_result.refined_query.strip():
            raw_result.refined_query = raw_result.entities[0] if raw_result.entities else "financial news"
        
        return raw_result
    
    def route_query(self, query: str) -> QueryRouting:
        """
        Route query using LLM reasoning.
        Returns QueryRouting with MongoDB filter generation support.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        # Prepare system message with few-shot examples
        system_message_template = self.config.prompts.query_routing.get('system_message', '')
        few_shot_examples = self._build_few_shot_examples()
        system_message = system_message_template.format(few_shot_examples=few_shot_examples)
            
        prompt = self._build_routing_prompt(query)
        
        try:
            # Generate structured output from LLM
            result_dict = self.llm_client.generate_structured_output(
                prompt=prompt,
                schema=QueryRouterSchema,
                system_message=system_message
            )
            
            # Validate and enrich
            raw_result = QueryRouterSchema.model_validate(result_dict)
            validated_result = self._validate_and_enrich(raw_result)
            
            # Convert to QueryRouting dataclass
            routing = QueryRouting(
                entities=validated_result.entities,
                sectors=validated_result.sectors,
                stock_symbols=validated_result.stock_symbols,
                sentiment_filter=validated_result.sentiment_filter,
                refined_query=validated_result.refined_query,
                strategy=validated_result.strategy,
                confidence=validated_result.confidence,
                reasoning=validated_result.reasoning,
                regulators=validated_result.regulators,
                temporal_scope=validated_result.temporal_scope
            )
            
            return routing
            
        except Exception as e:
            raise LLMServiceError(f"Query routing failed: {e}")
    
    def generate_mongodb_filter(self, routing: QueryRouting) -> Dict[str, Any]:
        """
        Generate MongoDB query filter from routing result.
        
        Returns MongoDB-compatible filter dict based on strategy and extracted entities.
        Empty dict {} means no filtering (unrestricted search).
        """
        
        if routing.strategy == QueryIntent.DIRECT_ENTITY:
            # Priority 1: Filter by stock symbols (highest precision)
            if routing.stock_symbols:
                if len(routing.stock_symbols) == 1:
                    return {
                        "impacted_stocks.symbol": routing.stock_symbols[0]
                    }
                else:
                    return {
                        "impacted_stocks.symbol": {"$in": routing.stock_symbols}
                    }
            
            # Priority 2: Filter by company names
            elif routing.entities:
                if len(routing.entities) == 1:
                    return {
                        "entities.Companies": routing.entities[0]
                    }
                else:
                    return {
                        "entities.Companies": {"$in": routing.entities}
                    }
            
            return {}
        
        elif routing.strategy == QueryIntent.SECTOR_WIDE:
            # Filter by sector names
            if routing.sectors:
                if len(routing.sectors) == 1:
                    return {
                        "entities.Sectors": routing.sectors[0]
                    }
                else:
                    return {
                        "entities.Sectors": {"$in": routing.sectors}
                    }
            return {}
        
        elif routing.strategy == QueryIntent.REGULATORY:
            # Filter by regulator names
            if routing.regulators:
                if len(routing.regulators) == 1:
                    return {
                        "entities.Regulators": routing.regulators[0]
                    }
                else:
                    return {
                        "entities.Regulators": {"$in": routing.regulators}
                    }
            return {}
        
        elif routing.strategy == QueryIntent.SENTIMENT_DRIVEN:
            # Filter by sentiment + optional sectors
            if routing.sentiment_filter:
                base_filter = {
                    "sentiment.classification": routing.sentiment_filter
                }
                
                if routing.sectors:
                    if len(routing.sectors) == 1:
                        return {
                            "$and": [
                                base_filter,
                                {"entities.Sectors": routing.sectors[0]}
                            ]
                        }
                    else:
                        return {
                            "$and": [
                                base_filter,
                                {"entities.Sectors": {"$in": routing.sectors}}
                            ]
                        }
                
                return base_filter
            return {}
        
        elif routing.strategy == QueryIntent.CROSS_IMPACT:
            # Filter by multiple sectors (supply chain analysis)
            if len(routing.sectors) >= 2:
                return {
                    "entities.Sectors": {"$in": routing.sectors}
                }
            elif routing.sectors:
                return {
                    "entities.Sectors": routing.sectors[0]
                }
            return {}
        
        elif routing.strategy == QueryIntent.TEMPORAL:
            # Temporal strategy: Falls back to entity filtering if available
            if routing.stock_symbols:
                if len(routing.stock_symbols) == 1:
                    return {
                        "impacted_stocks.symbol": routing.stock_symbols[0]
                    }
                else:
                    return {
                        "impacted_stocks.symbol": {"$in": routing.stock_symbols}
                    }
            elif routing.entities:
                if len(routing.entities) == 1:
                    return {
                        "entities.Companies": routing.entities[0]
                    }
                else:
                    return {
                        "entities.Companies": {"$in": routing.entities}
                    }
            return {}
        
        elif routing.strategy == QueryIntent.SEMANTIC_SEARCH:
            # No metadata filtering - pure vector search
            return {}
        
        # Default: no filter
        return {}