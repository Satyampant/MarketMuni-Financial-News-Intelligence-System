"""
LLM Query Router Agent
Routes user queries to optimal search strategies.
"""

from typing import Optional, Dict, Any

from app.core.llm_schemas import QueryRouterSchema, QueryIntent
from app.services.llm_client import GroqLLMClient, LLMServiceError
from app.core.config_loader import get_config


class LLMQueryRouter:
    """LLM-based router for determining search strategies and dynamic entity extraction."""
    
    def __init__(self, llm_client: Optional[GroqLLMClient] = None):
        self.config = get_config()
        
        # Initialize LLM client (defaults to fast model for routing efficiency)
        if llm_client is None:
            self.llm_client = GroqLLMClient(
                model=self.config.llm.models.fast,
                temperature=0.1,
                max_tokens=2048
            )
        else:
            self.llm_client = llm_client
        
        print(f"✓ LLMQueryRouter initialized")
        print(f"  - Model: {self.llm_client.model}")
        print(f"  - Entity extraction: LLM-based")
    
    def _build_few_shot_examples(self) -> str:
        return self.config.prompts.query_routing.get('few_shot_examples', '')
    
    def _build_routing_prompt(self, query: str) -> str:
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
        
        # Ensure refined query is not empty; fallback to first entity or default
        if not raw_result.refined_query.strip():
            raw_result.refined_query = raw_result.entities[0] if raw_result.entities else "financial news"
        
        return raw_result
    
    def route_query(self, query: str) -> QueryRouterSchema:
        """
        Route query using LLM reasoning.
        Determines strategy, extracts entities dynamically, and generates a refined query.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        # Prepare system message with few-shot examples
        system_message_template = self.config.prompts.query_routing.get('system_message', '')
        few_shot_examples = self._build_few_shot_examples()
        system_message = system_message_template.format(few_shot_examples=few_shot_examples)
            
        prompt = self._build_routing_prompt(query)
        
        try:
            # Generate structured output directly from LLM
            result_dict = self.llm_client.generate_structured_output(
                prompt=prompt,
                schema=QueryRouterSchema,
                system_message=system_message
            )
            
            # Validate via Pydantic and enrich
            raw_result = QueryRouterSchema.model_validate(result_dict)
            validated_result = self._validate_and_enrich(raw_result)
            
            return validated_result
            
        except Exception as e:
            raise LLMServiceError(f"Query routing failed: {e}")