"""
LLM Query Router Agent
Routes user queries to optimal search strategies using LLM reasoning.
File: app/agents/llm_query_router.py
"""

from typing import Optional, Dict, Any

from app.core.llm_schemas import QueryRouterSchema, QueryIntent
from app.services.llm_client import GroqLLMClient, LLMServiceError
from app.core.config_loader import get_config


class LLMQueryRouter:
    """
    LLM-powered query router that determines optimal search strategy.
    Uses LLM to both classify strategy AND extract entities dynamically.
    """
    
    def __init__(self, llm_client: Optional[GroqLLMClient] = None):
        """
        Initialize LLM query router.
        
        Args:
            llm_client: Optional pre-configured LLM client
        """
        config = get_config()
        
        # Initialize LLM client (use fast model for routing)
        if llm_client is None:
            self.llm_client = GroqLLMClient(
                model=config.llm.models.fast,
                temperature=0.1,
                max_tokens=2048
            )
        else:
            self.llm_client = llm_client
        
        print(f"✓ LLMQueryRouter initialized")
        print(f"  - Model: {self.llm_client.model}")
        print(f"  - Entity extraction: LLM-based (no static lists)")
    
    def _build_few_shot_examples(self) -> str:
        """
        Construct few-shot examples teaching query strategy selection.
        """
        examples = """
**Example 1: Direct Entity Query**
Query: "HDFC Bank dividend news"
Analysis:
- Strategy: DIRECT_ENTITY
- Reasoning: Query explicitly mentions company name "HDFC Bank" and event "dividend"
- Entities: ["HDFC Bank"]
- Stock Symbols: ["HDFCBANK"]
- Sectors: ["Banking"]
- Refined Query: "HDFC Bank dividend announcement payout"
- Confidence: 0.95

**Example 2: Sector-Wide Query**
Query: "Banking sector outlook"
Analysis:
- Strategy: SECTOR_WIDE
- Reasoning: Query asks about entire "Banking sector" without specific company
- Entities: []
- Stock Symbols: []
- Sectors: ["Banking"]
- Refined Query: "Banking sector performance outlook trends"
- Confidence: 0.90

**Example 3: Regulatory Query**
Query: "RBI rate hike impact"
Analysis:
- Strategy: REGULATORY
- Reasoning: Query mentions regulator "RBI" and policy action "rate hike"
- Entities: []
- Stock Symbols: []
- Sectors: []
- Regulators: ["RBI"]
- Refined Query: "RBI repo rate increase monetary policy impact"
- Confidence: 0.92

**Example 4: Sentiment-Driven Query**
Query: "Bullish tech stocks"
Analysis:
- Strategy: SENTIMENT_DRIVEN
- Reasoning: Query explicitly asks for "Bullish" sentiment in "tech" sector
- Entities: []
- Stock Symbols: []
- Sectors: ["IT"]
- Sentiment Filter: "Bullish"
- Refined Query: "positive technology sector stocks growth"
- Confidence: 0.88

**Example 5: Cross-Impact Query**
Query: "Steel price increase effect on auto sector"
Analysis:
- Strategy: CROSS_IMPACT
- Reasoning: Query asks about supply chain relationship between "Steel" and "Auto"
- Entities: []
- Stock Symbols: []
- Sectors: ["Steel", "Auto"]
- Refined Query: "steel price impact automobile manufacturing costs"
- Confidence: 0.85

**Example 6: Temporal Query**
Query: "Recent Reliance earnings"
Analysis:
- Strategy: TEMPORAL
- Reasoning: Query uses temporal keyword "Recent" with company "Reliance" and event "earnings"
- Entities: ["Reliance Industries"]
- Stock Symbols: ["RELIANCE"]
- Sectors: ["Energy"]
- Temporal Scope: "recent"
- Refined Query: "Reliance Industries latest quarterly earnings results"
- Confidence: 0.93

**Example 7: Semantic Search (Fallback)**
Query: "market trends in emerging technologies"
Analysis:
- Strategy: SEMANTIC_SEARCH
- Reasoning: Broad, thematic query without specific entities or sectors
- Entities: []
- Stock Symbols: []
- Sectors: []
- Refined Query: "emerging technology market trends investment opportunities"
- Confidence: 0.70

**Example 8: Multiple Companies**
Query: "Compare TCS and Infosys Q3 results"
Analysis:
- Strategy: DIRECT_ENTITY
- Reasoning: Query mentions two specific companies "TCS" and "Infosys"
- Entities: ["TCS", "Infosys"]
- Stock Symbols: ["TCS", "INFY"]
- Sectors: ["IT"]
- Refined Query: "TCS Infosys quarterly results comparison Q3"
- Confidence: 0.94

**Example 9: Unknown Company**
Query: "What's happening with Zomato IPO"
Analysis:
- Strategy: DIRECT_ENTITY
- Reasoning: Query mentions specific company "Zomato" even if not in reference data
- Entities: ["Zomato"]
- Stock Symbols: ["ZOMATO"]
- Sectors: ["E-commerce"]
- Refined Query: "Zomato IPO listing initial public offering"
- Confidence: 0.92
"""
        return examples.strip()
    
    def _build_routing_prompt(self, query: str) -> str:
        """
        Construct routing prompt with few-shot examples.
        
        Args:
            query: User's natural language query
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""Analyze this user query and determine the optimal search strategy.

**User Query**: "{query}"

**Available Strategies**:
1. **DIRECT_ENTITY**: User mentions specific company/companies by name
   - Use when: Company explicitly named (e.g., "HDFC Bank results", "TCS dividend")
   - Extract: Company names, stock symbols, sectors

2. **SECTOR_WIDE**: User asks about an entire industry sector
   - Use when: Sector mentioned without specific companies (e.g., "Banking sector news")
   - Extract: Sector names

3. **REGULATORY**: User asks about regulatory bodies or policy changes
   - Use when: Regulator mentioned (RBI, SEBI, etc.) or policy focus
   - Extract: Regulator names, affected sectors

4. **SENTIMENT_DRIVEN**: User filters by sentiment (Bullish/Bearish/Neutral)
   - Use when: Sentiment keyword present (e.g., "Bullish IT stocks")
   - Extract: Sentiment filter, relevant sectors

5. **CROSS_IMPACT**: User asks about supply chain or cross-sectoral effects
   - Use when: Multiple sectors mentioned with causal relationship
   - Extract: Source and target sectors

6. **TEMPORAL**: User emphasizes time-based filtering
   - Use when: Temporal keywords present (recent, latest, today, last week)
   - Extract: Entities, temporal scope

7. **SEMANTIC_SEARCH**: Broad/thematic query without clear structure
   - Use when: None of above strategies fit clearly
   - Extract: Refined search query

**Your Task**:
1. Choose the MOST APPROPRIATE strategy (only one)
2. Extract ALL relevant entities from the query:
   - **Companies**: Any company/organization mentioned (even if not well-known)
   - **Stock Symbols**: Ticker symbols in NSE/BSE format (e.g., HDFCBANK, TCS, RELIANCE)
   - **Sectors**: Industry sectors (Banking, IT, Auto, Pharma, etc.)
   - **Regulators**: Regulatory bodies (RBI, SEBI, US FDA, etc.)
3. Identify sentiment filter if mentioned (Bullish/Bearish/Neutral)
4. Identify temporal scope if mentioned (recent/last_week/last_month)
5. Generate a refined query optimized for semantic search
6. Assign confidence score (0.0-1.0) based on clarity

**Important Guidelines**:
- **Extract ANY company mentioned**, even if unfamiliar (e.g., startups, foreign companies)
- **Infer ticker symbols** based on company names (use common patterns)
- **Identify sectors** even if not explicitly mentioned (e.g., "TCS" → IT sector)
- Prioritize DIRECT_ENTITY if ANY company name appears
- Confidence: 0.9+ for clear queries, 0.7-0.9 for moderate, 0.5-0.7 for ambiguous
- Refined query: Expand with synonyms and related terms for better retrieval

**Entity Extraction Rules**:
- Company names: Full names as mentioned (e.g., "HDFC Bank", "Reliance Industries")
- Stock symbols: Use NSE format, infer from company name if possible
- Sectors: Standard industry classifications (Banking, IT, Auto, Energy, etc.)
- Regulators: Include acronym and full name if possible

Now analyze the user query and return structured routing decision with ALL extracted entities."""
        
        return prompt
    
    def _validate_and_enrich(self, raw_result: QueryRouterSchema) -> QueryRouterSchema:
        """
        Post-process routing result: normalize and deduplicate entities.
        
        Args:
            raw_result: Raw LLM routing output
            
        Returns:
            Validated and enriched QueryRouterSchema
        """
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
    
    def route_query(self, query: str) -> QueryRouterSchema:
        """
        Route query using LLM reasoning and return strategy + entities.
        
        This method uses LLM to:
        1. Determine the optimal search strategy
        2. Extract ALL entities dynamically (no static reference lists)
        3. Generate refined query for semantic search
        
        Args:
            query: User's natural language query
            
        Returns:
            QueryRouterSchema with routing decision and extracted entities
            
        Raises:
            LLMServiceError: If routing fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        # Build system message with few-shot examples
        system_message = f"""You are an expert query analyzer for financial news search.
Your task is to analyze user queries and determine the optimal search strategy.

You must extract ALL entities mentioned in the query, even if they are:
- Unfamiliar companies or startups
- Foreign companies
- New market entrants
- Any organization relevant to financial news

Do NOT limit extraction to a predefined list - use your knowledge to identify all relevant entities.

**Few-Shot Examples**:

{self._build_few_shot_examples()}

---

Always return results in strict JSON format matching the QueryRouterSchema.
Choose the MOST APPROPRIATE strategy based on query content and structure.
Extract ALL relevant entities with high precision, including unknown/new companies."""
        
        # Build routing prompt
        prompt = self._build_routing_prompt(query)
        
        try:
            # Call LLM with structured output
            result_dict = self.llm_client.generate_structured_output(
                prompt=prompt,
                schema=QueryRouterSchema,
                system_message=system_message
            )
            
            # Validate with Pydantic
            raw_result = QueryRouterSchema.model_validate(result_dict)
            
            # Post-process and enrich
            validated_result = self._validate_and_enrich(raw_result)
            
            return validated_result
            
        except Exception as e:
            raise LLMServiceError(f"Query routing failed: {e}")