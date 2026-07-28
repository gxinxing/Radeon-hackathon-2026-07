"""RAG knowledge base for crypto trading strategy generation.

Provides keyword-based retrieval of trading knowledge to enhance LLM
prompt context. No external dependencies (no vector DB, no embeddings).

Knowledge entries are structured YAML-like dicts with:
- keywords: terms that trigger retrieval
- category: indicator / strategy / risk / market
- content: knowledge text injected into LLM prompt

Usage:
    from src.knowledge_base.retriever import KnowledgeRetriever

    kr = KnowledgeRetriever()
    context = kr.retrieve("BTC RSI超卖策略，止损3%")
    # Returns relevant indicator docs, strategy patterns, risk rules
"""

from .retriever import KnowledgeRetriever, retrieve_knowledge

__all__ = ["KnowledgeRetriever", "retrieve_knowledge"]
