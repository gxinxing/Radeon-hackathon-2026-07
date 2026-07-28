"""Keyword-based knowledge retriever for crypto trading RAG.

Retrieves relevant trading knowledge entries based on keyword matching
against user input. No external dependencies — pure Python TF-IDF-lite.

The knowledge base covers:
- Indicator documentation (what each indicator does, typical params)
- Strategy patterns (crossover, mean reversion, breakout, etc.)
- Risk management rules (position sizing, stop placement)
- Market context rules (volatility regimes, regime detection)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .knowledge_entries import KNOWLEDGE_ENTRIES


@dataclass
class KnowledgeEntry:
    """A single knowledge base entry."""
    keywords: list[str]
    category: str  # indicator | strategy | risk | market
    title: str
    content: str
    weight: float = 1.0  # Priority weight for ranking


class KnowledgeRetriever:
    """Retrieves relevant knowledge entries based on user query.

    Uses simple keyword matching with TF-based scoring.
    No external dependencies (no embeddings, no vector DB).
    """

    def __init__(self, entries: list[KnowledgeEntry] | None = None):
        self._entries = entries or _load_entries()
        self._index = self._build_index()

    def retrieve(
        self,
        query: str,
        max_results: int = 5,
        min_score: float = 0.1,
    ) -> list[KnowledgeEntry]:
        """Retrieve relevant knowledge entries for a query.

        Args:
            query: User's natural language input.
            max_results: Maximum entries to return.
            min_score: Minimum relevance score threshold.

        Returns:
            List of KnowledgeEntry sorted by relevance.
        """
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scored: list[tuple[float, KnowledgeEntry]] = []
        for entry in self._entries:
            score = self._score_entry(entry, query_tokens)
            if score >= min_score:
                scored.append((score, entry))

        scored.sort(key=lambda x: (-x[0], -x[1].weight))
        return [entry for _, entry in scored[:max_results]]

    def retrieve_as_context(
        self,
        query: str,
        max_results: int = 5,
        max_chars: int = 2000,
    ) -> str:
        """Retrieve knowledge and format as LLM prompt context string.

        Args:
            query: User's natural language input.
            max_results: Max entries to retrieve.
            max_chars: Truncate context to this many characters.

        Returns:
            Formatted context string for LLM prompt injection.
        """
        entries = self.retrieve(query, max_results=max_results)
        if not entries:
            return ""

        sections: list[str] = []
        current_category = ""
        total_chars = 0

        for entry in entries:
            if entry.category != current_category:
                current_category = entry.category
                section_header = f"\n## {current_category.title()}"
                if total_chars + len(section_header) > max_chars:
                    break
                sections.append(section_header)
                total_chars += len(section_header)

            entry_text = f"\n### {entry.title}\n{entry.content}"
            if total_chars + len(entry_text) > max_chars:
                # Truncate
                remaining = max_chars - total_chars
                if remaining > 50:
                    sections.append(entry_text[:remaining] + "...")
                break
            sections.append(entry_text)
            total_chars += len(entry_text)

        return "\n".join(sections)

    def _tokenize(self, text: str) -> set[str]:
        """Tokenize query into lowercase word tokens."""
        # Handle Chinese characters as individual tokens
        text = text.lower()
        # Split on non-alphanumeric (keeps Chinese chars as single tokens)
        tokens = re.findall(r"[a-z]+|[\u4e00-\u9fff]+", text)
        return set(tokens)

    def _score_entry(
        self,
        entry: KnowledgeEntry,
        query_tokens: set[str],
    ) -> float:
        """Score an entry's relevance to the query tokens."""
        score = 0.0
        for keyword in entry.keywords:
            kw_lower = keyword.lower()
            # Exact match
            if kw_lower in query_tokens:
                score += 1.0 * entry.weight
            # Partial match (keyword is substring of a token or vice versa)
            else:
                for token in query_tokens:
                    if kw_lower in token or token in kw_lower:
                        score += 0.3 * entry.weight
        return score

    def _build_index(self) -> dict[str, list[KnowledgeEntry]]:
        """Build keyword → entries index for fast lookup."""
        index: dict[str, list[KnowledgeEntry]] = {}
        for entry in self._entries:
            for kw in entry.keywords:
                kw_lower = kw.lower()
                index.setdefault(kw_lower, []).append(entry)
        return index


def _load_entries() -> list[KnowledgeEntry]:
    """Load knowledge entries from knowledge_entries.py."""
    entries: list[KnowledgeEntry] = []
    for raw in KNOWLEDGE_ENTRIES:
        entries.append(KnowledgeEntry(**raw))
    return entries


# Module-level convenience function
_retriever: KnowledgeRetriever | None = None


def retrieve_knowledge(query: str, max_results: int = 5) -> str:
    """Retrieve knowledge context for a query.

    Convenience function that caches the retriever instance.

    Args:
        query: User's natural language input.
        max_results: Max entries to retrieve.

    Returns:
        Formatted context string for LLM prompt injection.
    """
    global _retriever
    if _retriever is None:
        _retriever = KnowledgeRetriever()
    return _retriever.retrieve_as_context(query, max_results=max_results)
