"""Prepare FNSPID dataset for QLoRA fine-tuning.

FNSPID: Financial News and Stock Price Integration Dataset.
Source: https://huggingface.co/datasets/Zihan1004/FNSPID

This script downloads, filters, and formats FNSPID data into
instruction-response pairs suitable for SFT (Supervised Fine-Tuning).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from datasets import load_dataset


OUTPUT_DIR = Path(__file__).parent.parent / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Templates for converting FNSPID news → instruction pairs
NEWS_ANALYSIS_TEMPLATES = [
    {
        "instruction": (
            "You are a professional crypto trader. "
            "Analyze the following market news and explain its potential "
            "impact on price movement.\n\nNews: {news}"
        ),
        "response": (
            "Market Analysis:\n"
            "1. Sentiment: {sentiment}\n"
            "2. Key factors: {factors}\n"
            "3. Expected impact: {impact}\n"
            "4. Trading implication: {implication}"
        ),
    },
    {
        "instruction": (
            "As an experienced trader, assess this news headline. "
            "Is it bullish or bearish for the market?\n\n"
            "Headline: {headline}"
        ),
        "response": (
            "Assessment: {assessment}\n"
            "Confidence: {confidence}\n"
            "Reasoning: {reasoning}"
        ),
    },
]


def prepare_fnspid(
    max_samples: int = 50000,
    output_path: str | None = None,
) -> str:
    """Download and process FNSPID dataset.

    Args:
        max_samples: Maximum number of samples to process.
        output_path: Custom output path. Defaults to data/processed/.

    Returns:
        Path to the processed JSONL file.
    """
    print("[FNSPID] Loading dataset from HuggingFace...")
    try:
        ds = load_dataset(
            "Zihan1004/FNSPID",
            "stock_news",
            split="train",
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"[FNSPID] Warning: Could not load full dataset: {e}")
        print("[FNSPID] Falling back to filtered subset...")
        ds = load_dataset(
            "Zihan1004/FNSPID",
            split="train[:5000]",
            trust_remote_code=True,
        )

    print(f"[FNSPID] Loaded {len(ds)} samples, processing up to {max_samples}...")

    processed = []
    for i, sample in enumerate(ds):
        if i >= max_samples:
            break

        # FNSPID columns: url, title, summary, date, stock, price_movement
        title = sample.get("title", "")
        summary = sample.get("summary", "")
        date = sample.get("date", "")
        stock = sample.get("stock", "")

        if not title and not summary:
            continue

        # Format using template
        template = NEWS_ANALYSIS_TEMPLATES[i % len(NEWS_ANALYSIS_TEMPLATES)]
        news_text = summary if summary else title

        # Heuristic sentiment from title/summary keywords
        sentiment = _detect_sentiment(title + " " + summary)

        instruction = template["instruction"].format(
            news=news_text[:500],  # Truncate to fit context
            headline=title,
        )

        response = template["response"].format(
            sentiment=sentiment,
            factors=title[:100] if title else "Multiple factors at play",
            impact="Short-term price movement expected" if sentiment != "Neutral" else "Limited immediate impact",
            implication=_sentiment_to_advice(sentiment),
            assessment=sentiment,
            confidence="Medium" if sentiment != "Neutral" else "Low",
            reasoning=_generate_reasoning(title, summary, sentiment),
        )

        processed.append({
            "instruction": instruction,
            "input": "",
            "output": response,
            "source": "fnspid",
        })

    output = output_path or str(OUTPUT_DIR / "fnspid_train.jsonl")
    with open(output, "w") as f:
        for item in processed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[FNSPID] Processed {len(processed)} samples → {output}")
    return output


def _detect_sentiment(text: str) -> str:
    """Heuristic sentiment detection from text keywords."""
    bullish_words = [
        "surge", "rally", "gain", "bullish", "breakthrough", "upgrade",
        "beat", "exceed", "soar", "jump", "rise", "positive", "growth",
        "上涨", "突破", "利好", "增长", "超预期", "飙升",
    ]
    bearish_words = [
        "drop", "fall", "crash", "bearish", "downgrade", "miss",
        "decline", "plunge", "sell-off", "loss", "negative", "warning",
        "下跌", "暴跌", "利空", "下滑", "不及预期", "警告",
    ]

    text_lower = text.lower()
    bull_count = sum(1 for w in bullish_words if w in text_lower)
    bear_count = sum(1 for w in bearish_words if w in text_lower)

    if bull_count > bear_count:
        return "Bullish"
    elif bear_count > bull_count:
        return "Bearish"
    return "Neutral"


def _sentiment_to_advice(sentiment: str) -> str:
    advice = {
        "Bullish": "Consider long positions with appropriate risk management.",
        "Bearish": "Consider reducing exposure or exploring short positions.",
        "Neutral": "Monitor for clearer directional signals before entering positions.",
    }
    return advice.get(sentiment, "Monitor position closely.")


def _generate_reasoning(title: str, summary: str, sentiment: str) -> str:
    """Generate a brief reasoning string."""
    text = title if title else summary[:100]
    reason_map = {
        "Bullish": f"Positive sentiment detected in '{text[:50]}...'. "
        f"This suggests potential upward price pressure.",
        "Bearish": f"Negative signals from '{text[:50]}...'. "
        f"Market may face downward pressure.",
        "Neutral": f"Mixed signals in '{text[:50]}...'. "
        f"No clear directional bias.",
    }
    return reason_map.get(sentiment, "Further analysis needed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare FNSPID dataset")
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    prepare_fnspid(max_samples=args.max_samples, output_path=args.output)
