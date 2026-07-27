"""Merge all training datasets into a unified JSONL for SFT.

Combines FNSPID, FinGPT, and NL→DSL pairs into a single
training file with consistent format.
"""

from __future__ import annotations

import json
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


def merge_datasets(
    output_path: str | None = None,
    shuffle: bool = True,
) -> str:
    """Merge all processed datasets into one JSONL.

    Args:
        output_path: Custom output path.
        shuffle: Whether to shuffle the merged data.

    Returns:
        Path to the merged JSONL file.
    """
    all_samples: list[dict] = []
    source_counts: dict[str, int] = {}

    input_files = [
        "fnspid_train.jsonl",
        "fingpt_train.jsonl",
        "dsl_pairs.jsonl",
    ]

    for fname in input_files:
        fpath = DATA_DIR / fname
        if not fpath.exists():
            print(f"[Merge] Skipping {fname} — not found")
            continue

        count = 0
        with open(fpath) as f:
            for line in f:
                sample = json.loads(line.strip())
                all_samples.append(sample)
                source = sample.get("source", "unknown")
                source_counts[source] = source_counts.get(source, 0) + 1
                count += 1
        print(f"[Merge] Loaded {count} samples from {fname}")

    if shuffle:
        import random
        random.seed(42)
        random.shuffle(all_samples)

    output = output_path or str(DATA_DIR / "merged_train.jsonl")
    with open(output, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"[Merge] Total: {len(all_samples)} samples → {output}")
    print(f"[Merge] Sources: {source_counts}")
    return output


if __name__ == "__main__":
    merge_datasets()
