"""Checkpointed sweep IO shared by every suite runner (kinesim AND Isaac).

One episode = one JSON line appended to ``episodes.jsonl`` the moment it
finishes, so any interrupted run resumes: completed ``(scenario, condition,
seed)`` triples are skipped on restart. ``finalize`` rewrites the derived
artifacts (episodes.csv, aggregate.csv, summary.md) from the full row pool.
"""

from __future__ import annotations

import json
from pathlib import Path

from .metrics import aggregate, to_markdown, write_csv


def ep_key(scenario: str, condition: str, seed: int) -> str:
    return f"{scenario}|{condition}|{seed}"


def load_checkpoint(out_dir: Path) -> tuple[list[dict], set[str]]:
    """Existing rows + their episode keys (empty when starting fresh)."""
    rows: list[dict] = []
    keys: set[str] = set()
    ckpt = out_dir / "episodes.jsonl"
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            rows.append(r)
            keys.add(ep_key(r["scenario"], r["condition"], r["seed"]))
    return rows, keys


class Checkpointer:
    """Append-only episode sink; keeps the in-memory row pool in sync."""

    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.rows, self.done = load_checkpoint(self.out_dir)
        self._fh = (self.out_dir / "episodes.jsonl").open("a")

    def skip(self, scenario: str, condition: str, seed: int) -> bool:
        return ep_key(scenario, condition, seed) in self.done

    def add(self, row: dict) -> None:
        self.rows.append(row)
        self.done.add(ep_key(row["scenario"], row["condition"], row["seed"]))
        self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()

    def finalize(self, md_fields: list[str] | None = None) -> str:
        write_csv(self.out_dir / "episodes.csv", self.rows)
        agg = aggregate(self.rows, keys=("scenario", "condition"))
        write_csv(self.out_dir / "aggregate.csv", agg)
        md = to_markdown(agg, fields=md_fields)
        (self.out_dir / "summary.md").write_text(md + "\n")
        return md

    def close(self) -> None:
        self._fh.close()
