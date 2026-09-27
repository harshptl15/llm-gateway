"""Benchmark workload built from Quora Question Pairs (GLUE QQP, validation split).

Each QQP row is (question1, question2, label) where label=1 means human annotators
judged the two questions to ask the same thing. That label is our ground truth
for whether serving q1's cached answer to q2 would be CORRECT.

Download (3.7MB, gitignored):
  curl -L -o bench/data/qqp_validation.parquet \
    https://huggingface.co/datasets/nyu-mll/glue/resolve/main/qqp/validation-00000-of-00001.parquet
"""
import random
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

DATA = Path(__file__).parent / "data" / "qqp_validation.parquet"


@dataclass(frozen=True)
class Request:
    text: str
    pair: int        # index of the QQP pair this came from
    role: str        # "original" (question1) or "followup" (question2)
    is_dup: bool     # the pair's label


def _norm(s: str) -> str:
    return " ".join(s.split())


class QQP:
    def __init__(self, path: Path = DATA):
        rows = pq.read_table(path).to_pylist()
        # Label lookup over ALL 40k pairs, not just the sampled ones: if a query
        # hits some other pair's question that humans happened to label, use it.
        self.labels: dict[frozenset, int] = {}
        for r in rows:
            q1, q2 = _norm(r["question1"]), _norm(r["question2"])
            if q1 and q2:
                self.labels[frozenset((q1, q2))] = r["label"]
        self.rows = [r for r in rows if _norm(r["question1"]) and _norm(r["question2"])
                     and _norm(r["question1"]) != _norm(r["question2"])]

    def sample(self, n_pairs: int, seed: int = 0) -> list[dict]:
        return random.Random(seed).sample(self.rows, n_pairs)

    @staticmethod
    def stream(pairs: list[dict], seed: int = 0) -> list[Request]:
        """Interleave pairs into one request stream.

        Each original arrives at a uniform random time t1 in [0, 1); its follow-up
        arrives at a uniform random time after it. So a follow-up always comes after
        its original, but with arbitrary unrelated traffic in between — closer to
        real traffic than "all originals, then all follow-ups".
        """
        rng = random.Random(seed)
        events = []
        for i, p in enumerate(pairs):
            t1 = rng.random()
            t2 = t1 + rng.random() * (1 - t1)
            dup = bool(p["label"])
            events.append((t1, Request(_norm(p["question1"]), i, "original", dup)))
            events.append((t2, Request(_norm(p["question2"]), i, "followup", dup)))
        return [r for _, r in sorted(events, key=lambda e: e[0])]

    def judge(self, query: str, matched: str) -> str:
        """Was serving `matched`'s cached answer to `query` correct?

        "correct"   - identical text, or humans labeled the pair a duplicate
        "wrong"     - humans labeled the pair NOT a duplicate
        "unlabeled" - QQP has no label for this pair; we count it as wrong
                      (conservative) but report it separately.
        """
        query, matched = _norm(query), _norm(matched)
        if query == matched:
            return "correct"
        label = self.labels.get(frozenset((query, matched)))
        if label is None:
            return "unlabeled"
        return "correct" if label == 1 else "wrong"
