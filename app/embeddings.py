from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> np.ndarray:
        """Return a unit-length float32 vector."""
        ...

    def fits(self, text: str) -> bool:
        """False if the model would truncate `text` (so the embedding would ignore part of it)."""
        ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # heavy import (torch); keep it lazy

        self.model = SentenceTransformer(model_name, device="cpu")
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> np.ndarray:
        # normalize_embeddings=True -> unit vectors, so cosine similarity == dot product.
        return self.model.encode(text, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    def fits(self, text: str) -> bool:
        # MiniLM silently truncates past max_seq_length (256 tokens). Two long prompts
        # that differ only after the cutoff would get identical embeddings -> a
        # guaranteed false hit. The gateway bypasses the cache for these.
        n_tokens = len(self.model.tokenizer(text, add_special_tokens=True)["input_ids"])
        return n_tokens <= self.model.max_seq_length
