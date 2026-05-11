"""Core vector index module for memsearch.

Provides the primary interface for creating and querying in-memory
vector indices with support for multiple distance metrics.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Union
from dataclasses import dataclass, field


@dataclass
class SearchResult:
    """Represents a single search result."""

    id: int
    distance: float
    metadata: Optional[dict] = None

    def __repr__(self) -> str:
        return f"SearchResult(id={self.id}, distance={self.distance:.6f})"


class VectorIndex:
    """In-memory vector index supporting nearest-neighbor search.

    Supports L2 (Euclidean) and cosine similarity distance metrics.
    Vectors are stored as a NumPy matrix for efficient batch operations.

    Args:
        dim: Dimensionality of the vectors to be indexed.
        metric: Distance metric to use. One of ``'l2'`` or ``'cosine'``.

    Example::

        index = VectorIndex(dim=128, metric="cosine")
        vectors = np.random.rand(100, 128).astype(np.float32)
        index.add(vectors)
        results = index.search(vectors[:1], top_k=5)
    """

    SUPPORTED_METRICS = ("l2", "cosine")

    def __init__(self, dim: int, metric: str = "l2") -> None:
        if dim <= 0:
            raise ValueError(f"dim must be a positive integer, got {dim}")
        if metric not in self.SUPPORTED_METRICS:
            raise ValueError(
                f"Unsupported metric '{metric}'. Choose from {self.SUPPORTED_METRICS}"
            )

        self.dim = dim
        self.metric = metric
        self._vectors: Optional[np.ndarray] = None
        self._metadata: list[Optional[dict]] = []
        self._size: int = 0

    @property
    def size(self) -> int:
        """Number of vectors currently stored in the index."""
        return self._size

    def add(
        self,
        vectors: np.ndarray,
        metadata: Optional[list[Optional[dict]]] = None,
    ) -> None:
        """Add vectors to the index.

        Args:
            vectors: Array of shape ``(n, dim)`` with dtype float32.
            metadata: Optional list of metadata dicts, one per vector.

        Raises:
            ValueError: If vector dimensionality does not match index dim.
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors[np.newaxis, :]

        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"Expected vectors of dim {self.dim}, got {vectors.shape[1]}"
            )

        if self.metric == "cosine":
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            vectors = vectors / norms

        if self._vectors is None:
            self._vectors = vectors
        else:
            self._vectors = np.vstack([self._vectors, vectors])

        n = vectors.shape[0]
        if metadata is not None:
            if len(metadata) != n:
                raise ValueError(
                    f"metadata length {len(metadata)} does not match vector count {n}"
                )
            self._metadata.extend(metadata)
        else:
            self._metadata.extend([None] * n)

        self._size += n

    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
    ) -> list[list[SearchResult]]:
        """Search the index for the nearest neighbors of each query vector.

        Args:
            query: Array of shape ``(n_queries, dim)`` or ``(dim,)``.
            top_k: Number of nearest neighbors to return per query.

        Returns:
            A list of result lists, one per query vector, each sorted by
            ascending distance.

        Raises:
            RuntimeError: If the index is empty.
            ValueError: If query dimensionality does not match index dim.
        """
        if self._vectors is None or self._size == 0:
            raise RuntimeError("Index is empty. Add vectors before searching.")

        query = np.asarray(query, dtype=np.float32)
        if query.ndim == 1:
            query = query[np.newaxis, :]

        if query.shape[1] != self.dim:
            raise ValueError(
                f"Expected query dim {self.dim}, got {query.shape[1]}"
            )

        if self.metric == "cosine":
            norms = np.linalg.norm(query, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            query = query / norms
            # cosine distance = 1 - cosine similarity
            similarities = query @ self._vectors.T
            distances = 1.0 - similarities
        else:
            # Efficient squared L2 via broadcasting
            diff = query[:, np.newaxis, :] - self._vectors[np.newaxis, :, :]
            distances = np.sqrt((diff ** 2).sum(axis=-1))

        top_k = min(top_k, self._size)
        results: list[list[SearchResult]] = []
        for i, dist_row in enumerate(distances):
            indices = np.argpartition(dist_row, top_k - 1)[:top_k]
            indices = indices[np.argsort(dist_row[indices])]
            results.append(
                [
                    SearchResult(
                        id=int(idx),
                        distance=float(dist_row[idx]),
                        metadata=self._metadata[idx],
                    )
                    for idx in indices
                ]
            )
        return results

    def reset(self) -> None:
        """Remove all vectors from the index."""
        self._vectors = None
        self._metadata = []
        self._size = 0

    def __repr__(self) -> str:
        return (
            f"VectorIndex(dim={self.dim}, metric='{self.metric}', size={self._size})"
        )
