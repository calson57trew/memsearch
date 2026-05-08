"""memsearch - A high-performance in-memory vector search library.

This package provides efficient vector similarity search capabilities
with support for multiple distance metrics and index types.

Note: Forked from zilliztech/memsearch for personal learning/experimentation.
"""

from memsearch.index import Index
from memsearch.collection import Collection
from memsearch.exceptions import (
    MemsearchError,
    IndexNotFoundError,
    DimensionMismatchError,
    InvalidMetricError,
)

__version__ = "0.1.0"
__author__ = "memsearch contributors"
__license__ = "Apache-2.0"

# Default number of results returned by search if top_k is not specified
DEFAULT_TOP_K = 10

__all__ = [
    "Index",
    "Collection",
    "MemsearchError",
    "IndexNotFoundError",
    "DimensionMismatchError",
    "InvalidMetricError",
    "DEFAULT_TOP_K",
    "__version__",
]
