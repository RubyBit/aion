from __future__ import annotations

from .context import Context, get_default_context, reset_default_context
from .errors import AionError
from .model import LoadedModel, TensorSpec
from .tensor import Tensor
from .enums import AionDType, AionStatus

__all__ = [
    "AionDType",
    "AionError",
    "AionStatus",
    "Context",
    "LoadedModel",
    "TensorSpec",
    "Tensor",
    "load_model",
    "reset_default_context",
]

# Python package version (may be independent of the core runtime version).
__version__ = "0.0.1"


def load_model(path: str, *, thread_count: int | None = None) -> LoadedModel:
    """Convenience helper for loading models.

    - If `thread_count` is provided, a dedicated Context is created for the model.
    - If `thread_count` is omitted, the process-wide default Context is used.
    """

    if thread_count is None:
        return LoadedModel.load(get_default_context(), path)

    ctx = Context(thread_count=thread_count)
    try:
        return LoadedModel.load(ctx, path)
    except Exception:
        ctx.close()
        raise
