"""Typed errors for model serving."""
from __future__ import annotations

class ServingError(Exception):
    """Base error."""

class ValidationError(ServingError, ValueError):
    """Invalid request or configuration."""

class ModelNotFoundError(ServingError, LookupError):
    """Requested model / version not in registry."""

class OverloadedError(ServingError):
    """Load-shedding: system at capacity."""

class DeadlineExceededError(ServingError, TimeoutError):
    """Request deadline exceeded."""

class RegistryError(ServingError):
    """Registry invariant violated."""
