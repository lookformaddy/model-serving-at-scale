"""Model version registry for serving."""
from __future__ import annotations

from .errors import ModelNotFoundError, RegistryError
from .models import LinearScorer, ModelSpec


class ModelRegistry:
    """In-memory registry: multiple versions per model, one active."""

    def __init__(self) -> None:
        self._models: dict[str, dict[str, LinearScorer]] = {}
        self._active: dict[str, str] = {}
        self._history: dict[str, list[str]] = {}

    def register(self, scorer: LinearScorer, *, make_active: bool = False) -> ModelSpec:
        spec = scorer.spec
        versions = self._models.setdefault(spec.name, {})
        if spec.version in versions:
            raise RegistryError(f"duplicate registration {spec.key}")
        versions[spec.version] = scorer
        if make_active or spec.name not in self._active:
            self._active[spec.name] = spec.version
            self._history.setdefault(spec.name, []).append(spec.version)
        return spec

    def get(self, name: str, version: str | None = None) -> LinearScorer:
        if name not in self._models:
            raise ModelNotFoundError(f"model {name!r} not found")
        v = version or self._active.get(name)
        if v is None or v not in self._models[name]:
            raise ModelNotFoundError(f"version {name}:{v} not found")
        return self._models[name][v]

    def active_version(self, name: str) -> str:
        if name not in self._active:
            raise ModelNotFoundError(f"model {name!r} not found")
        return self._active[name]

    def promote(self, name: str, version: str) -> None:
        self.get(name, version)  # validates
        self._active[name] = version
        self._history.setdefault(name, []).append(version)

    def rollback(self, name: str) -> str:
        hist = self._history.get(name, [])
        if len(hist) < 2:
            raise RegistryError(f"no previous version to roll back to for {name!r}")
        hist.pop()  # drop current
        prev = hist[-1]
        self._active[name] = prev
        return prev

    def list_models(self) -> list[str]:
        return sorted(self._models)

    def list_versions(self, name: str) -> list[str]:
        if name not in self._models:
            raise ModelNotFoundError(f"model {name!r} not found")
        return sorted(self._models[name])

    def __len__(self) -> int:
        return sum(len(v) for v in self._models.values())


def build_demo_registry() -> ModelRegistry:
    reg = ModelRegistry()
    reg.register(LinearScorer(ModelSpec("churn", "v1", (0.8, -0.4, 0.6, 0.2), bias=-0.1, description="stable")), make_active=True)
    reg.register(LinearScorer(ModelSpec("churn", "v2", (1.0, -0.5, 0.7, 0.3), bias=-0.2, description="canary candidate")))
    reg.register(LinearScorer(ModelSpec("fraud", "v1", (1.2, 0.9, -0.3), bias=-0.5)), make_active=True)
    return reg
