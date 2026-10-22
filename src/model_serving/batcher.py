"""Dynamic request batching: flush on max size OR max wait."""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from .errors import OverloadedError, ValidationError


@dataclass(frozen=True)
class BatchConfig:
    max_batch_size: int = 8
    max_wait_ms: float = 20.0
    max_queue_size: int = 128

    def __post_init__(self) -> None:
        if self.max_batch_size < 1:
            raise ValidationError("max_batch_size must be >= 1")
        if self.max_wait_ms <= 0:
            raise ValidationError("max_wait_ms must be > 0")
        if self.max_queue_size < 1:
            raise ValidationError("max_queue_size must be >= 1")


@dataclass
class _Item:
    features: tuple[float, ...]
    event: threading.Event = field(default_factory=threading.Event)
    result: float | None = None
    error: Exception | None = None
    enqueued_at: float = field(default_factory=time.monotonic)


class DynamicBatcher:
    """Background-thread batcher. predict_fn maps a batch -> scores."""

    def __init__(self, predict_fn: Callable[[list[tuple[float, ...]]], list[float]],
                 config: BatchConfig | None = None):
        self._fn = predict_fn
        self.config = config or BatchConfig()
        self._q: queue.Queue[_Item] = queue.Queue(maxsize=self.config.max_queue_size)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.batches_executed = 0
        self.items_processed = 0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="batcher")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def __enter__(self): self.start(); return self
    def __exit__(self, *a): self.stop()

    @property
    def queue_depth(self) -> int:
        return self._q.qsize()

    def submit(self, features: tuple[float, ...], timeout_s: float = 5.0) -> tuple[float, int]:
        """Submit one request; returns (score, batch_size)."""
        if self._thread is None or not self._thread.is_alive():
            self.start()
        item = _Item(features=tuple(features))
        try:
            self._q.put_nowait(item)
        except queue.Full:
            raise OverloadedError("batch queue full — load shed") from None
        if not item.event.wait(timeout=timeout_s):
            raise TimeoutError("batch wait timed out")
        if item.error is not None:
            raise item.error
        assert item.result is not None
        # batch size stashed on item via result tuple hack: stored in _last_batch attr
        return item.result, getattr(item, "_batch_size", 1)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                first = self._q.get(timeout=0.05)
            except queue.Empty:
                continue
            batch = [first]
            deadline = time.monotonic() + self.config.max_wait_ms / 1000.0
            while len(batch) < self.config.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(self._q.get(timeout=remaining))
                except queue.Empty:
                    break
            try:
                scores = self._fn([b.features for b in batch])
                if len(scores) != len(batch):
                    raise RuntimeError("predict_fn returned wrong batch size")
                for item, score in zip(batch, scores):
                    item.result = score
                    item._batch_size = len(batch)  # type: ignore[attr-defined]
            except Exception as exc:  # propagate to all waiters
                for item in batch:
                    item.error = exc
            finally:
                self.batches_executed += 1
                self.items_processed += len(batch)
                for item in batch:
                    item.event.set()
                    self._q.task_done()


def collect_batch(items: list, max_batch_size: int) -> list[list]:
    """Pure helper: chunk a list into batches (tested directly)."""
    if max_batch_size < 1:
        raise ValidationError("max_batch_size must be >= 1")
    return [items[i:i + max_batch_size] for i in range(0, len(items), max_batch_size)]
