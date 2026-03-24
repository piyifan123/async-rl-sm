"""Fixed-capacity replica pool for constraining concurrent in-flight work.

A simulation may have separate pools for inference replicas (bounding rollout
throughput) and judge replicas (bounding judging throughput).  The scheduler
consults each pool's :attr:`available` slots before dispatching work to tasks,
and releases slots when :meth:`Task.tick` reports completions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["ReplicaPool"]


@dataclass
class ReplicaPool:
    """A pool of identical replicas with a fixed slot capacity.

    Tracks how many slots are currently occupied and exposes an
    :meth:`acquire` / :meth:`release` interface for the scheduler.

    Args:
        name: Human-readable label for this pool (e.g. ``"inference"``,
            ``"judge"``).
        capacity: Maximum number of concurrent slots (must be >= 1).

    Examples:
        >>> pool = ReplicaPool(name="inference", capacity=4)
        >>> pool.acquire(3)
        >>> pool.available
        1
        >>> pool.release(2)
        >>> pool.available
        3
    """

    name: str
    capacity: int
    _in_use: int = field(init=False, default=0, repr=False)

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {self.capacity}")

    @property
    def in_use(self) -> int:
        """Number of slots currently occupied."""
        return self._in_use

    @property
    def available(self) -> int:
        """Number of free slots that can be acquired."""
        return self.capacity - self._in_use

    def acquire(self, n: int) -> None:
        """Reserve *n* replica slots.

        Args:
            n: Number of slots to acquire (must be >= 1).

        Raises:
            ValueError: If *n* is non-positive or exceeds :attr:`available`.
        """
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        if n > self.available:
            raise ValueError(
                f"Cannot acquire {n} slots from pool {self.name!r} "
                f"(only {self.available} available)"
            )
        self._in_use += n
        self._assert_invariant()

    def release(self, n: int) -> None:
        """Free *n* previously-acquired replica slots.

        Args:
            n: Number of slots to release (must be >= 1).

        Raises:
            ValueError: If *n* is non-positive or exceeds :attr:`in_use`.
        """
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        if n > self._in_use:
            raise ValueError(
                f"Cannot release {n} slots from pool {self.name!r} (only {self._in_use} in use)"
            )
        self._in_use -= n
        self._assert_invariant()

    def _assert_invariant(self) -> None:
        """Verify ``0 <= in_use <= capacity``.

        Raises:
            AssertionError: If the invariant is violated.
        """
        assert 0 <= self._in_use <= self.capacity, (
            f"Invariant violated for pool {self.name!r}: "
            f"in_use={self._in_use}, capacity={self.capacity}"
        )
