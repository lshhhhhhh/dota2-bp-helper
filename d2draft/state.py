from __future__ import annotations

from dataclasses import dataclass


PHASE_VISIBLE_COUNT = {1: 0, 2: 2, 3: 4}


@dataclass(frozen=True)
class DraftState:
    """Public state before one player locks a hero in a ranked draft round."""

    phase: int
    allies: tuple[int, ...]
    enemies: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.phase not in PHASE_VISIBLE_COUNT:
            raise ValueError("phase must be 1, 2, or 3")
        expected = PHASE_VISIBLE_COUNT[self.phase]
        if len(self.allies) != expected or len(self.enemies) != expected:
            raise ValueError(
                f"phase {self.phase} requires {expected} visible heroes per side; "
                f"got {len(self.allies)} and {len(self.enemies)}"
            )
        all_ids = self.allies + self.enemies
        if any(hero_id <= 0 for hero_id in all_ids):
            raise ValueError("hero ids must be positive")
        if len(set(all_ids)) != len(all_ids):
            raise ValueError("a public draft state cannot contain duplicate heroes")

    @property
    def used(self) -> frozenset[int]:
        return frozenset(self.allies + self.enemies)

