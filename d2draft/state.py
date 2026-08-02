from __future__ import annotations

from dataclasses import dataclass


PHASE_VISIBLE_COUNT = {1: 0, 2: 2, 3: 4}
MAXIMUM_TEAM_SIZE = 5


def phase_for_next_pick(own_count: int) -> int | None:
    """Which round a side's next pick belongs to, or None once it has five.

    Rounds do not lock in simultaneously in practice: one player confirms before
    their teammate, and screen recognition happily catches a 3v2 board mid-round.
    A side holding three heroes is still making a second-round pick.
    """

    if not 0 <= own_count <= MAXIMUM_TEAM_SIZE:
        raise ValueError(f"a side holds between 0 and {MAXIMUM_TEAM_SIZE} heroes")
    if own_count == MAXIMUM_TEAM_SIZE:
        return None
    if own_count < PHASE_VISIBLE_COUNT[2]:
        return 1
    return 2 if own_count < PHASE_VISIBLE_COUNT[3] else 3


@dataclass(frozen=True)
class DraftState:
    """Public state before one player locks a hero in a ranked draft round.

    The two sides need not hold the same number of heroes. The model sums over
    whichever heroes are revealed and treats the rest as unknown, so an uneven
    board scores correctly; the estimate simply rests on less information about
    the side that has revealed less.
    """

    phase: int
    allies: tuple[int, ...]
    enemies: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.phase not in PHASE_VISIBLE_COUNT:
            raise ValueError("phase must be 1, 2, or 3")
        for label, heroes in (("allies", self.allies), ("enemies", self.enemies)):
            if len(heroes) > MAXIMUM_TEAM_SIZE:
                raise ValueError(
                    f"{label} holds {len(heroes)} heroes; "
                    f"a side cannot exceed {MAXIMUM_TEAM_SIZE}"
                )
        all_ids = self.allies + self.enemies
        if any(hero_id <= 0 for hero_id in all_ids):
            raise ValueError("hero ids must be positive")
        if len(set(all_ids)) != len(all_ids):
            raise ValueError("a public draft state cannot contain duplicate heroes")

    @classmethod
    def for_next_pick(
        cls, allies: tuple[int, ...], enemies: tuple[int, ...]
    ) -> "DraftState":
        """Build the state for whoever holds ``allies``, inferring their round."""

        phase = phase_for_next_pick(len(allies))
        if phase is None:
            raise ValueError("this side has already locked five heroes")
        return cls(phase=phase, allies=tuple(allies), enemies=tuple(enemies))

    @property
    def used(self) -> frozenset[int]:
        return frozenset(self.allies + self.enemies)
