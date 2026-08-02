from __future__ import annotations

import unittest

from d2draft.state import MAXIMUM_TEAM_SIZE, DraftState, phase_for_next_pick


class PhaseForNextPickTest(unittest.TestCase):
    def test_rounds_follow_how_many_a_side_already_holds(self) -> None:
        self.assertEqual([phase_for_next_pick(n) for n in range(5)], [1, 1, 2, 2, 3])

    def test_a_full_side_has_no_next_pick(self) -> None:
        self.assertIsNone(phase_for_next_pick(MAXIMUM_TEAM_SIZE))

    def test_impossible_counts_are_rejected(self) -> None:
        for count in (-1, MAXIMUM_TEAM_SIZE + 1):
            with self.assertRaises(ValueError):
                phase_for_next_pick(count)


class DraftStateTest(unittest.TestCase):
    def test_sides_may_hold_different_numbers_of_heroes(self) -> None:
        # One player locks in before their teammate, so a 3v2 board is ordinary.
        state = DraftState(phase=2, allies=(1, 2, 3), enemies=(4, 5))
        self.assertEqual(state.used, frozenset({1, 2, 3, 4, 5}))

    def test_a_side_facing_a_finished_opponent_is_valid(self) -> None:
        state = DraftState(phase=3, allies=(1, 2, 3, 4), enemies=(5, 6, 7, 8, 9))
        self.assertEqual(len(state.enemies), MAXIMUM_TEAM_SIZE)

    def test_more_than_five_heroes_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DraftState(phase=3, allies=(1, 2, 3, 4, 5, 6), enemies=())

    def test_duplicate_and_invalid_ids_are_still_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DraftState(phase=2, allies=(1, 2), enemies=(2, 3))
        with self.assertRaises(ValueError):
            DraftState(phase=2, allies=(0,), enemies=())

    def test_unknown_phase_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DraftState(phase=4, allies=(), enemies=())

    def test_for_next_pick_infers_the_round_from_the_asking_side(self) -> None:
        self.assertEqual(DraftState.for_next_pick((1, 2, 3), (4, 5)).phase, 2)
        self.assertEqual(DraftState.for_next_pick((1, 2, 3, 4), (5, 6)).phase, 3)
        self.assertEqual(DraftState.for_next_pick((), ()).phase, 1)

    def test_for_next_pick_refuses_a_side_that_is_done(self) -> None:
        with self.assertRaises(ValueError):
            DraftState.for_next_pick((1, 2, 3, 4, 5), (6, 7))


if __name__ == "__main__":
    unittest.main()
