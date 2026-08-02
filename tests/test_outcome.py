from __future__ import annotations

import json
import unittest

import numpy as np

from d2draft.outcome import OutcomeEmbeddingModel, OutcomeExamples, outcome_examples


class OutcomeModelTest(unittest.TestCase):
    def test_winning_and_losing_picks_receive_opposite_labels(self) -> None:
        row = {
            "radiant_win": 1,
            "phase_1_radiant": json.dumps([1, 2]),
            "phase_1_dire": json.dumps([6, 7]),
            "phase_2_radiant": json.dumps([3, 4]),
            "phase_2_dire": json.dumps([8, 9]),
            "phase_3_radiant": json.dumps([5]),
            "phase_3_dire": json.dumps([10]),
        }
        examples = outcome_examples([row], {hero: hero - 1 for hero in range(1, 11)})
        self.assertEqual(len(examples), 10)
        self.assertTrue(np.all(examples.outcome[:5] == 1.0))
        self.assertTrue(np.all(examples.outcome[5:] == 0.0))
        self.assertEqual(examples.phase.tolist(), [1, 1, 2, 2, 3, 1, 1, 2, 2, 3])

    def test_training_penalizes_a_repeated_losing_candidate(self) -> None:
        count = 400
        candidates = np.tile(np.asarray([0, 1], dtype=np.int16), count // 2)
        outcomes = (candidates == 0).astype(np.float32)
        examples = OutcomeExamples(
            allies=np.full((count, 4), -1, dtype=np.int16),
            enemies=np.full((count, 4), -1, dtype=np.int16),
            candidate=candidates,
            phase=np.ones(count, dtype=np.int8),
            outcome=outcomes,
        )
        rng = np.random.default_rng(7)
        model = OutcomeEmbeddingModel.create(heroes=2, dimensions=4, rng=rng)
        model.fit(
            examples,
            epochs=10,
            batch_size=64,
            learning_rate=0.02,
            l2=0.0,
            rng=rng,
        )
        scores = model.score_state([], [], phase=1)
        self.assertGreater(scores[0], 0.75)
        self.assertLess(scores[1], 0.25)


if __name__ == "__main__":
    unittest.main()
