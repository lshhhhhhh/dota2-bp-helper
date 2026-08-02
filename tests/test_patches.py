from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from d2draft.collect import init_database
from d2draft.patches import canonical_patch_for_time


class PatchSchemaTest(unittest.TestCase):
    def test_schema_and_time_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connection = init_database(Path(directory) / "test.sqlite3")
            candidate_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(candidates)")
            }
            match_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(matches)")
            }
            self.assertIn("canonical_patch", candidate_columns)
            self.assertTrue(
                {"data_source", "source_patch_id", "canonical_patch"} <= match_columns
            )
            timestamp = int(datetime(2026, 4, 1, tzinfo=UTC).timestamp())
            self.assertEqual(canonical_patch_for_time(connection, timestamp), "7.41")
            connection.close()


if __name__ == "__main__":
    unittest.main()
