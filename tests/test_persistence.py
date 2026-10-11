import json
import tempfile
import unittest
from pathlib import Path

from core.persistence import write_json_atomically


class AtomicJsonPersistenceTests(unittest.TestCase):
    def test_writes_valid_json_to_a_new_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "memory.json"

            write_json_atomically(path, {"facts": ["tea"]})

            self.assertEqual({"facts": ["tea"]}, json.loads(path.read_text(encoding="utf-8")))

    def test_failed_serialization_preserves_existing_file_and_cleans_temp_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            path.write_text('{"facts":["kept"]}', encoding="utf-8")

            with self.assertRaises(TypeError):
                write_json_atomically(path, {"facts": [object()]})

            self.assertEqual({"facts": ["kept"]}, json.loads(path.read_text(encoding="utf-8")))
            self.assertEqual([path], list(Path(directory).iterdir()))


if __name__ == "__main__":
    unittest.main()
