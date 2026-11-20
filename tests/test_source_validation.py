"""Acquisition checks run with the Python standard library."""
import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/download_dataset.py"
SOURCE = SCRIPT.parents[1] / "data/raw/03_Dataset.csv"
SPEC = importlib.util.spec_from_file_location("download_dataset", SCRIPT)
DOWNLOAD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DOWNLOAD)


class SourceValidationTests(unittest.TestCase):
    def test_rejects_truncated_download(self):
        with self.assertRaisesRegex(ValueError, "Expected 768"):
            DOWNLOAD.validate_payload(b"")

    @unittest.skipUnless(SOURCE.is_file(), "Download the authentic Pima source to run this check")
    def test_authentic_source_shape(self):
        self.assertEqual(DOWNLOAD.validate_payload(SOURCE.read_bytes()), 768)


if __name__ == "__main__":
    unittest.main()
