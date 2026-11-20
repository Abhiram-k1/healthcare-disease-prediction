"""Structural CSV parser checks use generic utility fixtures, not patient records."""
import tempfile
import unittest
from pathlib import Path

from src.csv_validation import validate_csv_shape


class CsvValidationTests(unittest.TestCase):
    def check_text(self, text, **options):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table.csv"
            path.write_text(text, encoding="utf-8")
            return validate_csv_shape(path, **options)

    def test_short_record_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "2 fields, expected 3"):
            self.check_text("a,b\n", expected_width=3)

    def test_extra_record_field_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "3 fields, expected 2"):
            self.check_text("a,b,c\n", expected_width=2)

    def test_missing_values_retain_their_field_positions(self):
        self.assertEqual(self.check_text("first,second\n,\n", header=True), ["first", "second"])

    def test_header_determines_patient_record_width(self):
        with self.assertRaisesRegex(ValueError, "1 fields, expected 2"):
            self.check_text("first,second\na\n", header=True)

    def test_unclosed_quote_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Malformed CSV quoting"):
            self.check_text('"a,b\n', expected_width=2)


if __name__ == "__main__":
    unittest.main()
