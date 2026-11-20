"""Stream CSV structure checks before Spark's parser can pad or truncate fields."""
import csv
from pathlib import Path


def validate_csv_shape(path, expected_width=None, header=False, expected_header=None):
    """Check record widths while retaining empty field values for later imputation."""
    path = Path(path)
    columns = None
    records = 0
    try:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.reader(stream, strict=True)
            if header:
                columns = next(reader, None)
                if not columns:
                    raise ValueError(f"CSV header is missing in {path}")
                if len(set(columns)) != len(columns):
                    raise ValueError(f"CSV header has duplicate column names in {path}")
                if expected_header is not None and columns != expected_header:
                    raise ValueError(f"Training CSV header must use this exact order: {', '.join(expected_header)}")
                expected_width = expected_width or len(columns)
            if expected_width is None:
                raise ValueError("CSV validation requires a known field count or a header")
            for row in reader:
                if not row:
                    continue
                if len(row) != expected_width:
                    raise ValueError(f"CSV record ending at line {reader.line_num} has {len(row)} fields, expected {expected_width} in {path}")
                records += 1
    except csv.Error as error:
        raise ValueError(f"Malformed CSV quoting in {path}: {error}") from error
    if records == 0:
        raise ValueError(f"CSV contains no data records in {path}")
    return columns
