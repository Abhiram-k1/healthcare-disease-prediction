"""Download the authentic Pima source without synthesizing patient records."""
import argparse
import csv
import hashlib
import io
import json
import math
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
NAMES_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.names"


def validate_payload(payload):
    rows = list(csv.reader(io.StringIO(payload.decode("utf-8-sig"))))
    rows = [row for row in rows if row]
    if len(rows) != 768:
        raise ValueError(f"Expected 768 authentic Pima records, received {len(rows)}. Verify the source before use.")
    for index, row in enumerate(rows, start=1):
        if len(row) != 9:
            raise ValueError(f"Source row {index} has {len(row)} fields, expected nine.")
        values = [float(value) for value in row]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Source row {index} has a nonfinite value")
        if values[-1] not in (0.0, 1.0):
            raise ValueError(f"Source row {index} has an invalid target")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Download Pima Indians Diabetes from the documented public mirror")
    parser.add_argument("--output", type=Path, default=ROOT / "data/raw/03_Dataset.csv")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and not args.force:
        print(f"Existing dataset retained at {output}. Use --force to download again.")
        validate_payload(output.read_bytes())
        return
    try:
        request = urllib.request.Request(URL, headers={"User-Agent": "BigDataAssignment3/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
        count = validate_payload(payload)
    except (urllib.error.URLError, ValueError) as error:
        raise SystemExit(f"Dataset acquisition failed: {error}. Download {URL} manually and save to {output}.") from error
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".download")
    temporary.write_bytes(payload)
    temporary.replace(output)
    metadata = {"download_url": URL, "data_dictionary_url": NAMES_URL, "rows": count, "columns": 9, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(), "redistribution_license": "The public mirror has no explicit dataset redistribution license. Consult source terms before uploading."}
    output.with_suffix(".provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Downloaded {count} genuine records to {output} ({len(payload)} bytes).")


if __name__ == "__main__":
    main()
