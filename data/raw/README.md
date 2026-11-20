# Authentic Pima source

`03_Dataset.csv` is the complete authentic Pima Indians Diabetes source obtained from the documented public mirror. It contains 768 records, nine numerical fields, no header, and 23,278 bytes. The submission filename preserves the original CSV bytes.

The default training command uses this exact file. Run `python scripts/download_dataset.py` from the project root to validate the included source. Use `--force` to reacquire the documented source at the same path. Keep the CSV headerless.

`03_Dataset.provenance.json` records the source URL, shape, size and SHA-256 digest. The public mirror does not specify a dataset redistribution license.
