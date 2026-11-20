# Validation record

The project includes syntax validation, formula and acquisition tests, and authentic-source Spark integration tests. Test commands and prerequisites appear in the README.

## Checks executed during creation

The following checks were executed on 20 November 2026 with Python 3.12.14.

```bash
python -m compileall -q main.py src pipeline scripts tests
python -m unittest tests.test_evaluation tests.test_source_validation tests.test_csv_validation -v
```

- Python syntax compilation passed for `main.py`, `src/`, `pipeline/`, `scripts/` and `tests/` using the bundled Python runtime.
- The standalone acquisition validator rejected an empty download with the expected 768-record shape error.
- Ten standard-library tests passed. They cover positive-class metric formulas, zero-denominator handling, model selection by F1 and tie breakers, truncated downloads, the authentic source's 768-by-9 shape, short and extra CSV records, empty field preservation, header widths and malformed quoting.
- The authentic downloaded source also passed the nine-field streaming preflight check for all 768 records.

## Runtime checks

The build coordinator records any further executed runtime checks below. The authentic Pima source was downloaded into the test workspace and validated with the acquisition checker. The publication project includes the validated complete source at `data/raw/03_Dataset.csv` with its byte-level source fingerprint. At the time of these checks, the bundled Python environment had no PySpark, matplotlib or pytest installed. The Spark pipeline, classifier fitting, charts, saved-model inference and Spark integration tests could not be executed in that environment. No runtime model metrics are pre-populated in the source project.

The Docker environment is provided for a repeatable Linux runtime. Its build and execution require Docker Desktop or a compatible Docker engine.

## Publication preparation checks

On 20 November 2026, Python 3.12.14 parsed all 15 Python source and test files without syntax errors. `python -s -m unittest tests.test_evaluation tests.test_source_validation tests.test_csv_validation -v` passed 10 tests against the publication project.

The submitted `03_Dataset.csv` contains 768 records and is 23,278 bytes, below 10 MB. Its bytes match the standalone dataset prepared for Drive upload. The project input path and documented filenames were checked after renaming. Spark execution and its infrastructure checks retain the unverified status documented above.
