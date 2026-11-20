# Dataset acquisition and schema

The project includes the authentic Pima Indians Diabetes dataset as `data/raw/03_Dataset.csv`. Validate the included source or reacquire it with

```bash
python scripts/download_dataset.py
```

The exact expected file is `data/raw/03_Dataset.csv`. It has 768 records, nine numerical CSV fields and no header. The script validates its shape and target values and records its SHA-256 digest. Manual download uses the same expected path.

- [Original data mirror](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)
- [Original data dictionary](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.names)

The source identifies the original owner as the National Institute of Diabetes and Digestive and Kidney Diseases. It describes a female Pima Indian cohort aged at least 21 and diabetes signs according to World Health Organization criteria. The mirror does not provide an explicit redistribution license. The included source is attributed to the documented public mirror, and that licensing uncertainty is retained here. The unrelated current UCI dataset 34 has a different format and origin.

| Position | Project column | Meaning |
| --- | --- | --- |
| 1 | `Pregnancies` | Number of pregnancies |
| 2 | `Glucose` | Two-hour plasma glucose concentration from an oral glucose tolerance test |
| 3 | `BloodPressure` | Diastolic blood pressure in mm Hg |
| 4 | `SkinThickness` | Triceps skin-fold thickness in mm |
| 5 | `Insulin` | Two-hour serum insulin in μU/ml |
| 6 | `BMI` | Body mass index, kg/m² |
| 7 | `DiabetesPedigreeFunction` | Diabetes pedigree function |
| 8 | `Age` | Age in years |
| 9 | `Outcome` | 0 for negative, 1 for positive diabetes classification |

Zeros in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin` and `BMI` are treated as missing measurements. Zero pregnancies remains valid. Negative, nonnumeric and nonfinite features become null. Fractional pregnancies and age are invalid, and age under 21 falls outside this source cohort. Median imputation learns replacements from the training set. Invalid targets are removed and reported.

For custom headered training data, use this exact header order and pass `--header`

```text
Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
```

For inference, supply a headered CSV containing the eight feature columns. `PatientID` is optional. When absent, the pipeline creates a deterministic identifier from the supplied feature values. Identical feature tuples therefore share that generated identifier. Use your own unique `PatientID` when tracking different patients with identical measurements.

The pipeline creates `outputs/results/demo_patients.csv` by taking actual test-set records and removing labels. This is a demonstration of unseen-record inference. You can replace it with your own properly authorized patient records.

The submitted full Pima file contains 768 records, is 23,278 bytes, and is below the form's 10 MB limit. The default pipeline reads this exact file. It is tracked with source provenance, while generated patient-level outputs remain ignored by Git. Aggregate reports can be added after an actual run.
