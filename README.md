# Heart Disease Risk Prediction using KNN and SVM

This project implements a **heart disease risk classification** workflow using the supplied CSV dataset.

It includes:

- Missing-value handling with median imputation.
- Feature scaling with `StandardScaler`.
- K-Nearest Neighbors (KNN).
- Support Vector Machine (SVM).
- Logistic Regression as a comparison baseline.
- Hyperparameter tuning in the Google Colab notebook.
- **Recall-focused** model selection because missing a positive/risk case can be important in a screening context.
- A Streamlit interface for interactive predictions.
- A probability-threshold control so you can explore the precision/recall trade-off.

> **Important:** This is an educational machine-learning screening project, not a medical device or diagnostic system. Predictions should not replace evaluation by a qualified healthcare professional.

## Dataset

The included file is:

`heart_disease_dataset.csv`

The target column is:

`heart_disease`

Expected feature columns:

```text
age
sex
cp
trestbps
chol
fbs
restecg
thalach
exang
oldpeak
slope
ca
thal
smoking
diabetes
bmi
```

The supplied dataset contains 3,069 rows and 16 input features plus the target.

## 1. Google Colab

Open `heart_disease_prediction_colab.ipynb` in Google Colab.

The first notebook cell installs the required packages. The next cells:

1. Upload/load the CSV.
2. Inspect the dataset.
3. Check missing values and class distribution.
4. Split the data using stratification.
5. Build preprocessing pipelines.
6. Tune KNN, SVM and Logistic Regression with cross-validation.
7. Use **recall** as the primary GridSearchCV scoring metric.
8. Report accuracy, precision, recall, F1 and ROC-AUC.
9. Plot a confusion matrix.
10. Demonstrate a sample prediction.

In Colab, you can use **Runtime → Run all**.

## 2. Run Streamlit locally

Put these files in the same folder:

```text
heart_disease_streamlit_project/
├── app.py
├── heart_disease_dataset.csv
├── requirements.txt
└── README.md
```

Create/activate a virtual environment if desired, then install dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
streamlit run app.py
```

The app opens in your browser.

## 3. Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload:
   - `app.py`
   - `requirements.txt`
   - `heart_disease_dataset.csv`
   - `README.md`
3. In Streamlit Community Cloud, create a new app and select the repository.
4. Set the main file to:

```text
app.py
```

5. Deploy.

The app also provides an optional CSV uploader if you want to test another dataset with the same column names.

## Model design

### Preprocessing

Every classifier uses this pipeline:

```text
Input data
   ↓
Median imputation
   ↓
StandardScaler
   ↓
Classifier
```

Scaling is particularly relevant for KNN because KNN uses distances, and for SVM because feature magnitudes can affect the learned margin/kernel geometry.

### Hyperparameter tuning

The notebook uses `GridSearchCV` with stratified cross-validation and `scoring="recall"`.

The final Streamlit app uses the recall-tuned settings found in the notebook configuration:

- **KNN:** `n_neighbors=5`, `weights="uniform"`, `p=1`
- **SVM:** `C=0.1`, `kernel="rbf"`, `class_weight="balanced"`
- **Logistic Regression:** `C=1.0`, `class_weight="balanced"`

The notebook remains the authoritative place to rerun tuning if the dataset changes.

## Why recall?

Recall for the positive class is:

```text
Recall = TP / (TP + FN)
```

A recall-focused screening model attempts to reduce false negatives. This can also create more false positives, so recall should be considered together with precision, specificity, F1, ROC-AUC and the confusion matrix.

## Interpreting the Streamlit threshold

The default threshold is `0.50`.

Moving the threshold lower generally makes the classifier more willing to assign the positive class. This can increase recall while also increasing false positives. The threshold slider is included for educational exploration of this trade-off.

## Notes about coded variables

The dataset contains numeric/coded clinical attributes such as:

- `sex`
- `cp`
- `fbs`
- `restecg`
- `exang`
- `slope`
- `ca`
- `thal`
- `smoking`
- `diabetes`

The app uses the codes present in the dataset rather than inventing a different encoding. Check the dataset/source documentation before interpreting a code clinically.

## Project files

- `heart_disease_prediction_colab.ipynb` — complete Colab workflow.
- `app.py` — Streamlit application.
- `requirements.txt` — Python dependencies.
- `heart_disease_dataset.csv` — supplied dataset, renamed for deployment convenience.
- `README.md` — setup and deployment instructions.
