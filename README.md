
# Credit Card Fraud Detection

Detecting fraudulent card transactions in a heavily imbalanced dataset, comparing unsupervised anomaly detection against supervised models, and serving the result through a FastAPI dashboard.

<img width="1280" height="731" alt="image" src="https://github.com/user-attachments/assets/b64c6d95-3f07-4375-937b-76802c091b09" />


## Problem

Only **492 of 284,807 transactions (0.172%)** are fraud. A model that predicts "not fraud" every time scores about 99.8% accuracy and catches nothing, so accuracy is the wrong metric here. This project evaluates with **precision, recall, F1, and PR-AUC** instead.

## Dataset

[Credit Card Fraud Detection (Kaggle, ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- 284,807 transactions, 492 fraudulent
- Features: `Time`, `Amount`, and 28 anonymized PCA components (`V1` to `V28`)

## Approach

1. **Baseline: Isolation Forest (unsupervised).** Scaled the features and tuned the `contamination` parameter to trade off recall against false positives.
2. **Supervised models:** <!-- TODO: Logistic Regression / Random Forest / XGBoost with class weights or SMOTE -->
3. **Evaluation:** confusion matrix, precision, recall, F1, PR-AUC on a held-out test set.

## Results

| Model | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| Isolation Forest (best contamination) | TODO | 0.284 | TODO | TODO |
| Logistic Regression (class weights) | TODO | TODO | TODO | TODO |
| Random Forest / XGBoost + SMOTE | TODO | TODO | TODO | TODO |

**Takeaway:** the Isolation Forest caught about 28.4% of fraud at best, which makes it a starting baseline rather than a production model. Raising recall costs precision, since more legitimate transactions get flagged and blocked. <!-- TODO: add one sentence on what the supervised models changed -->

## Dashboard

A FastAPI backend serves the trained pipeline (`fraud_pipeline.joblib`) and a small web UI:

- Converts the model's anomaly score into a **0 to 100 risk score**
- Puts the actionable columns first (`Risk Score`, `Amount`, `Time`)
- Charts built with Chart.js

## Run locally

```bash
git clone https://github.com/varun-aahil/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt   # TODO: add this file
uvicorn main:app --reload
```

Open http://127.0.0.1:8000. To retrain, run `credit_card_fraud_detection.ipynb` (download `creditcard.csv` from Kaggle first).

## Tech stack

Python, scikit-learn, pandas, NumPy, Matplotlib, FastAPI, Chart.js

## Limitations and next steps

- Only 492 positive examples, so results are sensitive to the train/test split. Consider stratified cross-validation.
- Threshold tuning based on the cost of a missed fraud vs. a blocked legitimate transaction.
