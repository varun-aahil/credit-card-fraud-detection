# Credit Card Fraud Detection (Anomaly Detection)

The Aim:

Given the transaction data in the dataset - we want to detect fraudulent transactions (a mere 0.43% out of 284.807), when even by guessing the transaction is not fraudulent on all transactions it would still hit 99.5% accuracy, which is useless. We want to try and detect these minority class examples, that is anomalies in this dataset and evaluate the predictions with Confusion matrix, precision and recall metrics.

The Dataset:

* Number of transactions: 284,807

* Fraudulent transactions: 492

* Imbalance: Fraud data is 0.172% of overall data
* Features: We have Time (time since the first transaction in the dataset), Amount, and 28 principle components with anonymized feature names V1-V28.

The creditcard.csv raw data is available at kaggle here - https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud 

The Methodology:

* The model: I decided to go with the unsupervised 'Isolation Forest'. This approach is to explicitly isolate the outliers by randomly partitioning a random subset of the data using random splitting criteria for random features. This is efficient in the high-dimensional dataset.
* Data processing: I scaled the feature matrix to standardize the financial amount that might heavily outweigh other metrics. 
* Evaluation: We skip the traditional accuracyscore here, we use classificationreport.

The Result & Business Implications:

After playing with the contamination this anomaly detection model managed to isolate 28.4% of all fraudulent transactions at best while not predicting alot of false postiives. In an actual use case - one can tweak the contamination value in Isolation Forest that helps the business to decide the threshold in how many anomalies they want detected, highlighting the most crucial business decision: as recall is maximised (all fraud identified), the precision will fall due to increase in false positives, hence declining valid transactions. This serves as the starting base.

The Tech Stack:

* Language: Python

* Libraries used: Sklearn, Pandas, Numpy, Matplotlib
