from fastapi import FastAPI,UploadFile,File
from fastapi.responses import FileResponse, Response, JSONResponse
import numpy as np
import pandas as pd
import joblib
import io

model = joblib.load('fraud_pipeline.joblib')

app = FastAPI()

@app.get('/')
async def root():
    return FileResponse('index.html')


@app.post('/predict')
async def predict(file : UploadFile = File(...)):
    df = pd.read_csv(io.BytesIO(await file.read()))
    features = df.drop(['Class'], axis=1, errors='ignore')
    
    # Get predictions and raw anomaly scores
    pred = model.predict(features)
    scores = model.decision_function(features)
    
    # Isolate anomalies (-1)
    anomaly_mask = pred == -1
    anomalies = df[anomaly_mask].copy()
    
    # Calculate a Risk Score (0-100) based on the decision function
    # decision_function returns negative values for anomalies (more negative = more anomalous)
    anomaly_scores = scores[anomaly_mask]
    if len(anomaly_scores) > 0:
        # Invert so higher positive number = higher risk
        risk = -anomaly_scores
        # Normalize to 0-100 (approximate, depending on model bounds, we use min-max scaling for the batch)
        min_risk, max_risk = risk.min(), risk.max()
        if max_risk > min_risk:
            anomalies['Risk_Score'] = ((risk - min_risk) / (max_risk - min_risk) * 100).round(1)
        else:
            anomalies['Risk_Score'] = 100.0
    else:
        anomalies['Risk_Score'] = []

    # Sort by highest risk first
    if not anomalies.empty:
        anomalies = anomalies.sort_values(by='Risk_Score', ascending=False)
    
    total_scanned = len(df)
    fraud_detected = len(anomalies)
    fraud_rate = round((fraud_detected / total_scanned) * 100, 2) if total_scanned > 0 else 0
    
    buffer = io.StringIO()
    anomalies.to_csv(buffer, index=False)
    csv_data = buffer.getvalue()
    
    display_anomalies = anomalies.head(1000).replace({np.nan: None}).to_dict(orient="records")
    
    return JSONResponse({
        "total_scanned": total_scanned,
        "fraud_detected": fraud_detected,
        "fraud_rate": fraud_rate,
        "flagged_rows": display_anomalies,
        "csv_data": csv_data
    })