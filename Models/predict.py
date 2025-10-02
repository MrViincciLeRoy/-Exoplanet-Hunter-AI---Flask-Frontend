
def predict_exoplanet(features_dict):
    """Predict exoplanet classification"""
    import joblib
    import pandas as pd
    import numpy as np

    model = joblib.load('models/final_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    imputer = joblib.load('models/imputer.pkl')
    label_encoder = joblib.load('models/label_encoder.pkl')

    df = pd.DataFrame([features_dict])
    df_imputed = imputer.transform(df)
    df_scaled = scaler.transform(df_imputed)

    prediction = model.predict(df_scaled)[0]
    probabilities = model.predict_proba(df_scaled)[0]

    predicted_class = label_encoder.classes_[prediction]
    confidence = probabilities[prediction]

    return predicted_class, confidence, probabilities
