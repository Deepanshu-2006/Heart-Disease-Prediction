import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def predict(patient_data):
    """
    Takes a dictionary of patient features as input, preprocesses it, 
    and returns a prediction (0 or 1) and disease probability.
    """
    model = joblib.load('heart_disease_model.pkl')
    num_imp = joblib.load('num_imputer.pkl')
    cat_imp = joblib.load('cat_imputer.pkl')
    scr = joblib.load('scaler.pkl')
    enc = joblib.load('encoder.pkl')
    
    # List of expected cols
    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    
    df_patient = pd.DataFrame([patient_data])
    
    # Fill missing cols if any
    for col in num_cols + cat_cols:
        if col not in df_patient.columns:
            df_patient[col] = np.nan
            
    # Preprocess
    df_num = pd.DataFrame(num_imp.transform(df_patient[num_cols]), columns=num_cols)
    df_cat = pd.DataFrame(cat_imp.transform(df_patient[cat_cols]), columns=cat_cols)
    
    df_num_scaled = pd.DataFrame(scr.transform(df_num), columns=num_cols)
    df_cat_encoded = pd.DataFrame(enc.transform(df_cat), columns=enc.get_feature_names_out(cat_cols))
    
    X_unseen = pd.concat([df_num_scaled, df_cat_encoded], axis=1)
    
    pred = model.predict(X_unseen)[0]
    prob = model.predict_proba(X_unseen)[0, 1]
    
    return pred, prob

if __name__ == '__main__':
    # Sample Patient 1 (Low Risk Profile)
    sample_patient_1 = {
        'age': 35, 'sex': 0, 'cp': 0, 'trestbps': 110, 'chol': 180, 'fbs': 0, 
        'restecg': 0, 'thalach': 160, 'exang': 0, 'oldpeak': 0.1, 'slope': 2, 
        'ca': 0.0, 'thal': 2.0
    }
    
    # Sample Patient 2 (High Risk Profile)
    sample_patient_2 = {
        'age': 65, 'sex': 1, 'cp': 3, 'trestbps': 180, 'chol': 280, 'fbs': 1, 
        'restecg': 2, 'thalach': 110, 'exang': 1, 'oldpeak': 4.5, 'slope': 0, 
        'ca': 2.0, 'thal': 3.0
    }
    
    print("--- Running Predictions using Saved XGBoost Model ---")
    
    print(f"\nEvaluating Patient 1 (Low Risk Profile)...")
    pred1, prob1 = predict(sample_patient_1)
    print(f"Prediction: {'Disease (1)' if pred1 == 1 else 'No Disease (0)'}")
    print(f"Probability: {prob1:.4f}")
    
    print(f"\nEvaluating Patient 2 (High Risk Profile)...")
    pred2, prob2 = predict(sample_patient_2)
    print(f"Prediction: {'Disease (1)' if pred2 == 1 else 'No Disease (0)'}")
    print(f"Probability: {prob2:.4f}")
