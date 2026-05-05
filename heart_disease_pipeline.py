# %% [markdown]
# # Heart Disease Prediction Model Pipeline
# **Dataset:** UCI Cleveland Heart Disease Dataset  
# This script covers loading data, EDA, preprocessing, model training & evaluation, hyperparameter tuning, feature importance, saving the model, and providing a reusable prediction function.

# %% [markdown]
# ## 1. LOAD DATA
# Load the dataset and inspect basic shapes and datatypes.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, roc_curve

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

print("="*50)
print("1. LOAD DATA")
print("="*50)

# Use the expanded dataset
df = pd.read_csv('data_expanded.csv')

print(f"Dataset Shape: {df.shape}")
print("\nTarget Distribution:")
print(df['target'].value_counts())

# %% [markdown]
# ## 3. PREPROCESSING
# Handle missing values, encode categoricals, scale features, and perform train-test split.

# %%
print("\n" + "="*50)
print("3. PREPROCESSING")
print("="*50)

# USE ALL 13 FEATURES
num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

X = df[num_cols + cat_cols]
y = df['target']

# Train-Test Split (80% train, 20% test, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Impute missing values (Important for the expanded dataset as it has more missing values)
num_imputer = SimpleImputer(strategy='median')
X_train_num = pd.DataFrame(num_imputer.fit_transform(X_train[num_cols]), columns=num_cols, index=X_train.index)
X_test_num = pd.DataFrame(num_imputer.transform(X_test[num_cols]), columns=num_cols, index=X_test.index)

cat_imputer = SimpleImputer(strategy='most_frequent')
X_train_cat = pd.DataFrame(cat_imputer.fit_transform(X_train[cat_cols]), columns=cat_cols, index=X_train.index)
X_test_cat = pd.DataFrame(cat_imputer.transform(X_test[cat_cols]), columns=cat_cols, index=X_test.index)

# Scale numerical features
scaler = StandardScaler()
X_train_num_scaled = pd.DataFrame(scaler.fit_transform(X_train_num), columns=num_cols, index=X_train.index)
X_test_num_scaled = pd.DataFrame(scaler.transform(X_test_num), columns=num_cols, index=X_test.index)

# Encode categorical variables (One-Hot Encoding)
encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
X_train_cat_encoded = pd.DataFrame(encoder.fit_transform(X_train_cat), columns=encoder.get_feature_names_out(cat_cols), index=X_train.index)
X_test_cat_encoded = pd.DataFrame(encoder.transform(X_test_cat), columns=encoder.get_feature_names_out(cat_cols), index=X_test.index)

# Combine processed numerical and categorical variables
X_train_processed = pd.concat([X_train_num_scaled, X_train_cat_encoded], axis=1)
X_test_processed = pd.concat([X_test_num_scaled, X_test_cat_encoded], axis=1)

print("\nData preprocessing complete.")
print(f"Processed Train Shape: {X_train_processed.shape}")

# Save the preprocessors
joblib.dump(num_imputer, 'num_imputer.pkl')
joblib.dump(cat_imputer, 'cat_imputer.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(encoder, 'encoder.pkl')

# %% [markdown]
# ## 4 & 5. TRAIN MULTIPLE MODELS & EVALUATE
# Train Logistic Regression, Random Forest, XGBoost, SVM, and KNN. 
# Evaluate each with Accuracy, Classification Report, ROC-AUC, CM, and plot ROC Curves.

# %%
print("\n" + "="*50)
print("4 & 5. TRAIN & EVALUATE MULTIPLE MODELS")
print("="*50)

models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42)
}

results = {}
plt.figure(figsize=(10, 8))
roc_fig, roc_ax = plt.subplots(figsize=(10, 8))

for name, model in models.items():
    print(f"\n--- {name} ---")
    model.fit(X_train_processed, y_train)
    
    y_pred = model.predict(X_test_processed)
    y_proba = model.predict_proba(X_test_processed)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    results[name] = {'Accuracy': acc, 'ROC-AUC': roc_auc, 'Model': model}
    
    print(f"Accuracy Score: {acc:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix Iteration
    cm = confusion_matrix(y_test, y_pred)
    cm_fig, cm_ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=cm_ax)
    cm_ax.set_title(f'{name} - Confusion Matrix')
    cm_ax.set_ylabel('Actual')
    cm_ax.set_xlabel('Predicted')
    cm_fig.savefig(f'cm_{name.replace(" ", "_")}.png')
    plt.close(cm_fig)
    
    # ROC Curve Iteration
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

roc_ax.plot([0, 1], [0, 1], 'k--')
roc_ax.set_xlim([0.0, 1.0])
roc_ax.set_ylim([0.0, 1.05])
roc_ax.set_xlabel('False Positive Rate')
roc_ax.set_ylabel('True Positive Rate')
roc_ax.set_title('ROC Curves Comparison')
roc_ax.legend(loc="lower right")
roc_fig.savefig('roc_curves_comparison.png')
plt.close()

# %% [markdown]
# ## 6. HYPERPARAMETER TUNING
# We'll select XGBoost for hyperparameter tuning.

# %%
print("\n" + "="*50)
print("6. HYPERPARAMETER TUNING")
print("="*50)

print("Optimizing XGBoost Classifier using GridSearchCV...")
xgb_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=XGBClassifier(random_state=42),
    param_grid=xgb_params,
    cv=5,
    n_jobs=-1,
    scoring='roc_auc'
)

grid_search.fit(X_train_processed, y_train)
best_xgb = grid_search.best_estimator_

print(f"Best Parameters: {grid_search.best_params_}")

y_pred_best = best_xgb.predict(X_test_processed)
y_proba_best = best_xgb.predict_proba(X_test_processed)[:, 1]

print(f"Improved Accuracy Score: {accuracy_score(y_test, y_pred_best):.4f}")
print(f"Improved ROC-AUC Score: {roc_auc_score(y_test, y_proba_best):.4f}")

# %% [markdown]
# ## 7. FEATURE IMPORTANCE & SHAP
# Understand what features drive the final model.

# %%
print("\n" + "="*50)
print("7. FEATURE IMPORTANCE")
print("="*50)

# XGBoost Feature Importances
importances = best_xgb.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(10, 8))
plt.title("Feature Importances (Tuned XGBoost)")
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), [X_train_processed.columns[i] for i in indices])
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig('feature_importances.png')
plt.close()

# SHAP values
print("Generating SHAP summary plot...")
explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer.shap_values(X_test_processed)

# Handle different shap version outputs for binary classification
if isinstance(shap_values, list):
    shap_vals = shap_values[1]
elif len(np.array(shap_values).shape) == 3:
    shap_vals = shap_values[:, :, 1]
else:
    shap_vals = shap_values

shap.summary_plot(shap_vals, X_test_processed, show=False)
plt.savefig('shap_summary.png', bbox_inches='tight')
plt.close()

# %% [markdown]
# ## 8. SAVE THE MODEL
# Save the tuned model to disk.

# %%
print("\n" + "="*50)
print("8. SAVE THE MODEL")
print("="*50)

joblib.dump(best_xgb, 'heart_disease_model.pkl')
print("Model saved successfully as 'heart_disease_model.pkl'")

# %% [markdown]
# ## 9. PREDICT FUNCTION
# A reusable predict function taking raw dictionary input.

# %%
print("\n" + "="*50)
print("9. PREDICT FUNCTION")
print("="*50)

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
    
    # List of expected cols (FULL 13 FEATURES)
    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
    cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
    
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

# Quick Test
sample_patient = {
    'age': 63, 'sex': 1, 'cp': 1, 'trestbps': 145, 'chol': 233, 'fbs': 1, 
    'restecg': 2, 'thalach': 150, 'exang': 0, 'oldpeak': 2.3, 'slope': 3, 
    'ca': 0.0, 'thal': 6.0
}
print(f"Sample Input: {sample_patient}")
prediction, probability = predict(sample_patient)
print(f"Prediction: {'Disease (1)' if prediction == 1 else 'No Disease (0)'}")
print(f"Probability of Heart Disease: {probability:.4f}")
