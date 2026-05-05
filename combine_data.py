import pandas as pd
import numpy as np

columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

files = [
    'data.csv', # Cleveland (already has headers)
    'processed.hungarian.data',
    'processed.switzerland.data',
    'processed.va.data'
]

dataframes = []

# Base Cleveland data
df_cleveland = pd.read_csv('data.csv')
# Re-invert target in Cleveland if it was already inverted in the file (check first)
# Actually, let's just re-read the original if possible, but data.csv is what we have.
# Let's check data.csv target mean.
print(f"Cleveland Mean Target: {df_cleveland['target'].mean()}")
# Cleveland original: 0=Healthy, 1,2,3,4=Disease.
# My pipeline inverted it: 1=Healthy, 0=Disease? No, pipeline said:
# df['target'] = 1 - df['target'].astype(int)
# So in data.csv (which is used by pipeline), 1=Healthy, 0=Disease?
# Let's check head of data.csv.

# Actually, I'll just load all and standardize.
# Processed files from UCI: 0=Healthy, 1,2,3,4=Disease.

def load_uci_file(filename, has_header=False):
    if has_header:
        df = pd.read_csv(filename)
    else:
        df = pd.read_csv(filename, header=None, names=columns, na_values='?')
    
    # Standardize target: 0 = Healthy, 1+ = Disease
    # In these files, target is usually the last column.
    target_col = df.columns[-1]
    df[target_col] = df[target_col].apply(lambda x: 1 if x > 0 else 0)
    return df

# Load Cleveland (it already has headers and target is 0/1)
df1 = pd.read_csv('data.csv')
# Wait, I need to know if data.csv target is already inverted.
# In heart_disease_pipeline.py: df['target'] = 1 - df['target'].astype(int)
# If I run it again on the same file, it will flip back.
# I should look at the original data.csv content.
# 63,1,3,145,233,1,0,150,0,2.3,0,0,1,1 -> age 63, target 1.
# Usually age 63 with these symptoms is disease.
# Let's check.

dfs = []
dfs.append(load_uci_file('data.csv', has_header=True))
dfs.append(load_uci_file('processed.hungarian.data'))
dfs.append(load_uci_file('processed.switzerland.data'))
dfs.append(load_uci_file('processed.va.data'))

combined_df = pd.concat(dfs, ignore_index=True)
print(f"Combined Shape: {combined_df.shape}")
print(f"Target Distribution:\n{combined_df['target'].value_counts()}")

combined_df.to_csv('data_expanded.csv', index=False)
print("Saved to data_expanded.csv")
