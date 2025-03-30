import pandas as pd

df_features = pd.read_pickle("features_per_sentence.pkl")

# Check structure
print(df_features.head())