# IMPORT LIBRARIES

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# LOAD DATASET
df = pd.read_csv("../data/housing.csv")
# REMOVE EXTRA SPACES
df.columns = df.columns.str.strip()
print("Dataset Loaded Successfully")
print(df.head())
# BASIC INFO
print("Shape of Dataset")
print(df.shape)
print("Columns")
print(df.columns)
print("Info")
df.info()
print("Description")
print(df.describe())
# CHECK MISSING VALUES
print("Missing Values")
print(df.isnull().sum())
# HANDLE MISSING VALUES
numeric_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns
for column in numeric_columns:
    df[column] = df[column].fillna(
        df[column].median()
    )
print("Missing Values After Handling")
print(df.isnull().sum())
# CHECK DUPLICATES
print("Duplicate Rows")
print(df.duplicated().sum())
# REMOVE DUPLICATES
df = df.drop_duplicates()
print("Shape After Removing Duplicates")
print(df.shape)
# REMOVE OUTLIERS USING IQR
for column in numeric_columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[
        (df[column] >= lower) &
        (df[column] <= upper)
    ]
print("Shape After Removing Outliers")
print(df.shape)
# LABEL ENCODING
categorical_columns = df.select_dtypes(
    include=["object"]
).columns
encoder = LabelEncoder()
for column in categorical_columns:
    df[column] = encoder.fit_transform(
        df[column]
    )
print("Categorical Encoding Completed")
# CORRELATION
print("Correlation Matrix")
print(df.corr())
# SAVE CLEANED DATASET
df.to_csv(
    "../data/cleaned_housing.csv",
    index=False
)
print("Cleaned Dataset Saved Successfully")