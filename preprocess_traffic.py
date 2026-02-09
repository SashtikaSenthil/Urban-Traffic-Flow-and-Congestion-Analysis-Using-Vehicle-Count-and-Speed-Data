import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# -----------------------------
# 1. LOAD DATASET
# -----------------------------
file_path = "traffic_dataset.csv"   # <-- keep dataset in same folder
df = pd.read_csv(file_path)

print("Original Shape:", df.shape)
print(df.head())

# -----------------------------
# 2. BASIC CLEANING
# -----------------------------

# remove duplicates
df = df.drop_duplicates()

# standardize column names
df.columns = df.columns.str.lower().str.strip()

# -----------------------------
# 3. HANDLE MISSING VALUES
# -----------------------------

# numeric → fill with median
num_cols = df.select_dtypes(include=np.number).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# categorical → fill with mode
cat_cols = df.select_dtypes(include="object").columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# -----------------------------
# 4. DATE FEATURE ENGINEERING
# -----------------------------

df["date"] = pd.to_datetime(df["date"])

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["hour"] = df["date"].dt.hour

df = df.drop(columns=["date"])

# -----------------------------
# 5. OUTLIER HANDLING (IQR)
# -----------------------------

for col in ["vehicle_count", "avg_speed_kmph"]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df[col] = np.where(df[col] < lower, lower, df[col])
    df[col] = np.where(df[col] > upper, upper, df[col])

# -----------------------------
# 6. ENCODE CATEGORICAL DATA
# -----------------------------

encoder = LabelEncoder()

for col in ["city", "area_type", "road_type", "congestion_level", "weather"]:
    df[col] = encoder.fit_transform(df[col])

# -----------------------------
# 7. FEATURE SCALING
# -----------------------------

scaler = StandardScaler()
scale_cols = ["vehicle_count", "avg_speed_kmph", "accidents_reported"]

df[scale_cols] = scaler.fit_transform(df[scale_cols])

# -----------------------------
# 8. SAVE CLEAN DATASET
# -----------------------------

output_file = "traffic_cleaned.csv"
df.to_csv(output_file, index=False)

print("\nPreprocessing Completed ✅")
print("Clean Shape:", df.shape)
print("Saved as:", output_file)
print(df.head())
