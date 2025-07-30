import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# Load dataset
data_path = "Employers_data.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found: {data_path}")

df = pd.read_csv(data_path)  # ✅ First: load data
df.columns = df.columns.str.strip()  # Clean column names

print("Original columns:", df.columns.tolist())

# ✅ Now it's safe to rename columns
df = df.rename(columns={
    "Experience_Years": "Experience",
    "Education_Level": "Education",
    "Job_Title": "Role"
})

# Drop unused columns (optional)
df = df.drop(columns=["Employee_ID", "Name", "Age"], errors='ignore')

# Check for required columns
required_columns = ["Experience", "Education", "Role", "Department", "Location", "Gender", "Salary"]
missing = [col for col in required_columns if col not in df.columns]
if missing:
    raise ValueError(f"❌ Missing required columns: {missing}")

# Split features and target
X = df.drop("Salary", axis=1)
y = df["Salary"]

# Define categorical columns (include Gender now)
categorical_features = ["Education", "Role", "Department", "Location", "Gender"]

# Preprocessor
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
], remainder="passthrough")

# Model pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.joblib")
print("✅ Model trained and saved as model.joblib")
