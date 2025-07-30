import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

np.random.seed(42)  # For reproducibility

# Define possible categories
education_levels = ['Bachelor', 'Master', 'PhD']
job_roles = ['Developer', 'Data Scientist', 'Manager', 'Tester', 'Designer',
             'DevOps Engineer', 'Analyst', 'Project Manager', 'ML Engineer', 'QA Engineer']

# Generate 100 samples
n_samples = 100

data = {
    'Education': np.random.choice(education_levels, n_samples),
    'Age': np.random.randint(22, 60, n_samples),
    'Experience': np.random.randint(0, 30, n_samples),
    'JobRole': np.random.choice(job_roles, n_samples)
}

df = pd.DataFrame(data)

# Assign numeric values to education to calculate salary
edu_map = {'Bachelor': 1, 'Master': 2, 'PhD': 3}
df['EduNum'] = df['Education'].map(edu_map)

# Assign numeric values to job role to calculate salary influence
job_map = {job: i for i, job in enumerate(job_roles, 1)}
df['JobNum'] = df['JobRole'].map(job_map)

# Generate salary as a function of features + noise
df['Salary'] = (
    30000 + 
    df['EduNum'] * 8000 + 
    df['Age'] * 500 + 
    df['Experience'] * 1000 + 
    df['JobNum'] * 2000 + 
    np.random.normal(0, 5000, n_samples)  # noise
).round(0)

# Drop helper columns before training
df_train = df.drop(['EduNum', 'JobNum'], axis=1)

# Encode categorical features
le_edu = LabelEncoder()
le_job = LabelEncoder()

df_train['Education'] = le_edu.fit_transform(df_train['Education'])
df_train['JobRole'] = le_job.fit_transform(df_train['JobRole'])

# Define features and target
X = df_train[['Education', 'Age', 'Experience', 'JobRole']]
y = df_train['Salary']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")

# Save model and encoders
joblib.dump(model, 'model.pkl')
joblib.dump(le_edu, 'edu_encoder.pkl')
joblib.dump(le_job, 'job_encoder.pkl')

print("✅ Model and encoders saved successfully!")