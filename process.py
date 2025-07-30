import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("6a0a41a0-6858-4089-8d2c-d5e8158958de.csv")

# Fill missing values if any
df.fillna(method='ffill', inplace=True)

# Define features and target
X = df.drop("Salary", axis=1)  # Use your actual column name
y = df["Salary"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Train and Test sets created.")