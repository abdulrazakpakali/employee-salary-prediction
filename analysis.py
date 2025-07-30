import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Employers_data.csv")

# Show first 5 rows
print(df.head())

# Show missing values
print("\nMissing values:\n", df.isnull().sum())

# Show summary statistics
print("\nStatistics:\n", df.describe())

# Show column names and data types
print("\nInfo:")
print(df.info())