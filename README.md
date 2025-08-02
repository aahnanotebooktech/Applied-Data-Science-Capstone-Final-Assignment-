import pandas as pd

# Load dataset
df = pd.read_csv('your_dataset.csv')

# Basic cleaning
df.dropna(inplace=True)  # Remove nulls
df = df.drop_duplicates()
df.columns = df.columns.str.strip().str.lower()

# Convert data types
df['date'] = pd.to_datetime(df['date_column'])

# Save cleaned version
df.to_csv('data/cleaned_data.csv', index=False)# Applied-Data-Science-Capstone-Final-Assignment-
Applied Data Science Capstone 
