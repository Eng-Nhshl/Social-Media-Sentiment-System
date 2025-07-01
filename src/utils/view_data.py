import pandas as pd

# Read the dataset
df = pd.read_csv('data/social_media_data.csv')

# Display first few rows with proper encoding
print("\nFirst 5 rows of the dataset:")
for idx, row in df.head().iterrows():
    print(f"\nRow {idx + 1}:")
    print(f"Date: {row['date']}")
    print(f"Text: {row['text']}")
