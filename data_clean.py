import pandas as pd

df = pd.read_csv('diamonds_test_set.csv')
# Add an index column starting from 1
df.insert(0, 'index', range(1, len(df) + 1))

# Save to a new CSV file
df.to_csv('diamonds_test_set_with_index.csv', index=False)

# Optionally, you can print the DataFrame to check
print(df)