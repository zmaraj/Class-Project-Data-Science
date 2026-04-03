import pandas as pd

# Load both files

red = pd.read_csv('winequality-red.csv', sep=';')

white = pd.read_csv('winequality-white.csv', sep=';')

# Add a type column to each

red['type'] = 'red'

white['type'] = 'white'

# Combine them into one dataframe

wine = pd.concat([red, white], ignore_index=True)

# Save as cleaned_wine.csv

wine.to_csv('cleaned_wine.csv', index=False)

print(wine.shape)  # should show (6497, 13)
