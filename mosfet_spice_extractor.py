
import pandas as pd

# Let's read the newly uploaded final15_06.csv file to see its structure
df = pd.read_csv('final15_06.csv')
print("Columns:", df.columns.tolist())
print(df.head(3))