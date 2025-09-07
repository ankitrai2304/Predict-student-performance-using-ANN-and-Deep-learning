import pandas as pd

# URL for the math student performance dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/student/student-mat.csv'

# Read the dataset directly into a pandas DataFrame
# Make sure to specify the separator is a semicolon ';'
df = pd.read_csv(url, sep=';')

# Display the first 5 rows to check if it loaded correctly
print(df.head())