import pandas as pd
'''
pip install pandas pyarrow
# or
pip install pandas fastparquet
'''

# Step 2: Read the Parquet file
df = pd.read_parquet(
    'path_to_file/hate_sp.parquet')

# Step 3: Write to CSV
df.to_csv('path_to_file/hate_sp.csv', index=False)
