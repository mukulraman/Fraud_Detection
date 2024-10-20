import sqlite3
import pandas as pd
import os

# Define the path to the folder
folder_path = 'data'

# Create the folder if it doesn't exist
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Execute this program if only the output_data.csv is not there in data folder. Please ensure to download Database.db
# from 'https://drive.google.com/file/d/1lyblXaqd0LEaGHuiz9uFv93f7voUssMQ/view?usp=drive_link'. Place downloaded
# Database.db in the data folder of the Project Or you can download it anywhere and specify the entire path in file path
file_path = os.path.join(folder_path,'Database.db')

conn = sqlite3.connect(file_path)
cursor = conn.cursor()

cursor.execute("SELECT name FROM  sqlite_master WHERE type='table';")

tables = cursor.fetchall()

for table in tables:
    print(table[0])

conn.close()

# Connect to the SQLite database (Database.db file)
conn = sqlite3.connect(file_path)

# Execute the query and load the data into a pandas DataFrame
df = pd.read_sql_query('Select * from Fraud_detection', conn)

# Save the DataFrame to a CSV file
file_path = os.path.join(folder_path, 'output_data.csv')
df.to_csv(file_path, index=False)

print(f"Data has been successfully saved to {file_path}")