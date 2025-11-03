from google.cloud import bigquery
import json
import os

# Load GCP project ID from credentials file
with open("config/credentials.json") as f:
    credentials = json.load(f)
    project_id = credentials.get("project_id")

# Create client
client = bigquery.Client(project=project_id)
 
# SQL Query
query = credentials.get("query")
df = client.query(query).to_dataframe()
 
print(df.head())