import numpy as np
import pandas as pd

# Load the dataset
url = 'https://drive.google.com/file/d/1-lhxHcH5YGPDg28rJuivfO8q27wvfiEl/view'
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
raw_data = pd.read_csv(path)

# Convert the 'date' column to datetime type
raw_data['date'] = pd.to_datetime(raw_data['date'])

# Select and create a copy of data for the year 2023
raw_data_2023 = raw_data[raw_data.date.dt.year == 2023].copy()

# Modify 'topic' column
raw_data_2023['topic_cnt'] = raw_data_2023['topic'].apply(lambda s: len(s.split(';')))
raw_data_2023['topic'] = raw_data_2023['topic'].str.split(';')

# Explode the 'topic' column
topic_exploded = raw_data_2023.explode('topic').reset_index(drop=True)

# Remove 'index' column
topic_exploded.drop(columns=['index'], inplace=True, errors='ignore')

# Group and count operations
df = (topic_exploded.groupby(['country', 'scv_id', 'source_system', 'topic']).size().reset_index(name='events'))

# Pivot operation
df_pivot = df.pivot_table(columns='source_system', index=['country', 'scv_id', 'topic'], fill_value=0).reset_index()

# Rename columns
df_pivot.columns = ['_'.join(map(str, col)) for col in df_pivot.columns]

# Check and modify the last character of the column names
df_pivot.columns = [col[:-1] if col.endswith('_') else col for col in df_pivot.columns]

# Create 'events' column
df_pivot['events'] = df_pivot['events_PX'] + df_pivot['events_activecampaign'] + df_pivot['events_catalogue']

# Detect outliers and determine threshold
user_percentiles = [25, 50, 75]  # Percentiles to be determined by the user
thresholds = np.percentile(df_pivot['events'], user_percentiles)

# Count per 'topic' basis
topics_count = df.groupby(['country', 'topic']).size().reset_index(name='topic_count')

# Get 'scv_id' values
ids = df_pivot['scv_id'].unique()

# Create Result DataFrame
result = (
    pd.DataFrame({'scv_id': ids})
    .merge(topics_count, how='cross')
    .merge(df_pivot, on=['scv_id', 'topic', 'country'], how='left')
    .fillna(0)
)

# Adjust data types
result = result.astype({
    'scv_id' : 'int32',
    'country' : 'category',
    'topic' : 'category',
    'events_PX': 'int16',
    'events_activecampaign': 'int16',
    'events_catalogue': 'int16',
    'events': 'int16',
})

# Create and constrain the 'affinity' column
max_threshold = thresholds[-1]
result['affinity'] = (result['events'] * 10 / max_threshold).clip(upper=10).astype('int16')

# Save the results
result_path = 'output/'
result_filename = 'affinity_score'
result.to_csv( result_path + result_filename + '.csv', index=False)
