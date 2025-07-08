import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import ttest_ind
from scipy.stats import pearsonr
from hmmlearn import hmm
from hmmlearn.hmm import GaussianHMM
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
import scipy.cluster.hierarchy as sch
from scipy.special import logsumexp
from sklearn.model_selection import train_test_split
from collections import defaultdict
import scipy.spatial.distance as ssd
from datetime import timedelta

# Set your directory
base_path = r"D:\anaconda3\Diabetes"
file_names = ["patients_record", "survey(first)", "survey(second)", "phone calls", "height-weight"]

dataframes = {}

# Load data
for file_name in file_names:
    file_path = f"{base_path}\\{file_name}.csv"
    try:
        dataframes[file_name] = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error: {e}")

record = dataframes["patients_record"]
first_survey = dataframes['survey(first)']
second_survey = dataframes['survey(second)']
calls = dataframes['phone calls']
hw = dataframes['height-weight']

record['환자명'] = record['환자명'].str.replace(r'\s+', '', regex=True)
first_survey['성명'] = first_survey['성명'].str.replace(r'\s+', '', regex=True)
second_survey['성명'] = second_survey['성명'].str.replace(r'\s+', '', regex=True)
calls['이름'] = calls['이름'].str.replace(r'\s+', '', regex=True)

# 1. Remove non-respondents
first_0 = first_survey[first_survey['실험'] == 0]
first_1 = first_survey[first_survey['실험'] == 1]

second_0 = second_survey[second_survey['실험군'] == 0]
second_1 = second_survey[second_survey['실험군'] == 1]

nan_rows_names = second_survey[second_survey['실험군'].isnull()]['성명']
diaNote_names = second_1[second_1['실험군'].notnull()]['성명']

names_to_drop = nan_rows_names.tolist()

first_0 = first_0[~first_0['성명'].isin(names_to_drop)]
first_1 = first_1[~first_1['성명'].isin(names_to_drop)]

record_unique_names = set(record['환자명'].unique())
first_survey_unique_names = set(first_1['성명'].unique())

names_only_in_record = record_unique_names - first_survey_unique_names
names_only_in_first_survey = first_survey_unique_names - record_unique_names

difference_names = {
    "only_in_record": list(names_only_in_record),
    "only_in_first_survey": list(names_only_in_first_survey)
}

record = record[record['환자명'].isin(first_survey['성명'])].reset_index(drop=True)
record = record[~record['환자명'].isin(names_to_drop)]

record['작성일자'] = pd.to_datetime(record[['연', '월', '일']].astype(str).agg('-'.join, axis=1))
record_cleaned = record.drop(columns=['연', '월', '일'])

# 2. Time range filtering
calls.loc[:34, '가입일'] = '2023-04-10'
calls.loc[35, '가입일'] = '2023-04-06'
calls.loc[36, '가입일'] = '2023-04-05'
calls.loc[37:, '가입일'] = '2023-04-04'

calls = calls[calls['이름'].isin(diaNote_names)].reset_index(drop=True)
calls['가입일'] = pd.to_datetime(calls['가입일'])
calls['실험종료일'] = calls['가입일'] + pd.to_timedelta(12, unit='W') - pd.Timedelta(days=1)

record_cleaned['작성일자'] = pd.to_datetime(record_cleaned['작성일자'])
calls['실험종료일'] = pd.to_datetime(calls['실험종료일'])
calls['가입일'] = pd.to_datetime(calls['가입일'])

filtered_records = []

for name in calls['이름'].unique():
    call_entry = calls[calls['이름'] == name].iloc[0]
    start_date = call_entry['가입일']
    end_date = call_entry['실험종료일']

    records_for_name = record_cleaned[
        (record_cleaned['작성일자'] >= start_date) &
        (record_cleaned['작성일자'] <= end_date) &
        (record_cleaned['환자명'] == name)
    ]
    filtered_records.append(records_for_name)

filtered_record = pd.concat(filtered_records, ignore_index=True)

calls = calls.rename(columns={'이름': '환자명'})
filtered_record = filtered_record.merge(
    calls[['환자명', '가입일', '실험종료일']], 
    on='환자명', 
    how='left'
)

# 3. Post-intervention period
after_records = []

for name in calls['환자명'].unique():
    call_entry = calls[calls['환자명'] == name].iloc[0]
    end_date = call_entry['실험종료일']
    end_date_plus_4w = end_date + timedelta(days=28)

    after_data = record_cleaned[
        (record_cleaned['환자명'] == name) &
        (record_cleaned['작성일자'] > end_date) &
        (record_cleaned['작성일자'] <= end_date_plus_4w)
    ]
    
    if not after_data.empty:
        after_records.append(after_data)
        
after_records_df = pd.concat(after_records, ignore_index=True)
after_records_df['end_plus_4w'] = after_records_df['환자명'].map(
    lambda name: calls.loc[calls['환자명'] == name, '실험종료일'].iloc[0] + timedelta(days=28)
)

after_records_df = after_records_df.merge(
    calls[['환자명', '실험종료일']],
    on='환자명',
    how='left'
)

# 4.1 Daily activity aggregation (12 week)
filtered_record['작성일자'] = pd.to_datetime(filtered_record['작성일자'])
patients = filtered_record['환자명'].unique()
data = []

for patient in patients:
    patient_data = filtered_record[filtered_record['환자명'] == patient]
    start_date = patient_data['가입일'].iloc[0]
    end_date = patient_data['실험종료일'].iloc[0]
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    patient_df = pd.DataFrame({'작성일자': date_range, '환자명': patient})

    for category in ['exercise', 'meal', 'medication', 'blood']:
        patient_df[category] = 0

    for _, row in patient_data.iterrows():
        date = row['작성일자']
        category = row['구분']

        if category in ['exercise', 'meal', 'medication', 'blood']:
            patient_df.loc[patient_df['작성일자'] == date, category] += 1
    data.append(patient_df)

sequence_df = pd.concat(data, ignore_index=True)

# 4.2 Daily activity aggregation (+4 week)
after_records_df['작성일자'] = pd.to_datetime(after_records_df['작성일자'])
after_records_df['실험종료일'] = pd.to_datetime(after_records_df['실험종료일'])
after_records_df['end_plus_4w'] = pd.to_datetime(after_records_df['end_plus_4w'])

filtered_after_records = after_records_df[
    (after_records_df['작성일자'] >= after_records_df['실험종료일'] + timedelta(days=1)) &
    (after_records_df['작성일자'] <= after_records_df['end_plus_4w'])
].copy()

categories = ['exercise', 'meal', 'medication', 'blood']
data = []

for patient in filtered_after_records['환자명'].unique():
    patient_data = filtered_after_records[filtered_after_records['환자명'] == patient]
    start_date = patient_data['실험종료일'].iloc[0] + timedelta(days=1)
    end_date = patient_data['end_plus_4w'].iloc[0]
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    patient_df = pd.DataFrame({'작성일자': date_range, '환자명': patient})
    for cat in categories:
        patient_df[cat] = 0

    for _, row in patient_data.iterrows():
        date = row['작성일자']
        cat = row['구분']
        if cat in categories:
            patient_df.loc[patient_df['작성일자'] == date, cat] += 1

    data.append(patient_df)

sequence_df_after = pd.concat(data, ignore_index=True)

# 5. Data cleaning
meal_outliers = sequence_df[sequence_df["meal"] >= 4]
med_outliers = sequence_df[sequence_df["medication"] >= 4]

unique_patient = meal_outliers["환자명"].unique()
unique_yakpatient = med_outliers["환자명"].unique()

sequence_df = sequence_df[~sequence_df["환자명"].isin(unique_patient)]
sequence_df = sequence_df[~sequence_df["환자명"].isin(unique_yakpatient)]

# 6.1 Patient-level time series generation (12 week)
df = sequence_df.copy()
df['작성일자'] = pd.to_datetime(df['작성일자'])
df = df.sort_values(by=['환자명', '작성일자'])

multivariate_cols = ['exercise', 'meal', 'medication', 'blood']
patient_sequences = {}

for patient, group in df.groupby('환자명'):
    sequence = group[multivariate_cols].values
    patient_sequences[patient] = sequence

# 6.2 Patient-level time series generation (+4 week)
df_filtered = sequence_df_after.copy()
df_filtered['작성일자'] = pd.to_datetime(df_filtered['작성일자'])
df_filtered = df_filtered.sort_values(by=['환자명', '작성일자'])
patient_sequences_after = {}

for patient, group in df_filtered.groupby('환자명'):
    sequence = group[multivariate_cols].values
    patient_sequences_after[patient] = sequence