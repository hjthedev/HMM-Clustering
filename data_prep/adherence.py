# Load the user's mhealth app records after data preprocessing
filtered_record = filtered_record[filtered_record['환자명'].isin(patient_list)]

# 1. blood adherence
blood_filtered = filtered_record[filtered_record['구분'].str.contains('blood', na=False)].copy()
blood_filtered['blood_none'] = 0
blood_filtered['blood_all'] = 0
blood_filtered['blood_before_only'] = 0
blood_filtered['blood_after_only'] = 0

## Update the counts based on conditions
blood_filtered.loc[
    (pd.isna(blood_filtered['a']) | (blood_filtered['a'] == 0)) &
    (pd.isna(blood_filtered['n']) | (blood_filtered['n'] == 0)), 'blood_none'] = 1

blood_filtered.loc[
    (blood_filtered['a'].notna()) & (blood_filtered['a'] != 0) &
    (blood_filtered['n'].notna()) & (blood_filtered['n'] != 0), 'blood_all'] = 1

blood_filtered.loc[
    (blood_filtered['a'].notna()) & (blood_filtered['a'] != 0) &
    (pd.isna(blood_filtered['n']) | (blood_filtered['n'] == 0)), 'blood_before_only'] = 1

blood_filtered.loc[
    (pd.isna(blood_filtered['a']) | (blood_filtered['a'] == 0)) &
    (blood_filtered['n'].notna()) & (blood_filtered['n'] != 0), 'blood_after_only'] = 1

blood_summary = blood_filtered.groupby('환자명')[
    ['blood_none', 'blood_all', 'blood_before_only', 'blood_after_only']
].sum().reset_index()

# 2. medication adherence
med_counts = filtered_record[filtered_record['구분'] == 'medication'].groupby('환자명')['a'].value_counts().unstack(fill_value=0)
med_counts.reset_index(inplace=True)
med_counts.columns = ['환자명', 'med_0', 'med_1'] 

# 3. exercise adherence (min)
exer_data = filtered_record[filtered_record['구분'] == 'exercise']
Q1 = exer_data['a'].quantile(0.25)
Q3 = exer_data['a'].quantile(0.75)
IQR = Q3 - Q1
filtered_exer = exer_data[(exer_data['a'] >= Q1 - 1.5 * IQR) & (exer_data['a'] <= Q3 + 1.5 * IQR)]

exer_means = filtered_exer.groupby('환자명')['a'].mean().reset_index()
exer_means.columns = ['환자명', 'exer_mean']

# 4. exercise adherence (day)
exer_counts = filtered_record[(filtered_record['구분'] == 'exercise') & (filtered_record['a'] >= 30)].groupby('환자명')['a'].count().reset_index()
exer_counts.columns = ['환자명', 'exer_count_30min']

# 5. meal adherece (0: skipped, 1: light, 2: normal, 3:overeating)
meal_counts = filtered_record[filtered_record['구분'] == 'meal'].groupby('환자명')['a'].value_counts().unstack(fill_value=0)
meal_counts.reset_index(inplace=True)
meal_counts.columns = ['환자명', 'meal_0', 'meal_1', 'meal_2', 'meal_3']

final_result = med_counts\
    .merge(exer_means, on='환자명', how='outer') \
    .merge(exer_counts, on='환자명', how='outer')\
    .merge(meal_counts, on='환자명', how='outer') \
    .merge(blood_summary, on='환자명', how='outer')