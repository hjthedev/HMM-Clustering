# Load post-intervention mHealth log data
sequence_df_after.loc[:, '작성일자'] = pd.to_datetime(sequence_df_after['작성일자'])
sequence_df_after['day'] = sequence_df_after.sort_values(['환자명', '작성일자']) \
    .groupby('환자명').cumcount() + 1
sequence_df_after['week'] = ((sequence_df_after['day'] - 1) // 7) + 13

for col in ['exercise', 'meal', 'medication', 'blood']:
    sequence_df_after[col] = sequence_df_after[col].fillna(0)

sequence_df_after['total_action'] = sequence_df_after[['exercise', 'meal', 'medication', 'blood']].sum(axis=1)

weeks = pd.DataFrame({'week': sorted(sequence_df_after['week'].dropna().unique())})
users_df = df_user_clusters[['환자명', 'cluster']]
user_week_combinations = sequence_df_after[['환자명', 'week']].drop_duplicates()
user_week_combinations = user_week_combinations.merge(df_user_clusters[['환자명', 'cluster']], on='환자명', how='left')

active_logs = sequence_df_after[sequence_df_after['total_action'] > 0]
days_active = active_logs.groupby(['week', '환자명'])['작성일자'] \
    .nunique().reset_index(name='days_active')

behavior_sum = sequence_df_after.groupby(['week', '환자명']) \
    [['exercise', 'meal', 'medication', 'blood']].mean().reset_index()

user_weekly = user_week_combinations \
    .merge(days_active, on=['week', '환자명'], how='left') \
    .merge(behavior_sum, on=['week', '환자명'], how='left')

user_weekly['days_active'] = user_weekly['days_active'].fillna(0)
user_weekly[['exercise', 'meal', 'medication', 'blood']] = user_weekly[['exercise', 'meal', 'medication', 'blood']].fillna(0)

final_df = user_weekly.groupby(['week', 'cluster']).agg(
    days_active_mean=('days_active', 'mean'),
    exercise_mean=('exercise', 'mean'),
    meal_mean=('meal', 'mean'),
    medication_mean=('medication', 'mean'),
    blood_mean=('blood', 'mean'),
    n_users=('환자명', 'nunique')
).reset_index().round(2)