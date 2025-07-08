import pandas as pd
from scipy.stats import shapiro, levene, f_oneway, ttest_ind, kruskal, mannwhitneyu

# Import user adherence logs
final_result = final_result.merge(cluster_df[['환자명', 'cluster']].drop_duplicates(),
                                            left_on='환자명',
                                            right_on='환자명',
                                            how='left')
final_result.drop(columns=['환자명'], inplace=True)
final_result = final_result.dropna(subset=['cluster'])
final_result = final_result.rename(columns={'cluster': 'Cluster'})

# Calculate individual-level adherence
## 1) 'medication intake' proportion
final_result['med_total'] = final_result['med_0'] + final_result['med_1']
final_result['med_1_proportion'] = final_result['med_1'] / final_result['med_total']

## 2) 'light' and 'normal' meal proportion
final_result['meal_total'] = final_result[['meal_0', 'meal_1', 'meal_2', 'meal_3']].sum(axis=1)
final_result['meal_1_2_proportion'] = (final_result['meal_1'] + final_result['meal_2']) / final_result['meal_total']

# 3) 'Pre/post-meal glucose' check proportion
final_result['blood_total'] = final_result[['blood_none', 'blood_before_only', 'blood_after_only', 'blood_all']].sum(axis=1)
final_result['blood_all_proportion'] = final_result['blood_all'] / final_result['blood_total']


# Perform assumption checks & choose appropriate statistical tests
assumption_tests = {}
stat_test_results = {}

def check_assumptions(metric, final_result):
    clusters = final_result['Cluster'].unique()
    groups = [final_result[final_result['Cluster'] == cluster][metric].dropna()
              for cluster in clusters]

    # Shapiro-Wilk Test for Normality (only for groups with > 3 samples)
    normality_pvals = [shapiro(group)[1] for group in groups if len(group) > 3]
    normality_result = all(pval > 0.05 for pval in normality_pvals)

    # Levene's Test for Equal Variance (only if multiple groups exist)
    if len(groups) > 1:
        levene_pval = levene(*groups)[1]
        equal_variance = levene_pval > 0.05
    else:
        equal_variance = True

    return {
        'Normality': normality_result,
        'Equal Variance': equal_variance,
        'Normality P-Values': normality_pvals,
        'Levene P-Value': levene_pval if len(groups) > 1 else None
    }

metrics_to_test = ['med_1_proportion',
    'exer_mean',
    'exer_count_30min',
    'meal_1_2_proportion',
    'blood_all_proportion'
]

for metric in metrics_to_test:
    assumptions = check_assumptions(metric, final_result)
    assumption_tests[metric] = assumptions

    clusters = final_result['Cluster'].unique()
    groups = [final_result[final_result['Cluster'] == cluster][metric].dropna()
              for cluster in clusters]

    # Decide test based on normality & variance results
    if len(clusters) == 2:
        if assumptions['Normality'] and assumptions['Equal Variance']:
            stat, pval = ttest_ind(groups[0], groups[1], equal_var=True)
            test_used = "t-test"
        else:
            stat, pval = mannwhitneyu(groups[0], groups[1], alternative='two-sided')
            test_used = "Mann-Whitney U"
    elif len(clusters) > 2:
        if assumptions['Normality'] and assumptions['Equal Variance']:
            stat, pval = f_oneway(*groups)
            test_used = "ANOVA"
        else:
            stat, pval = kruskal(*groups)
            test_used = "Kruskal-Wallis"
    else:
        stat, pval, test_used = None, None, "Not enough clusters"

    stat_test_results[metric] = {"Test": test_used, "Statistic": stat, "p-value": pval}