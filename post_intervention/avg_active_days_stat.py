import pandas as pd
from scipy.stats import shapiro, levene, f_oneway, ttest_ind, kruskal, mannwhitneyu

# Weekly statistical test between clusters
def run_weekly_tests(df):
    results = []

    for week in sorted(df['week'].unique()):
        temp = df[df['week'] == week]
        clusters = temp['cluster'].unique()

        if len(clusters) != 2:
            continue 

        vals1 = temp[temp['cluster'] == clusters[0]]['days_active'].dropna()
        vals2 = temp[temp['cluster'] == clusters[1]]['days_active'].dropna()
        n1, n2 = len(vals1), len(vals2)

        if n1 < 3 or n2 < 3:
            results.append({
                'week': week,
                'cluster_1': clusters[0],
                'cluster_2': clusters[1],
                'n_1': n1,
                'n_2': n2,
                'test_used': 'Insufficient Data',
                'p_value': None,
                'u_stat': None,
                'all_normal': False,
                'equal_var': False,
                'is_significant': None
            })
            continue

        # Normality test
        p_norm1 = shapiro(vals1).pvalue
        p_norm2 = shapiro(vals2).pvalue
        all_normal = (p_norm1 > 0.05) and (p_norm2 > 0.05)

        # Variance equality test
        p_levene = levene(vals1, vals2).pvalue
        equal_var = p_levene > 0.05

        if all_normal and equal_var:
            test_used = 't-test'
            stat = ttest_ind(vals1, vals2, equal_var=True)
            p_val = stat.pvalue
            u_stat = None
        else:
            test_used = 'Mann-Whitney'
            u_test = mannwhitneyu(vals1, vals2, alternative='two-sided')
            p_val = u_test.pvalue
            u_stat = u_test.statistic

        results.append({
            'week': week,
            'cluster_1': clusters[0],
            'cluster_2': clusters[1],
            'n_1': n1,
            'n_2': n2,
            'test_used': test_used,
            'p_value': round(p_val, 4),
            'u_stat': round(u_stat, 2) if u_stat is not None else None,
            'all_normal': all_normal,
            'equal_var': equal_var,
            'is_significant': p_val < 0.05
        })

    return pd.DataFrame(results)