from collections import Counter
import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import chi2_contingency

# 0. Count state-to-state transitions using state names
def get_named_transition_counts(model, sequences, label_map):
    """
    Predicts HMM state sequences and returns a transition count matrix
    where transitions are labeled by descriptive state names.
    
    Parameters:
    - model: trained HMM model
    - sequences: list of observation sequences
    - label_map: mapping from state index to descriptive state name

    Returns:
    - transition count DataFrame (state_name x state_name)
    """
    total_transitions = Counter()
    for seq in sequences:
        states = model.predict(seq)
        transitions = zip(states[:-1], states[1:])
        total_transitions.update(transitions)
    matrix = pd.DataFrame(0, index=state_order, columns=state_order, dtype=int)
    for (s_from, s_to), count in total_transitions.items():
        name_from = label_map.get(s_from)
        name_to = label_map.get(s_to)
        if name_from in matrix.index and name_to in matrix.columns:
            matrix.loc[name_from, name_to] += count
    return matrix

## Define consistent state name order
state_order = ["Well-managed", "Non-invasive Care", "Skewed Adherence", "Unmanaged"]

## Map cluster-specific HMM state indices to descriptive names
cluster_labels = {
    1: {2: "Non-invasive Care", 1: "Well-managed", 0: "Skewed Adherence", 3: "Unmanaged"},
    2: {1: "Skewed Adherence", 0: "Well-managed", 1: "Unmanaged", 3: "Non-invasive Care"},
}

## Load trained HMMs for each cluster and map patient sequences to clusters
model_1 = cluster_hmms[1]
model_2 = cluster_hmms[2]

df_user_clusters_named = df_user_clusters.set_index("환자명")
cluster_1_sequences = []
cluster_2_sequences = []

for user, seq in patient_sequences.items():
    if user not in df_user_clusters_named.index:
        continue
    cluster = df_user_clusters_named.loc[user, "cluster"]
    if cluster == 1:
        cluster_1_sequences.append(seq)
    elif cluster == 2:
        cluster_2_sequences.append(seq)

## Compute transition count matrices for each cluster
trans_counts_1 = get_named_transition_counts(model_1, cluster_1_sequences, cluster_labels[1])
trans_counts_2 = get_named_transition_counts(model_2, cluster_2_sequences, cluster_labels[2])

# 1. Perform z-tests for each state transition between clusters
results = []
for from_state in state_order:
    for to_state in state_order:
        count1 = trans_counts_1.loc[from_state, to_state]
        total1 = trans_counts_1.loc[from_state].sum()
        count2 = trans_counts_2.loc[from_state, to_state]
        total2 = trans_counts_2.loc[from_state].sum()

        if total1 == 0 or total2 == 0:
            zval, pval = np.nan, np.nan
        else:
            try:
                zval, pval = proportions_ztest([count1, count2], [total1, total2])
            except:
                zval, pval = np.nan, np.nan

        results.append({
            "From": from_state,
            "To": to_state,
            "Count1": count1,
            "Total1": total1,
            "Count2": count2,
            "Total2": total2,
            "Z": zval,
            "p-value": pval
        })
results_df = pd.DataFrame(results)

## Apply Bonferroni correction for multiple comparisons
n_tests = results_df["p-value"].notna().sum()
results_df["p_bonf"] = results_df["p-value"] * n_tests
results_df["p_bonf"] = results_df["p_bonf"].clip(upper=1.0)
results_df["Significant (Bonferroni)"] = results_df["p_bonf"].apply(
    lambda p: "✔ Significant" if pd.notna(p) and p < 0.05 else "✘ Not significant"
)
results_df["p_bonf"] = results_df["p_bonf"].round(2)

# 2. Perform chi-square test on the entire transition matrix
combined_counts = []
for from_state in state_order:
    for to_state in state_order:
        count1 = trans_counts_1.loc[from_state, to_state]
        count2 = trans_counts_2.loc[from_state, to_state]
        combined_counts.append([count1, count2])

contingency_table = np.array(combined_counts).T
chi2, pval, dof, expected = chi2_contingency(contingency_table)

chi2_result = pd.DataFrame({
    "Chi2 Statistic": [chi2],
    "Degrees of Freedom": [dof],
    "p-value": [pval],
    "Significant": ["✔ Significant" if pval < 0.05 else "✘ Not significant"]
})