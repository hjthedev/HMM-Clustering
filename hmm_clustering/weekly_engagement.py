import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load user cluster assignments and patient-level time-series data
cluster_df = sequence_df.merge(df_user_clusters[['환자명', 'cluster']], on="환자명", how="left")
cluster_df["작성일자"] = pd.to_datetime(cluster_df["작성일자"])
cluster_df["Day"] = cluster_df.groupby("환자명")["작성일자"].transform(lambda x: (x - x.min()).dt.days + 1)
cluster_df["Week"] = ((cluster_df["Day"] - 1) // 7) + 1

weekly_means = cluster_df.groupby(["Week", "cluster"])[['exercise', 'meal', 'medication', 'blood']].mean().reset_index()
weekly_stats = cluster_df.groupby(["Week", "cluster"])[['exercise', 'meal', 'medication', 'blood']].agg(['mean', 'std', 'count']).reset_index()
weekly_stats.columns = ['Week', 'cluster'] + [f'{col}_{stat}' for col in ['exercise', 'meal', 'medication', 'blood'] for stat in ['mean', 'std', 'count']]

# Calculate standard error (s.e.m.) (SEM = std / sqrt(n))
for col in ['exercise', 'meal', 'medication', 'blood']:
    weekly_stats[f'{col}_sem'] = weekly_stats[f'{col}_std'] / np.sqrt(weekly_stats[f'{col}_count'])

titles = ["Exercise", "Meal", "Medicine", "Blood"]
columns = ['exercise', 'meal', 'medication', 'blood']
highlight_weeks = [1, 3, 5, 7, 9]
cluster_labels = {
    1: "Sustained Engagers",
    2: "Sporadic Engagers",
}
cluster_colors = {
    1: "#2a9d8f",
    2: "#e76f51",
}

sns.set(style="whitegrid", font_scale=1.2)
min_val = weekly_means[columns].min().min()
max_val = weekly_means[columns].max().max()
y_margin = 0.1 * (max_val - min_val)
y_min = min_val - y_margin
y_max = max_val + y_margin

# Weekly engagement status
fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
for i, col in enumerate(columns):
    ax = axes[i // 2, i % 2]
    for cluster_id in weekly_stats["cluster"].unique():
        cluster_data = weekly_stats[weekly_stats["cluster"] == cluster_id]
        label = cluster_labels.get(cluster_id, f"Cluster {cluster_id}")
        color = cluster_colors.get(cluster_id, None)
        
        ax.fill_between(
            cluster_data["Week"],
            cluster_data[f'{col}_mean'] - cluster_data[f'{col}_sem'],
            cluster_data[f'{col}_mean'] + cluster_data[f'{col}_sem'],
            color=color,
            alpha=0.15,
            interpolate=True
        )
        
        ax.plot(
            cluster_data["Week"],
            cluster_data[f'{col}_mean'],
            marker="o",
            linestyle="-",
            linewidth=2.5,
            markersize=5,
            label=label,
            color=color,
        )
    
    for idx, week in enumerate(highlight_weeks):
        label = "Nurse Intervention" if idx == 0 else None
        ax.axvline(x=week, color='red', linestyle="--", linewidth=1, alpha=0.5, label=label)
    
    ax.set_title(titles[i], fontsize=20)
    ax.set_xlabel("Weeks (1–12)", fontsize=14)
    ax.set_ylabel("Avg Daily Records", fontsize=14)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(np.arange(1, 13, step=1))
    ax.tick_params(axis='both', labelsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout(h_pad=3, w_pad=2)
plt.suptitle("Weekly Engagement Patterns by Cluster (with Confidence Intervals)", fontsize=22)
plt.subplots_adjust(hspace=0.3, wspace=0.2)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()