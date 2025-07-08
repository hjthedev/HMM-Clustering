import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_cluster_hmms(cluster_hmms):
    # State your cluster name
    cluster_labels = {
        1: "Sustained Engagers",
        2: "Sporadic Engagers"
    }

    for cluster_id, model in cluster_hmms.items():
        # Order states by descending sum of their means
        state_order = np.argsort(model.means_.sum(axis=1))[::-1]
        
        sorted_transmat = model.transmat_[state_order][:, state_order]
        sorted_means = model.means_[state_order]
        sorted_covars = model.covars_[state_order]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        sns.heatmap(sorted_transmat, annot=True, cmap="Blues", fmt=".2f", ax=axes[0])
        axes[0].set_title(f"{cluster_labels.get(cluster_id, f'Cluster {cluster_id}')} - Transition Matrix")
        axes[0].set_xlabel("To State")
        axes[0].set_ylabel("From State")

        sns.heatmap(sorted_means, annot=True, cmap="Greens", fmt=".2f", ax=axes[1])
        axes[1].set_title(f"{cluster_labels.get(cluster_id, f'Cluster {cluster_id}')} - State Mean")
        axes[1].set_xlabel("Feature Index")
        axes[1].set_ylabel("State")

        diag_covars = np.array([np.diag(cov) for cov in sorted_covars])
        sns.heatmap(diag_covars, annot=True, cmap="Reds", fmt=".2e", ax=axes[2])
        axes[2].set_title(f"{cluster_labels.get(cluster_id, f'Cluster {cluster_id}')} - State Covariance")
        axes[2].set_xlabel("Feature Index")
        axes[2].set_ylabel("State")

        plt.tight_layout()
        plt.show()

