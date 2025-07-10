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
from sklearn.preprocessing import LabelEncoder
import scipy.cluster.hierarchy as sch
from scipy.special import logsumexp
from sklearn.model_selection import train_test_split
from collections import defaultdict
from datetime import timedelta
import scipy.spatial.distance as ssd
import json
import itertools

# 1. Train an HMM for Each User Sequence
def apply_data_based_correction(start_probs, alpha=0.01):
    """
    Adjusts zero initial state probabilities with alpha, then normalizes the distribution.
    """
    zero_indices = np.where(start_probs == 0)[0]
    nonzero_indices = np.where(start_probs > 0)[0]

    if len(zero_indices) > 0:
        nonzero_sum = np.sum(start_probs[nonzero_indices])
        start_probs[zero_indices] = alpha
        start_probs[nonzero_indices] = (start_probs[nonzero_indices] / nonzero_sum) * (1 - len(zero_indices) * alpha)
        start_probs /= np.sum(start_probs)

    return start_probs

def correct_transition_matrix(transmat, alpha=0.01):
    """
    Normalizes each row; replaces all-zero rows with uniform distribution.
    """
    corrected = transmat.copy()
    for i in range(corrected.shape[0]):
        if np.sum(corrected[i]) == 0:
            corrected[i] = np.full(corrected.shape[1], 1.0 / corrected.shape[1])
        else:
            corrected[i] = corrected[i] / np.sum(corrected[i])
    return corrected

def train_individual_hmms(patient_sequences, num_states=4, num_iter=100, random_state=42, laplace_alpha=0.001):
    """
    Trains one HMM per user using unsupervised learning with K-Means-based initialization.

    Returns:
    - trained_hmms: Dictionary mapping user ID to trained HMM model.
    """
    trained_hmms = {}

    for patient, sequence in patient_sequences.items():
        if sequence.shape[0] < num_states:
            print(f"Skipping user {patient} due to insufficient data.")
            continue

        try:
            kmeans = KMeans(n_clusters=num_states, n_init=10, init='k-means++', random_state=random_state)
            labels = kmeans.fit_predict(sequence)
            cluster_means = kmeans.cluster_centers_
        except ValueError:
            print(f"K-means failed for user {patient}. Using mean-based initialization.")
            cluster_means = np.mean(sequence, axis=0, keepdims=True) + np.random.randn(num_states, sequence.shape[1]) * 0.01
            labels = np.zeros(sequence.shape[0], dtype=int)

        cluster_covariances = np.zeros((num_states, sequence.shape[1]))
        for i in range(num_states):
            cluster_data = sequence[labels == i]
            if cluster_data.shape[0] > 1:
                cluster_covariances[i] = np.var(cluster_data, axis=0)
            else:
                cluster_covariances[i] = np.var(sequence, axis=0)

        cluster_covariances = np.clip(cluster_covariances, 1e-2, None)

        model = hmm.GaussianHMM(n_components=num_states, covariance_type="diag",
                                n_iter=num_iter, init_params='', random_state=random_state)
        model.startprob_ = np.full(num_states, 1 / num_states)
        model.transmat_ = np.full((num_states, num_states), 1 / num_states)
        model.means_ = cluster_means
        model.covars_ = cluster_covariances
        model.fit(sequence)

        model.startprob_ = apply_data_based_correction(model.startprob_, alpha=laplace_alpha)
        model.transmat_ = correct_transition_matrix(model.transmat_, alpha=laplace_alpha)

        trained_hmms[patient] = model

    return trained_hmms

# 2. Compute Log-Likelihood Matrix and Perform Clustering
def compute_log_likelihood_matrix(trained_hmms, patient_sequences):
    """
    Computes the pairwise log-likelihood matrix L_ij = log P(S_j | M_i),
    where M_i is the model of user i and S_j is the sequence of user j.

    Returns:
    - log_likelihood_matrix: N x N matrix
    - patient_list: Ordered list of user IDs
    """
    patient_list = list(patient_sequences.keys())
    num_patients = len(patient_list)
    log_likelihood_matrix = np.zeros((num_patients, num_patients))

    for i, patient_i in enumerate(patient_list):
        model_i = trained_hmms[patient_i]
        for j, patient_j in enumerate(patient_list):
            sequence_j = patient_sequences[patient_j]
            log_likelihood_matrix[i, j] = model_i.score(sequence_j)

    return log_likelihood_matrix, patient_list

def compute_symmetric_distance(log_likelihood_matrix):
    """
    Converts the log-likelihood matrix into a symmetric distance matrix using:
    D_ij = -0.5 * (L_ij + L_ji)

    Returns:
    - distance_matrix: Symmetric distance matrix
    """
    num_patients = log_likelihood_matrix.shape[0]
    distance_matrix = np.zeros((num_patients, num_patients))

    for i in range(num_patients):
        for j in range(num_patients):
            distance_matrix[i, j] = - 0.5 * (log_likelihood_matrix[i, j] + log_likelihood_matrix[j, i])
    distance_matrix -= np.min(distance_matrix)
    np.fill_diagonal(distance_matrix, 0)
    return distance_matrix

def cluster_users_using_distance(distance_matrix, patient_list, num_clusters=2):
    """
    Performs hierarchical clustering using the symmetric distance matrix.

    Returns:
    - cluster_labels: Cluster label array
    - user_clusters: Mapping from user ID to cluster ID
    - patient_mapping: Mapping from dendrogram index to user ID
    """
    distance_matrix = ssd.squareform(distance_matrix)
    linkage_matrix = linkage(distance_matrix, method='complete')
    max_d = linkage_matrix[-(num_clusters-1), 2]
    
    patient_mapping = {i + 1: patient_list[i] for i in range(len(patient_list))}
    labeled_patient_list = list(patient_mapping.keys())
    
    plt.figure(figsize=(8, 5))
    dendro = sch.dendrogram(linkage_matrix, labels=labeled_patient_list, color_threshold=max_d)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Samples (Indexed)")
    plt.ylabel("Distance")
    plt.show()

    cluster_labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
    user_clusters = {patient_list[i]: cluster_labels[i] for i in range(len(patient_list))}
    
    return cluster_labels, user_clusters, patient_mapping

def save_user_clusters(user_clusters):
    """
    Save user-to-cluster assignment information.

    Returns:
    - df_user_clusters: DataFrame of user cluster assignments
    """
    df_user_clusters = pd.DataFrame(user_clusters.items(), columns=["환자명", "cluster"])
    df_user_clusters["사용자번호"] = range(len(df_user_clusters))
    df_user_clusters = df_user_clusters[["사용자번호", "환자명", "cluster"]]

    return df_user_clusters

# 3. Train Composite HMMs for Each Cluster
def train_composite_hmms(user_clusters, patient_sequences, num_states=4, num_iter=100, random_state=42, laplace_alpha=0.01):
    """
    Trains a composite HMM for each cluster using all user sequences assigned to the cluster.

    Returns:
    - composite_hmms: Mapping from cluster ID to trained composite HMM
    """
    clusters = {}
    for user, cluster in user_clusters.items():
        clusters.setdefault(cluster, []).append(patient_sequences[user])

    composite_hmms = {}
    for cluster, sequences in clusters.items():
        all_sequences = np.concatenate(sequences)
        lengths = [len(seq) for seq in sequences]

        if all_sequences.shape[0] < num_states:
            print(f"Skipping cluster {cluster} due to insufficient data.")
            continue

        try:
            kmeans = KMeans(n_clusters=num_states, n_init=10, init='k-means++', random_state=random_state)
            labels = kmeans.fit_predict(all_sequences)
            cluster_means = kmeans.cluster_centers_
        except ValueError:
            print(f"K-means failed for cluster {cluster}. Using mean-based initialization.")
            cluster_means = np.mean(all_sequences, axis=0, keepdims=True) + np.random.randn(num_states, all_sequences.shape[1]) * 0.01
            labels = np.zeros(all_sequences.shape[0], dtype=int)

        cluster_covariances = np.zeros((num_states, all_sequences.shape[1]))
        for i in range(num_states):
            cluster_data = all_sequences[labels == i]
            if cluster_data.shape[0] > 1:
                cluster_covariances[i] = np.var(cluster_data, axis=0)
            else:
                cluster_covariances[i] = np.var(all_sequences, axis=0)

        cluster_covariances = np.clip(cluster_covariances, 1e-2, None)

        model = hmm.GaussianHMM(n_components=num_states, covariance_type="diag",
                                n_iter=num_iter, init_params='', random_state=random_state)
        model.startprob_ = np.full(num_states, 1 / num_states)
        model.transmat_ = np.full((num_states, num_states), 1 / num_states)
        model.means_ = cluster_means
        model.covars_ = cluster_covariances
        model.fit(all_sequences,lengths=lengths)
        
        model.startprob_ = apply_data_based_correction(model.startprob_, alpha=laplace_alpha)
        model.transmat_ = correct_transition_matrix(model.transmat_, alpha=laplace_alpha)

        composite_hmms[cluster] = model
        
    return composite_hmms

# 4. Determine the Optimal Number of Clusters (𝐾)
def compute_aic_bic_extended(patient_sequences, k_values, num_states_values, num_iter=100):
    """
    Computes AIC and BIC scores for various combinations of (num_states, num_clusters),
    and identifies the optimal combination.

    Returns:
    - best_params_aic: Tuple of best (num_states, num_clusters) by AIC
    - best_params_bic: Tuple of best (num_states, num_clusters) by BIC
    - aic_matrix: AIC values as a 2D matrix
    - bic_matrix: BIC values as a 2D matrix
    - num_states_list: List of tested state sizes
    - k_list: List of tested cluster sizes
    """
    aic_matrix = np.zeros((len(num_states_values), len(k_values)))
    bic_matrix = np.zeros((len(num_states_values), len(k_values)))
    
    num_states_list = list(num_states_values)
    k_list = list(k_values)

    for i, num_states in enumerate(num_states_list):
        print(f"\n Testing num_states = {num_states}")
        trained_hmms = train_individual_hmms(patient_sequences, num_states=num_states, num_iter=num_iter)
        logL_matrix, patient_list = compute_log_likelihood_matrix(trained_hmms, patient_sequences)
        distance_matrix = compute_symmetric_distance(logL_matrix)

        for j, k in enumerate(k_list):
            print(f"  - Testing K = {k}")
            _, user_clusters, _ = cluster_users_using_distance(distance_matrix, patient_list, num_clusters=k)
            composite_hmms = train_composite_hmms(user_clusters, patient_sequences, num_states=num_states, num_iter=num_iter)

            total_log_likelihood = 0
            total_params = 0

            for cluster_id, model in composite_hmms.items():
                total_log_likelihood += model.score(np.vstack([patient_sequences[p] for p in user_clusters if user_clusters[p] == cluster_id]))
                N = model.n_components
                D = model.means_.shape[1]
                num_params = (N - 1) + N * (N - 1) + 2 * N * D # startprob + transmat + mean + var
                total_params += num_params

            num_samples = sum(seq.shape[0] for seq in patient_sequences.values())

            aic = 2 * total_params - 2 * total_log_likelihood
            bic = total_params * np.log(num_samples) - 2 * total_log_likelihood

            aic_matrix[i, j] = aic
            bic_matrix[i, j] = bic

            print(f"    K={k}, AIC={aic:.2f}, BIC={bic:.2f}")

    best_idx_aic = np.unravel_index(np.argmin(aic_matrix), aic_matrix.shape)
    best_idx_bic = np.unravel_index(np.argmin(bic_matrix), bic_matrix.shape)

    best_params_aic = (num_states_list[best_idx_aic[0]], k_list[best_idx_aic[1]])
    best_params_bic = (num_states_list[best_idx_bic[0]], k_list[best_idx_bic[1]])

    return best_params_aic, best_params_bic, aic_matrix, bic_matrix, num_states_list, k_list

def plot_heatmap(matrix, x_labels, y_labels, title, cmap="coolwarm"):
    """
    Plots a heatmap of the given matrix with labeled axes.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt=".1f", xticklabels=x_labels, yticklabels=y_labels, cmap=cmap, linewidths=0.5)
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Hidden States (num_states)")
    plt.title(title)
    plt.show()

# Run Full Pipeline
if __name__ == "__main__":
    k_values = range(2, 6)
    num_states_values = range(2, 5)

    best_params_aic, best_params_bic, aic_matrix, bic_matrix, num_states_list, k_list = compute_aic_bic_extended(
        patient_sequences, k_values, num_states_values, num_iter=100
    )

    plot_heatmap(aic_matrix, k_list, num_states_list, "AIC Scores for (num_states, K)")
    plot_heatmap(bic_matrix, k_list, num_states_list, "BIC Scores for (num_states, K)")

    print(f"Best (num_states, K) based on AIC: {best_params_aic}")
    print(f"Best (num_states, K) based on BIC: {best_params_bic}")

if __name__ == "__main__":
    # Set based on best AIC/BIC result
    num_states = 4
    num_clusters = 2

    trained_hmms = train_individual_hmms(patient_sequences, num_states=num_states)

    log_likelihood_matrix, patient_list = compute_log_likelihood_matrix(trained_hmms, patient_sequences)
    distance_matrix = compute_symmetric_distance(log_likelihood_matrix)

    cluster_labels, user_clusters, patient_mapping = cluster_users_using_distance(
        distance_matrix, patient_list, num_clusters=num_clusters
    )
    df_user_clusters = save_user_clusters(user_clusters)
    cluster_hmms = train_composite_hmms(user_clusters, patient_sequences, num_states=num_states, num_iter=100)