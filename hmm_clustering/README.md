# Hidden Markov Model (HMM) Clustering Algorithm
Building on Smyth’s HMM clustering approach [[link](https://papers.nips.cc/paper_files/paper/1996/hash/6a61d423d02a1c56250dc23ae7ff12f3-Abstract.html)], we extend it by learning cluster-specific HMMs tailored for multivariate mHealth time-series data, enhancing personalization and interpretability. We jointly optimize the number of clusters (K) and hidden states (m) using AIC and BIC criteria.

## 1. Train an HMM for Each User Sequence
 - Independently train an HMM with m states for each of the N user sequences.
 - Initialize the HMM parameters (means and covariances) using k-means clustering.
 - Initialize initial state and transition probabilities uniformly.
 - Optimize the HMM parameters using the Baum-Welch algorithm.
 
## 2. Compute Log-Likelihood Matrix and Perform Clustering
 - For all N sequences, compute the log-likelihood of each sequence under every other sequence's HMM (resulting in an N × N matrix)
 - Use the log-likelihood values to compute similarity distances
 - Perform hierarchical clustering based on these distances to form K clusters
 
## 3. Train Composite HMMs for Each Cluster
- For each cluster, train a new HMM on the sequences within the cluster.
- Follow the same training procedure as in Step 1: initialize with k-means and optimize using the Baum-Welch algorithm.
 
## 4. Determine the Optimal Number of Clusters (𝐾)
- Explore different values of K (number of clusters) and m (number of HMM states).
- Select the combination that yields the lowest AIC or BIC score.
