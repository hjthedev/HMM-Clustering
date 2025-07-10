# Beyond Frequency: Longitudinal Engagement Patterns in mHealth Interventions for Older Adults with Type 2 Diabetes
This is a research-driven repository analyzing user engagement patterns in a 12-week mHealth intervention for older adults with Type 2 Diabetes (T2D). This project leverages **Hidden Markov Models (HMMs)** to uncover longitudinal behavioral patterns and evaluate their clinical relevance and sustainability.


## Getting Started
### 1. Create a Conda environment with Python 3.8.0
```python
conda create -n hmm python=3.8.0
conda activate hmm
```
### 2. Install the requirements
```python
pip install -r requirements.txt
```
## How to Use This Repository

| If you're interested in...                                  | Check this folder              |
|---------------------------------------------------------------|----------------------------------|
| Applying time-series models to mobile health data             | [`hmm_clustering/`](./hmm_clustering) |
| Reproducing engagement-to-outcome modeling                    | [`gee_analysis/`](./gee_analysis)   |
| Investigating long-term behavioral change                     | [`post_intervention/`](./post_intervention) |
| Preprocessing mobile app logs                                 | [`data_prep/`](./data_prep)       |

> **Tip**: You can adapt the HMM structure (e.g., number of hidden states or behavioral features) for other mHealth datasets by modifying `hmm_clustering.py`.
