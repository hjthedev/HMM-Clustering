# Data Preprocessing Summary
This project involves analyzing data collected from a mobile health (mHealth) app for diabetes management. Below outlines the data preprocessing pipeline step by step for reproducibility and clarity.

## Participants
- Only include participants who completed both the 1st and 2nd surveys.

## Time Range Filtering
- In our study, each participant’s data should be limited to:
  - **12 weeks of intervention**
  - **+ 4 weeks of post-intervention follow-up**
- This time window is **individualized per participant**, starting from their **app enrollment date**, which is available in the **nurse call dataset**.
> **Note**: Modify the duration (e.g., 8 or 16 weeks) based on your specific study design.

## Daily Activity Aggregation
For each user and each day, compute the following:

### 1. Engagement (Logging Count)
- Count how many times each key feature was logged:
  - **Exercise**
  - **Meal**
  - **Medication**
  - **Blood sugar logs**

### 2. Adherence (Input Value)
- Record actual inputs values (e.g., blood sugar values, medication taken) **if available**
> Customize this step based on the features available in your mHealth app.

## Data Cleaning
To ensure quality and prevent noise in the data:
- Abnormal or unrealistic logs were excluded  
  - **Example**: If a user logged **more than 4 meals** or **blood sugar checks** in a day, these entries were **removed** as outliers.
 
## Patient-Level Time Series Generation

After aggregation and cleaning, transform the data into a **patient-wise multivariate time series** for modeling or clustering.

Each patient’s data becomes a NumPy array of shape **(N days × K features)**:   
Where `K` is the number of features (e.g., 4 in this case: exercise, meal, medicine, blood sugar)

A few rows in the daily dataframe might look like this:

| date       | id     | exercise | meal | medication | blood |
|------------|--------|----------|------|-----------|-------|
| 2023-04-10 | 유\*근 | 1        | 1    | 1        | 1     |
| 2023-04-11 | 유\*근 | 1        | 1    | 1        | 1     |
| ...        | ...    | ...      | ...  | ...      | ...   |

You then convert this into a **dictionary** format where:

```python
patient_sequences = {
    '유*근': np.array([[1, 1, 1, 1], ...]),
    '강*자': np.array([[1, 1, 1, 1], ...]),
    # ...
}
```

> **Note**: The values in patient_sequences reflect engagement counts. If your study focuses on adherence values instead, modify accordingly.
