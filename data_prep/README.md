# Data Preprocessing Summary
This project involves analyzing data collected from a mobile health (mHealth) app for diabetes management. Participants used the app to log daily health behaviors such as blood glucose, meals, medication intake, and exercise. Additional data was collected through two rounds of surveys and nurse call notes. Below is a detailed explanation of how each dataset was cleaned and prepared, structured for reproducibility.

## Participants
- Only include participants who completed both the 1st and 2nd surveys.

## Time Range Filtering
- In our study, each participant’s data should be limited to:
  - **12 weeks of intervention**
  - **+ 4 weeks of post-intervention follow-up**
- The time window is determined **individually**, based on the participant’s **app enrollment date**, which is available in the **nurse call dataset**.
> **Note**: The duration can be adjusted depending on your study design.  
> For example, if your intervention lasts for 8 weeks instead of 12, simply modify the filtering period accordingly.

## Daily Activity Aggregation & Data Format
For each user and each day, calculate both **engagement** and **adherence** based on the app's main features.  
In our diabetes mHealth app, the four key features were:
- **Exercise**
- **Meals**
- **Medication**
- **Blood sugar logs**
  
The final dataset should be structured as a **4-dimensional multivariate time series** per user.  
For **each day**, record the following:
- **Logging count** (*engagement*) for each feature  
- **Input values** (*adherence*), if applicable
> **Note**: These features may vary depending on the mHealth app used in your study.  
> Customize this step based on the types of behaviors or metrics your app tracks (e.g., sleep, mood, symptoms, etc.).

## Data Cleaning
To ensure quality and prevent noise in the data:
- Abnormal or unrealistic logs were excluded  
  - **Example**: If a user logged **more than 4 meals** or **blood sugar checks** in a day, these entries were **removed** as outliers.
