# Term Deposit Subscription Prediction  
**Author:** Miriam Ocloo Kwashiego  
**Date:** 15th June 2025  

---

## Objective  
To predict whether a bank client will subscribe to a term deposit based on features collected during a direct marketing campaign. This model supports the marketing team in identifying potential customers likely to convert.

---

## Dataset Overview  

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing)  
- **File used:** `bank-full.csv`  
- **Records:** 45,211  
- **Target variable:** `y` (`yes` = subscribed, `no` = not subscribed)  
- **Challenge:** Highly imbalanced data — only ~11% positive class

---

## Key Findings from EDA  

- Majority of clients did not subscribe (~89% “no”)
- Top influencing features:
  - `duration` – longer call durations correlate strongly with “yes”
  - `poutcome` – previous campaign success increases chance of subscription
  - `month` – May and August had higher “yes” rates
  - `contact` – mobile contact outperforms telephone
  - `job` – clients in management or technician roles were more likely to subscribe

---

## Data Preprocessing & Feature Engineering  

- Categorical variables encoded using Label Encoding  
- Numeric variables scaled using StandardScaler  
- Target variable encoded as binary (1 = yes, 0 = no)

---

## Model Development  

- **Model used:** Random Forest Classifier  
- **Class imbalance addressed** using `class_weight='balanced'`  
- **Train/Test Split:** 80/20  
- Trained on all 17 available features from the dataset

---

## Model Evaluation  

| Metric             | Value   |
|--------------------|---------|
| **Accuracy**       | 90%     |
| **Precision (yes)**| 66%     |
| **Recall (yes)**   | 35%     |
| **F1 Score (yes)** | 46%     |

> The model performs well overall but has lower recall for the minority class ("yes") due to class imbalance. Future work could include oversampling or other rebalancing techniques to boost recall.

---

## Business Insights & Recommendations  

- Focus outreach on clients with **prior success history**
- Prioritize engagement during **high-response months** (May, August)
- Encourage **longer and more meaningful phone interactions**
- Use **mobile contact** rather than telephone for better results
- Personalize campaigns using **job type** and **education level**

---

## Deliverables for Review  

| File | Description |
|------|-------------|
| `Term_Deposit_Model.ipynb` | Google Collab notebook with analysis, model training, and evaluation |
| `README.md` | Summary of the project |
| `requirements.txt` | All required Python packages |
| `model.pkl` | The trained model file (`model.pkl`) was too large to upload directly to GitHub. [Download model.pkl](https://drive.google.com/file/d/1m6N4vYimMUn4qkWnyLPFqdYlihqOZfrH/view?usp=sharing) |


---
