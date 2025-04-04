# 🔊 Audiobook App Churn Prediction

This project uses supervised machine learning to predict customer churn in an audiobook subscription app. By analyzing user behavior, purchase activity, and engagement metrics, the model identifies users who are likely to stop using the app — enabling proactive retention strategies.

---

## 📌 Objective

To build a machine learning model that predicts **whether a user will churn**, defined as **not making a purchase in the final 6 months** of the dataset. The end goal is to help the business **retain users before they leave**.

---

## 🧠 Key Insights from EDA

- **Completion does not equal retention** — many users churn after completing one audiobook, revealing a “one-and-done” behavior pattern.
- **Post-purchase engagement time** (how long users interact after buying) is a strong retention signal.
- **Review scores and book price** show predictive patterns tied to user satisfaction and churn.
- **Zero engagement after purchase** almost always leads to churn.

---

## 🏗️ Modeling Pipeline

1. **Data Preprocessing**
   - Column renaming and cleaning
   - Feature engineering (e.g., completion rates, engagement duration)
   - StandardScaler applied to numeric features

2. **Handling Class Imbalance**
   - Used **SMOTE** to oversample the minority class (`Target = 1`, active users)
   - Tuned the oversampling ratio from 2:1 to **0.75**, significantly improving precision and F1-score for active users

3. **Model Training**
   - Evaluated: Logistic Regression, SVC, Random Forest, HistGradientBoosting
   - Performed Grid Search hyperparameter tuning
   - Best performance from **Random Forest + SMOTE (0.75)**

4. **Evaluation**
   - Metrics tracked: precision, recall, F1-score, confusion matrix
   - Focused on minority class (active users) performance

---

## 📊 Final Model Performance (Random Forest + SMOTE @ 0.75)

Precision (Active) : 88% Recall (Active) : 76% F1-score (Active) : 82% Recall (Churned) : 90% Overall Accuracy : 83%


This final model achieves high confidence and balance in identifying users likely to remain active, while still correctly detecting churned users.

---

## 💾 Deployment

The model and scaler were serialized using `joblib`:

```python
joblib.dump(rf_best_estimator, 'random_forest_churn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
```

These can be used in production to:

- Score new user data in batch
- Power an API endpoint (e.g., with FastAPI)
- Drive a churn-risk dashboard (e.g., with Streamlit)
  
## 📁 Files

- notebook.ipynb – Full exploratory and modeling workflow
- random_forest_churn_model.pkl – Final serialized model
- scaler.pkl – StandardScaler object used in preprocessing
- churn_model_bundle.pkl – (Optional) full package with metadata

## 🧪 Future Improvements

- Add user-level time series features (recency, frequency, momentum)
- Fine-tune the decision threshold for higher recall
- Integrate with real-time data pipelines or frontend dashboards
- Monitor model drift as new user behavior emerges
