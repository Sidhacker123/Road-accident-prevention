Project successfully analyzed road traffic accident severity using various 
machine learning techniques, highlighting the potential of predictive modeling in improving 
road safety.

By applying models like Logistic Regression, Naïve Bayes, KNN, LDA/QDA, 
Random Forest, and XGBoost, we were able to evaluate their performance in predicting 
accident severity, particularly focusing on recall as the critical metric, having precision 
and recall > 80% for predicting “most severe accidents”, having overall accuracy >70%.
The results indicated that while Random Forest and XGBoost achieved high accuracy, 
precision and recall for severe accidents, logistic models like Lasso and Ridge
demonstrated a balanced performance, making them suitable for real-time applications. 

Additionally, the KNN model, combined with PCA and SMOTE, improved visuali ation and 
minority class detection, but its performance highlighted the importance of addressing 
data imbalance. Models like Random Forest and XG Boost can be helpful in determining key 
predictors with large datasets, but the algorithmic approach differs resulting in different 
picks in features. Across all models, key predictors such as weather conditions, vehicle 
movements, and causes of accidents emerged as consistent factors influencing severity, 
underscoring the importance of targeted interventions in these areas.
