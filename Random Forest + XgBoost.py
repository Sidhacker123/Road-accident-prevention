import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, precision_score, recall_score, 
confusion_matrix
data = pd.read_csv('/content/cleaned.csv')
# Merge class 0 and 1 into a single class (e.g., 0), and keep class 2 as is
data['Accident_severity'] = data['Accident_severity'].replace({0: 0, 1: 0, 2: 1})
X = data.drop('Accident_severity', axis=1) # Features
y = data['Accident_severity'] # Target (Accident_severity column)
# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, 
stratify=y)
# LabelEncoder
label_encoder = LabelEncoder()
for col in X_train.select_dtypes(include=['object']).columns:
 X_train[col] = label_encoder.fit_transform(X_train[col])
 X_test[col] = label_encoder.transform(X_test[col])
# SMOTE for oversampling
smote = BorderlineSMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print("\nResampled Training Set Description:")
print(f"Features Shape: {X_train_smote.shape}")
print(f"Target Counts:\n{y_train_smote.value_counts()}")
#RANDOM FOREST MODEL
# hyperparameter tuning
param_grid_rf = {
'n_estimators': [100, 200, 500],
'max_depth': [10, 20, None],

'min_samples_split': [2, 5, 10],
'min_samples_leaf': [1, 2, 4],
'max_features': ['sqrt', 'log2', None]
}
grid_search_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
param_grid=param_grid_rf,
scoring='accuracy', cv=3)
grid_search_rf.fit(X_train_smote, y_train_smote)
best_rf_model = grid_search_rf.best_estimator_
y_pred_rf = best_rf_model.predict(X_test)
print("\nRandom Forest - Classification Report:\n", classification_report(y_test, y_pred_rf))
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
print("\nRandom Forest - Confusion Matrix:\n", conf_matrix_rf)
precision_rf = precision_score(y_test, y_pred_rf, average='binary')
recall_rf = recall_score(y_test, y_pred_rf, average='binary')
print(f"Random Forest - Precision: {precision_rf}")
print(f"Random Forest - Recall: {recall_rf}")


#XGBOOST MODEL:
# hyperparameter tuning
param_grid_xgb = {
'n_estimators': [100, 200, 500],
'max_depth': [5, 10, 15],
'learning_rate': [0.01, 0.05, 0.1],
'subsample': [0.7, 0.8, 1.0],
'colsample_bytree': [0.7, 0.8, 1.0],
'gamma': [0, 0.1, 0.5]
}
grid_search_xgb = GridSearchCV(estimator=XGBClassifier(random_state=42),
param_grid=param_grid_xgb,
scoring='accuracy', cv=3)
grid_search_xgb.fit(X_train_smote, y_train_smote)
best_xgb_model = grid_search_xgb.best_estimator_
y_pred_xgb = best_xgb_model.predict(X_test)
print("\nXGBoost - Classification Report:\n", classification_report(y_test, y_pred_xgb))
conf_matrix_xgb = confusion_matrix(y_test, y_pred_xgb)
print("\nXGBoost - Confusion Matrix:\n", conf_matrix_xgb)
# Precision and Recall scores for XGBoost
precision_xgb = precision_score(y_test, y_pred_xgb, average='binary')
recall_xgb = recall_score(y_test, y_pred_xgb, average='binary')
print(f"XGBoost - Precision: {precision_xgb}")
print(f"XGBoost - Recall: {recall_xgb}")
import numpy as np
import matplotlib.pyplot as plt
# RANDOM FOREST VARIABLE IMPORTANCE
# Extract feature importances from the best Random Forest model
rf_importances = best_rf_model.feature_importances_
rf_feature_names = X_train.columns
sorted_indices_rf = np.argsort(rf_importances)[::-1]
print("\nRandom Forest - Feature Importances:")
for idx in sorted_indices_rf:
 print(f"{rf_feature_names[idx]}: {rf_importances[idx]:.4f}")
plt.figure(figsize=(10, 6))
plt.title("Random Forest - Feature Importances")
plt.bar(range(X_train.shape[1]), rf_importances[sorted_indices_rf], align="center")
plt.xticks(range(X_train.shape[1]), rf_feature_names[sorted_indices_rf], rotation=90)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()
# XGBoost model feature importance
xgb_importances = best_xgb_model.feature_importances_
xgb_feature_names = X_train.columns
sorted_indices_xgb = np.argsort(xgb_importances)[::-1]
print("\nXGBoost - Feature Importances:")
for idx in sorted_indices_xgb:
 print(f"{xgb_feature_names[idx]}: {xgb_importances[idx]:.4f}")
plt.figure(figsize=(10, 6))
plt.title("XGBoost - Feature Importances")
plt.bar(range(X_train.shape[1]), xgb_importances[sorted_indices_xgb], align="center")
plt.xticks(range(X_train.shape[1]), xgb_feature_names[sorted_indices_xgb], rotation=90)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()
