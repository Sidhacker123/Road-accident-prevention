# Loading libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, 
f1_score
# Load the dataset
df = pd.read_csv("cleaned.csv")
# Encoding categorical variables
label_encoder = LabelEncoder()
for column in df.columns:
  if df[column].dtype == 'object':
   df[column] = label_encoder.fit_transform(df[column])
# Combine class 0 and class 1 into a single class for binary classification
df['Accident_severity'] = df['Accident_severity'].apply(lambda x: 1 if x in [0, 1] else 2)
# Split data into features (X) and target variable (y)
X = df.drop('Accident_severity', axis=1) 
y = df['Accident_severity'] 
# Split the dataset into training (70%) and testing (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, 
                                                    stratify=y)
# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Apply SMOTE to balance the classes in the training set
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
# Apply PCA to retain 95% of variance in the data
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_smote)
X_test_pca = pca.transform(X_test_scaled)
# Train the KNN classifier on the SMOTE + PCA-transformed data
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_pca, y_train_smote)
# Make predictions on the PCA-transformed test data
y_pred = knn.predict(X_test_pca)
# Model performance
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
# Confusion matrix
fig, ax = plt.subplots(figsize=(8, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=knn.classes_)
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.title("Confusion Matrix for Binary KNN Classifier with SMOTE + PCA (70% Training, 30% Testing)")
plt.show()
# Visualize the KNN predictions using PCA components for the test set
plt.figure(figsize=(10, 6))
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap='coolwarm', edgecolor='k', s=50, 
            label='True Labels')
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_pred, marker='x', cmap='coolwarm', s=30, 
            label='Predicted Labels')
plt.title('KNN Classifier - PCA Components with True vs. Predicted Labels (70% Training, 30% Testing), K = 1')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(loc='upper right')
plt.show()
# results
print("Confusion Matrix:")
print(conf_matrix)
print("\nAccuracy:", accuracy)
print("\nClassification Report:")
print(class_report)
# Perform PCA on the SMOTE-standardized training data to get all components for the scree 
plot
pca_full = PCA()
pca_full.fit(X_train_smote)
# Variance for each component
explained_variance = pca_full.explained_variance_ratio_
# Plot the scree plot
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance)
plt.title('Scree Plot for PCA After SMOTE')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.xticks(range(1, len(explained_variance) + 1))
plt.show()
# Calculate precision and recall
num_classes = conf_matrix.shape[0]
precision_per_class = []
recall_per_class = []
for i in range(num_classes):
 tp = conf_matrix[i, i] # True Positives for class i
 fp = conf_matrix[:, i].sum() - tp # False Positives for class i
 fn = conf_matrix[i, :].sum() - tp # False Negatives for class i
 precision_class = tp / (tp + fp) if (tp + fp) > 0 else 0
 recall_class = tp / (tp + fn) if (tp + fn) > 0 else 0
 precision_per_class.append(precision_class)
 recall_per_class.append(recall_class)
# Print precision, recall
for i in range(num_classes):
  print(f"Class {i + 1}: Precision = {precision_per_class[i]:.4f}, Recall = 
{recall_per_class[i]:.4f}")
#Cross Validation with SMOTE + Binary Classification
k_values = range(1, 21)
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
for k in k_values:
  pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('pca', PCA(n_components=0.95)),
    ('knn', KNeighborsClassifier(n_neighbors=k))
  ])
cv_results = cross_val_score(pipeline, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, 
                                                                random_state=42), scoring='accuracy')
accuracy_scores.append(np.mean(cv_results))
# Plot cross-validation results
plt.figure(figsize=(12, 8))
plt.plot(k_values, accuracy_scores, marker='o', linestyle='-', label='Accuracy')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Cross-Validation Accuracy')
plt.title('Cross-Validation Accuracy for Binary KNN Classifier with SMOTE + PCA')
plt.grid()
plt.legend()
plt.show()
print(f"Best K: {k_values[np.argmax(accuracy_scores)]}, Best Accuracy: {max(accuracy_scores):.4f}")