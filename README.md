This project successfully analyzed road traffic accident severity using both traditional machine learning techniques and large language models (LLMs), highlighting the potential of predictive modeling and explainable AI in improving road safety.

🔍 Machine Learning Pipeline
We applied a range of supervised ML algorithms including:

Logistic Regression (Lasso & Ridge)

Naïve Bayes

K-Nearest Neighbors (KNN) with PCA + SMOTE

Linear/Quadratic Discriminant Analysis (LDA/QDA)

Random Forest

XGBoost

📈 Performance Highlights:

Precision and Recall > 80% for predicting “most severe accidents.”

Overall Accuracy > 70% across all models.

Random Forest & XGBoost showed top accuracy and recall, especially for severe cases.

Logistic models provided a balanced trade-off between interpretability and performance, making them ideal for real-time safety dashboards.

🔑 Key Predictors Identified:

Weather conditions

Vehicle movements

Cause of accident These emerged as consistent high-impact features across all models.

🧠 GPT-3 Integration (LLM-based Enhancement)
We enhanced the system by fine-tuning OpenAI’s GPT-3 model on our structured accident dataset. This transforms the system from just predictive modeling to interpretable AI with natural language input/output capabilities.

✅ New Capabilities:
Accepts natural language descriptions of driving conditions (e.g., “Rainy, nighttime, Y-junction, speeding”).

Predicts accident severity with GPT-3 using fine-tuned prompts.

Explains accident risks in human-readable language using LLM-driven logic.

Bridges the gap between ML predictions and real-world explainability.

🛠 Technical Stack
Python (Pandas, scikit-learn, XGBoost)

OpenAI GPT-3 (Fine-tuned using JSONL format)

Data preprocessing: SMOTE, PCA

Jupyter Notebooks + CLI scripts

Ready-to-use files:

convert_to_jsonl.py – prepares fine-tuning data

predict_with_finetuned_gpt3.py – runs severity predictions via GPT

🌐 Applications
Government road safety dashboards

AI co-pilot systems for transport agencies

Insurance risk analysis

Traffic pattern prediction & alert systems


