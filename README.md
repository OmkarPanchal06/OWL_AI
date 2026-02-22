#  Chronic OWL AI  
## Chronic Kidney Disease Prediction using Machine Learning & Explainable AI



##  Project Overview

**Chronic OWL AI** is an end-to-end Machine Learning and Deep Learning project developed to predict **Chronic Kidney Disease (CKD)** using clinical and laboratory parameters.

The system integrates advanced data preprocessing techniques, multiple classification models, and Explainable AI (XAI) methods to deliver accurate, reliable, and interpretable predictions.

This project demonstrates practical implementation of healthcare analytics, model comparison, and interpretable AI systems.



##  Objectives

- Develop a robust CKD prediction model  
- Compare traditional ML and Deep Learning approaches  
- Improve model reliability through preprocessing techniques  
- Provide interpretability using SHAP and LIME  
- Identify key medical indicators influencing CKD  



##  Dataset Description

The dataset consists of clinical features related to kidney health.

###  Key Features

- Blood Pressure (Bp)  
- Specific Gravity (Sg)  
- Albumin (Al)  
- Sugar (Su)  
- Blood Urea (Bu)  
- Serum Creatinine (Sc)  
- Sodium (Sod)  
- Potassium (Pot)  
- Hemoglobin (Hemo)  
- White Blood Cell Count (Wbcc)  
- Red Blood Cell Count (Rbcc)  
- Hypertension (Htn)  

###  Target Variable

- **CKD Classification**
  - 0 → No CKD  
  - 1 → CKD  



##  Exploratory Data Analysis (EDA)

The following analytical steps were performed:

- Dataset structure and summary statistics analysis  
- Missing value identification and treatment  
- Duplicate record removal  
- Class distribution visualization  
- Feature distribution analysis (Histograms)  
- Outlier detection using Boxplots  
- Correlation heatmap generation  
- Feature-to-target correlation assessment  



##  Data Preprocessing

- Missing value imputation  
- Outlier removal using the IQR method  
- Feature scaling using StandardScaler  
- Train-test split  
- Class imbalance handling techniques (if applied)  



##  Machine Learning Models Implemented

The following classification models were trained and evaluated:

- Logistic Regression  
- Support Vector Machine (SVM)  
- Decision Tree  
- Random Forest  
- XGBoost  
- CatBoost  

###  Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix  
- Cross-Validation Score  



##  Deep Learning Models

### 1️. Artificial Neural Network (ANN)

- Fully connected dense layers  
- ReLU activation  
- Sigmoid output layer  
- Binary classification framework  

### 2️. Hybrid CNN-LSTM Model

- Conv1D layer for feature extraction  
- MaxPooling layer  
- LSTM layer for sequential learning  
- Fully connected output layer  



##  Explainable AI (XAI)

To enhance model transparency and trust:

###  SHAP (SHapley Additive Explanations)

- Global feature importance visualization  
- Individual feature impact analysis  
- Model-level interpretability  

###  LIME (Local Interpretable Model-agnostic Explanations)

- Instance-level prediction explanation  
- Local feature contribution analysis  



##  Technology Stack

- Python  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- XGBoost  
- CatBoost  
- TensorFlow / Keras  
- SHAP  
- LIME  



##  Project Workflow

1. Data Collection  
2. Data Cleaning & Preprocessing  
3. Exploratory Data Analysis  
4. Feature Engineering  
5. Model Training  
6. Model Evaluation  
7. Explainable AI Integration  
8. Performance Comparison  



## Key Outcomes

- Ensemble models demonstrated strong predictive performance.  
- Deep learning models achieved competitive classification results.  
- SHAP and LIME improved interpretability and transparency.  
- Critical medical indicators influencing CKD were successfully identified.  



##  Future Enhancements

- Hyperparameter tuning (GridSearchCV / Optuna)  
- Model deployment using Flask or FastAPI  
- Web-based prediction dashboard  
- Real-time CKD risk prediction system  
- Model monitoring and retraining pipeline  



##  Author

**Omkar Panchal**



- If you found this project valuable, consider giving it a star!!
