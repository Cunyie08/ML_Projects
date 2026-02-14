# 🍷 White Wine Quality Classification

**End-to-End Machine Learning Pipeline with Model Optimization & Validation**

---

## 📌 Project Objective

Develop a supervised machine learning classification system to predict white wine quality using physicochemical attributes.

This project demonstrates:

* Structured exploratory data analysis (EDA)
* Data cleaning and preprocessing pipeline
* Feature scaling strategy
* Multi-model benchmarking
* Hyperparameter optimization
* Cross-validation
* Class-level performance analysis
* Business translation of model insights

---

## 📊 Dataset Description

* **Source:** UCI White Wine Quality Dataset
* **Samples:** 4,898
* **Features:** 11 numerical physicochemical measurements
* **Target:** Quality score (integer 3–9)

### Data Integrity

* No missing values
* 937 duplicate rows detected and removed
* All features numerical (no categorical encoding required)

---

## 🔎 Exploratory Data Analysis (EDA)

### 1️⃣ Distribution & Variability

* Most wines rated **5–6** → dataset is **class-imbalanced**
* Extreme classes (3–4, 7–9) are underrepresented
* Sulfur dioxide variables exhibit high variance
* Alcohol shows positive association with higher quality
* Density & volatile acidity negatively correlated with quality

### 2️⃣ Scaling Requirement

Feature ranges vary significantly (e.g., density vs alcohol vs sulfur dioxide).

Applied:

```python
MinMaxScaler()
```

Rationale:

* Ensures feature parity
* Prevents scale-dominant learning
* Required for linear models & margin-based classifiers

---

## 🧹 Data Preparation Pipeline

1. Remove duplicates
2. Separate features (X) and target (y)
3. Reclassify target into grouped categories
4. Train-test split
5. Apply scaling to training data
6. Transform test data using fitted scaler

---

## 🤖 Model Benchmarking

Implemented multiple supervised classifiers:

* Logistic Regression (Multinomial)
* LinearSVC
* RandomForestClassifier

Evaluation metrics:

* Accuracy
* Precision
* Recall
* Confusion Matrix
* Cross-validation score

---

## 🏆 Best Model: Random Forest Classifier

Random Forest outperformed linear models due to:

* Non-linear decision boundaries
* Robustness to feature interactions
* Reduced variance through ensemble averaging

---

## 🔄 Hyperparameter Optimization

Used:

```python
RandomizedSearchCV
```

Reason:

* More computationally efficient than GridSearch
* Suitable for moderate search spaces

### Best Parameters:

```python
{
    'n_estimators': 100,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_depth': 20
}
```

---

## 📈 Model Performance

### Test Accuracy:

**79%**

Interpretation:
≈ 3 out of 4 predictions are correct.

---

### Cross-Validation Score:

**~68.5%**

Used k-fold cross-validation to evaluate generalization.

Indicates:

* Moderate variance between folds
* Some instability due to class imbalance

---

## 📊 Class-Level Performance

| Class    | Recall    | Precision | Interpretation   |
| -------- | --------- | --------- | ---------------- |
| Good     | 0.84      | 0.74      | Strong detection |
| Average  | 0.56      | 0.58      | Moderate         |
| Best/Bad | 0.00–0.09 | Very Low  | Underrepresented |

### Key Technical Insight

The model struggles with rare classes due to:

* Severe class imbalance
* Insufficient training examples
* Limited separability in feature space

---

## ⚠️ Identified Limitations

* Class imbalance not explicitly handled (no SMOTE or class weighting applied)
* Quality labels are ordinal but treated as categorical
* Cross-validation score lower than test accuracy (possible overfitting or split variance)

---

## 💾 Model Persistence

Saved:

* Trained Random Forest model
* MinMaxScaler

For reproducibility and deployment readiness.

---

## 🧠 Feature-Level Insights

Based on model interpretation & EDA:

Higher quality wines are associated with:

* Higher alcohol content
* Lower density
* Lower volatile acidity

These variables act as potential production levers.

---

## 🚀 Future Improvements

* Apply class balancing techniques:

  * SMOTE
  * Class weighting
* Explore ordinal classification approaches
* Try Gradient Boosting / XGBoost / LightGBM
* Implement SHAP for feature explainability
* Perform ROC-AUC analysis per class
* Deploy model via FastAPI

---

## 🛠 Tech Stack

* Python
* Pandas
* NumPy
* Matplotlib / Seaborn
* Scikit-learn

---

## 📌 What This Project Demonstrates

* Practical ML pipeline implementation
* Structured experimentation
* Model selection based on empirical performance
* Proper validation strategy
* Business-aware interpretation of results
* Reproducible model training workflow

---
