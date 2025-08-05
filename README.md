# Multiclass Classification of Cancer Subtypes (Synthetic Dataset)

This project demonstrates a **multiclass machine learning pipeline** using synthetic gene expression data to predict **breast cancer subtypes**:  
**Basal-like**, **HER2-enriched**, **Luminal A**, and **Normal-like**.

> _Note: This dataset is synthetic and randomly generated. The project serves as a proof of principle for multiclass classification workflows, not for biomedical inference._

---

## Project Overview

We evaluate and compare the performance of three machine learning models (unbalanced and balanced):

1. **HistGradientBoostingClassifier**
2. **Logistic Regression**
3. **Random Forest**

Each model is assessed using:

- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)
- Multiclass ROC AUC Curve
- Optimal classification thresholds using **Youden’s J statistic**

---

## Technologies Used

| Library        | Purpose                          |
| -------------- | -------------------------------- |
| `scikit-learn` | ML models, evaluation metrics    |
| `matplotlib`   | Plotting ROC curves and matrices |
| `seaborn`      | Enhanced heatmaps                |
| `numpy`        | Data manipulation, J-statistic   |
| `joblib`       | Model saving/loading             |

---

## Models & Techniques

### Baseline Models:

- **Logistic Regression**
- **Random Forest**
- **HistGradientBoostingClassifier**

### Model Evaluation:

- Precision, Recall, F1-score
- Accuracy and Weighted Averages
- ROC AUC (One-vs-Rest)
- Threshold optimisation via **Youden's J** (TPR - FPR)

### Hyperparameter Tuning:

`GridSearchCV` was used to optimise the `HistGradientBoostingClassifier` with 5-fold cross-validation, scoring by **macro-averaged F1**.

---

## Project Structure

```
.
├── figures/                # Synthetic gene expression distribution, Confusion matrices, ROC curves
├── model/                  # Saved .joblib model
├── cancer-subtypes.py      # Core script
├── requirements.txt        # Project dependencies
└── README.md               # You're here!
```

---

## Limitations

- The dataset is **synthetic** and does not represent real gene expression data.
- **Normal-like** subtype consistently underperforms — illustrating challenges with class imbalance.
- ROC AUC scores are mostly low, as expected with randomly generated features.

---

## Future Directions

- Use **real-world RNA-seq datasets** (e.g., TCGA, METABRIC).
- Apply **SMOTE** or **ADASYN** to address minority classes.
- Introduce **biological features** such as pathway scores (e.g., GSVA) or immune signatures.
- Wrap model evaluation into **reusable functions** for future projects.

---

## How to Use

```bash
# Install dependencies
pip install -r requirements.txt

# Run the analysis
python cancer-subtypes.py
```
