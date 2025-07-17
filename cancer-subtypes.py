# 0. Import libraries
# Saving and loading
import os
import joblib

# Data manipulation
import numpy as np
import pandas as pd
import random

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# ML
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report, 
    confusion_matrix, 
    f1_score, 
    make_scorer,
    precision_recall_curve, 
    precision_score, 
    recall_score, 
    roc_auc_score, 
    roc_curve,
    RocCurveDisplay
    )
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import label_binarize


# 1. Data Acquisition and Initial Exploration (EDA)
print("---------- 1. Data Acquisition and Initial Exploration (EDA) ----------")

# Using pandas, we will simulate a realistic gene expression dataset

# Create a pandas dataframe with 150 rows (sample) and 750 columns (genes)
synthetic_gene_df = pd.DataFrame(index=np.arange(150), columns=np.arange(750))

# Rename column names to 'Gene_1', 'Gene_2' ... 'Gene_n'
synthetic_gene_df.columns = [f"Gene_{i+1}" for i in range(synthetic_gene_df.shape[1])]

# Set seed for reproducibility
np.random.seed(42)

# Generate random genes expression values ranging between 0.1 and 25000 but skewed around 1
    # First, define mean and skew for log normal distribution
    # Add a pseudocount value (epsilon) in case of log2(0)
mu = 0 
sigma = 1.0
epsilon = 1e-4
    # Generate random numbers using the skewed log normal distribution
raw_expression_values = np.random.lognormal(mean=mu, sigma=sigma, size=synthetic_gene_df.shape)
    # log2 transform the expression values
log2_expression_values = np.log2(raw_expression_values + epsilon)
    # Clip the maximum values at the typical upper limit of gene expression data (25000; log2(25000) = ~15)
log2_expression_values = np.clip(log2_expression_values, -np.inf, 15)
    # Apply the log2 expression values to the dataframe
synthetic_gene_df[:] = log2_expression_values
    # Visualise the distribution
plt.hist(log2_expression_values.flatten(), bins=100, color='skyblue', edgecolor='black')
plt.title('Figure 1a: Simulated Log2 Gene Expression')
plt.xlabel('log2(expression)')
plt.ylabel('Frequency')
plt.savefig("figures/fig1a", dpi=300)
plt.show()

# Add a column titled 'Subtype' to synthetic_gene_df
    # 'Subtype' contains the data: 'Luminal A', 'Basal-like', 'HER2-enriched', 'Normal-like'
        # 'Luminal A' the most common, 'Basal-like' and 'HER2-enriched' less common, and 'Normal-like' the least common
subtype_samples = synthetic_gene_df.shape[0]
subtype_counts = {
    'Luminal A': 67,
    'Basal-like': 38,
    'HER2-enriched': 33,
    'Normal-like': 12
}
    # Confirm total number of counts == sample number
assert sum(subtype_counts.values()) == subtype_samples, "Subtype counts must add up to 150"
    # Generate the data for the 'Subtype' column using the specified number for each label
subtype_labels = (
    ['Luminal A'] * subtype_counts['Luminal A'] +
    ['Basal-like'] * subtype_counts['Basal-like'] +
    ['HER2-enriched'] * subtype_counts['HER2-enriched'] +
    ['Normal-like'] * subtype_counts['Normal-like']
)
    # Shuffle the data
random.seed(42)
random.shuffle(subtype_labels)
    # Add the 'Subtype' data to synthetic_gene_df
synthetic_gene_df['Subtype'] = subtype_labels

# Display first 5 rows
print(f"\nA quick peek at the data:\n {synthetic_gene_df.head()}")
# Display the shape of the dataframe
print(f"\nShape of the data:\n {synthetic_gene_df.shape}")
# Summary information
print(f"\nSummary of the data:\n {synthetic_gene_df.info()}")
# Descriptive stats for gene columns
print(f"\nDescriptive stats for the gene columns:\n {synthetic_gene_df.iloc[:,0:749].describe()}")
# Check for missing values
print(f"\n Checking for missing values: {synthetic_gene_df.isnull().sum()}")
# Confirm counts for each subtype
print(f"\nSubtype counts:\n {synthetic_gene_df['Subtype'].value_counts()}")

# Check the expression of a random gene
    # Get all of the columns apart from 'Subtype'
gene_columns = synthetic_gene_df.columns.difference(['Subtype'])
    # Select a random gene (due to the seed selected above, it'll be Gene_9)
random_gene = random.choice(list(gene_columns))

plt.figure(figsize=(10,3))
sns.boxplot(x=synthetic_gene_df[random_gene], color='yellow')
plt.title("Figure 1b: Boxplot Showing the Gene Expression of a Random Gene")
plt.grid(visible=True, linestyle='--', linewidth=0.3, color='gray', alpha=0.3)
plt.tight_layout()
plt.savefig("figures/fig1b", dpi=300)
plt.show()

plt.figure(figsize=(8,6))
sns.violinplot(y=synthetic_gene_df[random_gene])
plt.title("Figure 1c: Violin Plot Showing the Gene Expression of a Random Gene")
plt.grid(visible=True, linestyle='--', linewidth=0.3, color='gray', alpha=0.3)
plt.tight_layout()
plt.savefig("figures/fig1c", dpi=300)
plt.show()

print("-"*100)  # Separator

# 2. Data Preprocessing and Feature Selection
print("---------- 2. Data Preprocessing and Feature Selection ----------")

# Separate features (X) and target (y)
X = synthetic_gene_df.drop(columns=['Subtype'])
y = synthetic_gene_df['Subtype']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3,      # 30% for training, 70% for testing
    random_state=42,    # for reproducibility
    stratify=y          # Ensures an equal proportion of 'subtype' class (as data is imbalanced)
    )

# Dimensionality Reduction/Feature Selection
    # Select the top 50 genes based on their F-statistic score
top_genes = SelectKBest(score_func=f_classif, k=50)
    # Fit and transform data
X_train_selected = top_genes.fit_transform(X_train, y_train)
X_test_selected = top_genes.transform(X_test)
    # Confirm the results
print(f"\nShape of X_train: {X_train_selected.shape}")   #(105, 50) as data has been split for training and testing
print(f"\nShape of X_test: {X_test_selected.shape}")    #(45, 50)
    # Print the top 10 selected genes
selected_genes_indices = top_genes.get_support()    # Returns Boolean values; True if selected
selected_genes = X.columns[selected_genes_indices]
print(f"\nTop 10 genes:\n {selected_genes[:10]}")

print("-"*100)  # Separator

# 3. Model Selection & Training
print("---------- 3. Model Selection & Training ----------")


# Will test three different classification models
    # A linear model: LogisticRegression
    # A tree-based ensemble model: RandomForestClassifier
    # A gradient boosting model: HistGradientBoostingClassifier
# Will test each model:
    # As the unbalanced dataset
    # Balanced using `class_weight = 'balanced'`

# Logistic Regression (Unbalanced)
print("---- Logistic Regression (Unbalanced) ---")

# Initialise the logistic regression model
model_lr = LogisticRegression(random_state=42)

# Fit the logistic regression model
model_lr.fit(X_train_selected, y_train)

# Predict on the test dataset
y_pred_lr = model_lr.predict(X_test_selected)

# Logistic Regression Model Evaluation
print("\nLogistic Regression (Unbalanced) Classification Report:\n")
print(classification_report(y_test, y_pred_lr, zero_division=0))

# Logistic Regression (Unbalanced) Classification Report:

#                precision    recall  f1-score   support

#    Basal-like       0.31      0.36      0.33        11
# HER2-enriched       0.33      0.20      0.25        10
#     Luminal A       0.50      0.60      0.55        20
#   Normal-like       0.00      0.00      0.00         4

#      accuracy                           0.40        45
#     macro avg       0.29      0.29      0.28        45
#  weighted avg       0.37      0.40      0.38        45


# Logistic Regression Confusion Matrix
ordered_labels = ['Basal-like', 'HER2-enriched', 'Luminal A', 'Normal-like']
cm_lr = confusion_matrix(y_test, y_pred_lr)
ax = plt.subplot()
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Figure 3a: Logistic Regression (Unbalanced) - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
ax.xaxis.set_ticklabels(ordered_labels)
plt.xticks(rotation=45, ha='right')
ax.yaxis.set_ticklabels(ordered_labels)
plt.yticks(rotation=0, ha='right')
plt.grid(visible=True, linestyle='--', linewidth=0.3, color='gray', alpha=0.3)
plt.tight_layout()
plt.savefig("figures/fig3a", dpi=300)
plt.show()

# Logistic Regression ROC AUC Curve
    # As this is multi-class data, we will undertake a one vs. rest approach
    # Binarise multi-class labels
class_names = sorted(y.unique())                                # ['Basal-like', 'HER2-enriched', 'Luminal A', 'Normal-like']
y_test_binarised = label_binarize(y_test, classes=class_names)  # Outputs each subtype as [0 0 1 0] (for Luminal A, for example)
y_prob_lr = model_lr.predict_proba(X_test_selected)             # Outputs predicted probabilities for each subtype
    # Set empty dictionaries for ROC
fpr_lr = {}
tpr_lr = {}
thresholds_lr = {}
roc_auc_lr = {}
optimal_thresholds_lr = {}
j_best_indices_lr = {}
    # Calculate ROC curve and AUC for each subtype
for i, class_label in enumerate(class_names):
    fpr_lr[class_label], tpr_lr[class_label], thresholds_lr[class_label] = roc_curve(y_test_binarised[:, i], y_prob_lr[:, i])
    roc_auc_lr[class_label] = auc(fpr_lr[class_label], tpr_lr[class_label])
    
    # Also find the optimal thresholds using Youden’s J statistic (Youden's J = TPR - FPR)
    j_scores_lr = tpr_lr[class_label] - fpr_lr[class_label]
    j_best_index_lr = np.argmax(j_scores_lr)
    j_best_indices_lr[class_label] = j_best_index_lr
    optimal_thresholds_lr[class_label] = thresholds_lr[class_label][j_best_index_lr]

    # Plot the data
plt.figure(figsize=(10,7))
colors = ['red', 'blue', 'green', 'purple']
for i, class_label in enumerate(class_names):
    plt.plot(
        fpr_lr[class_label], 
        tpr_lr[class_label],
        color=colors[i],
        label=f"{class_label} (AUC = {roc_auc_lr[class_label]:.2f})"
        )
    # Add optimal threshold
    best_idx_lr = j_best_indices_lr[class_label]
    plt.scatter(
        fpr_lr[class_label][best_idx_lr],
        tpr_lr[class_label][best_idx_lr],
        color=colors[i],
        marker='o',
        edgecolors='black',
        label=f"Best threshold for {class_label} = {optimal_thresholds_lr[class_label]:.2f}"
    )
    # Add diagonal line for random chance
plt.plot([0,1], [0,1], 'k--', alpha=0.6)
plt.title("Figure 3b: Multiclass ROC Curve (Logistic Regression (Unbalanced))")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.grid(visible=True, linestyle='--', linewidth=0.3, color='gray', alpha=0.3)
plt.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig("figures/fig3b", dpi=300)
plt.show()

print("\nOptimal thresholds (Youden's J) for Logistic Regression (Unbalanced):")
for class_label in class_names:
    print(f"{class_label}: {optimal_thresholds_lr[class_label]:.3f}")


summary_lr = """
Logistic Regression (Unbalanced) - Conclusions

Classification Report
    Basal-like:
        Precision: 0.31 - poor, only 31% of basal-like cases were correctly identified
        Recall: 0.36 - poor, of the cases identified as basal-like only 36% were correct
        F1-score: 0.33 - poor, due to poor performances on precision and recall
    HER2-enriched:
        Precision: 0.33 - poor, only 33% of HER2-enriched cases were correctly identified
        Recall: 0.20 - poor, of the cases identified as HER2-enriched only 20% were correct
        F1-score: 0.25 - poor, due to poor performances on precision and recall
    Luminal A:
        Precision: 0.50 - average, 50% of Luminal A cases were correctly identified
        Recall: 0.60 - decent, of the cases identified as Luminal A 60% were correct
        F1-score: 0.55 - average, due to average performances on precision and recall
    Normal-like:
        Precision: 0 - very poor, no normal-like cases were correctly identified
        Recall: 0 - very poor, model failed to predict any normal-like cases
        F1-score: 0 - very poor, due to very poor performances on precision and recall

Confusion Matrix
    Basal-like (actual) vs. basal-like (predicted): 4 (correct)
    Basal-like (actual) vs. HER2-enriched (predicted): 1 (misclassification)
    Basal-like (actual) vs. Luminal A (predicted): 6 (misclassification)
    Basal-like (actual) vs. Normal-like (predicted): 0

    HER2-enriched (actual) vs. basal-like (predicted): 5 (misclassification)
    HER2-enriched (actual) vs. HER2-enriched (predicted): 2 (correct)
    HER2-enriched (actual) vs. Luminal A (predicted): 3 (misclassification)
    HER2-enriched (actual) vs. Normal-like (predicted): 0

    Luminal A (actual) vs. basal-like (predicted): 4 (misclassification)
    Luminal A (actual) vs. HER2-enriched (predicted): 2 (misclassification)
    Luminal A (actual) vs. Luminal A (predicted): 12 (correct)
    Luminal A (actual) vs. Normal-like (predicted): 2 (misclassification)

    Normal-like (actual) vs. basal-like (predicted): 0
    Normal-like (actual) vs. HER2-enriched (predicted): 1 (misclassification)
    Normal-like (actual) vs. Luminal A (predicted): 3 (misclassification)
    Normal-like (actual) vs. Normal-like (predicted): 0 (none correct)

ROC AUC:
    AUC:
        Basal-like AUC: 0.52 - average separability, slightly better than random guessing
        HER2-enriched AUC: 0.52 - average separability, slightly better than random guessing
        Luminal A AUC: 0.52 - average separability, slightly better than random guessing
        Normal-like AUC: 0.18 - poor separability
    Thresholds:
        Basal-like best threshold: 0.41
            - TPR ≈ 0.45 (45% of actual Basal-like cases are correctly identified)
            - FPR ≈ 0.3 (30% of non-Basal-like cases are wrongly labeled as Basal-like)
        HER2-enriched best threshold: 0.43
            - TPR ≈ 0.30 (30% of actual HER2-enriched cases are correctly identified)
            - FPR ≈ 0.15 (15% of non-HER2-enriched cases are wrongly labeled as HER2-enriched)
        Luminal A best threshold: 0.28
            - TPR ≈ 0.80 (80% of actual Luminal A cases are correctly identified)
            - FPR ≈ 0.55 (55% of non-Luminal A cases are wrongly labeled as Luminal A)
        Normal-like best threshold: inf
            - TPR ≈ 0 (0% of actual Normal-like cases are correctly identified)
            - FPR ≈ 0 (0% of non-Normal-like cases are wrongly labeled as Normal-like)

Summary:
    Overall the logisitic regression model performed poorly.
    It was most effective at predicting Luminal A cases, which is unsurprising, as this is the dominant class.
    It fails to predict Normal-like cases (the minority class) altogether.
"""
print(summary_lr)

print("-"*100)  # Separator

# Random Forest
print("---- Random Forest (Unbalanced) ---")

# Initialise the random forest model
model_rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Fit the random forest model
model_rf.fit(X_train_selected, y_train)

# Predict on the test dataset
y_pred_rf = model_rf.predict(X_test_selected)

# Random Forest Model Evaluation
print("\nRandom Forest (Unbalanced) Classification Report:\n")
print(classification_report(y_test, y_pred_rf, zero_division=0))

# Random Forest (Unbalanced) Classification Report:

#                precision    recall  f1-score   support

#    Basal-like       0.20      0.09      0.12        11
# HER2-enriched       0.00      0.00      0.00        10
#     Luminal A       0.46      0.85      0.60        20
#   Normal-like       0.00      0.00      0.00         4

#      accuracy                           0.40        45
#     macro avg       0.16      0.24      0.18        45
#  weighted avg       0.25      0.40      0.30        45

# Random Forest Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
ax = plt.subplot()
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Figure 3c: Random Forest (Unbalanced) - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
ax.xaxis.set_ticklabels(ordered_labels)
plt.xticks(rotation=45, ha='right')
ax.yaxis.set_ticklabels(ordered_labels)
plt.yticks(rotation=0, ha='right')
plt.grid(visible=True, linestyle='--', linewidth=0.3, color='gray', alpha=0.3)
plt.tight_layout()
plt.savefig("figures/fig3c", dpi=300)
plt.show()

# Random Forest ROC AUC Curve
y_prob_rf = model_rf.predict_proba(X_test_selected)       
    # Set empty dictionaries for ROC
fpr_rf = {}
tpr_rf = {}
thresholds_rf = {}
roc_auc_rf = {}
optimal_thresholds_rf = {}
j_best_indices_rf = {}
    # Calculate ROC curve and AUC for each subtype
for i, class_label in enumerate(class_names):
    fpr_rf[class_label], tpr_rf[class_label], thresholds_rf[class_label] = roc_curve(y_test_binarised[:, i], y_prob_rf[:, i])
    roc_auc_rf[class_label] = auc(fpr_rf[class_label], tpr_rf[class_label])
    
    # Also find the optimal thresholds using Youden’s J statistic (Youden's J = TPR - FPR)
    j_scores_rf = tpr_rf[class_label] - fpr_rf[class_label]
    j_best_index_rf = np.argmax(j_scores_rf)
    j_best_indices_rf[class_label] = j_best_index_rf
    optimal_thresholds_rf[class_label] = thresholds_rf[class_label][j_best_index_rf]

    # Plot the data
plt.figure(figsize=(10,7))
colors = ['red', 'blue', 'green', 'purple']
for i, class_label in enumerate(class_names):
    plt.plot(
        fpr_rf[class_label], 
        tpr_rf[class_label],
        color=colors[i],
        label=f"{class_label} (AUC = {roc_auc_rf[class_label]:.2f})"
        )
    # Add optimal threshold
    best_idx_rf = j_best_indices_rf[class_label]
    plt.scatter(
        fpr_rf[class_label][best_idx_rf],
        tpr_rf[class_label][best_idx_rf],
        color=colors[i],
        marker='o',
        edgecolors='black',
        label=f"Best threshold for {class_label} = {optimal_thresholds_rf[class_label]:.2f}"
    )
    # Add diagonal line for random chance
plt.plot([0,1], [0,1], 'k--', alpha=0.6)
plt.title("Figure 3d: Multiclass ROC Curve (Random Forest (Unbalanced))")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.grid(visible=True, linestyle='--', linewidth=0.3, color='gray', alpha=0.3)
plt.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig("figures/fig3d", dpi=300)
plt.show()

print("\nOptimal thresholds (Youden's J) for Random Forest (Unbalanced):")
for class_label in class_names:
    print(f"{class_label}: {optimal_thresholds_rf[class_label]:.3f}")


summary_rf = """
Random Forest (Unbalanced) - Conclusions

Classification Report
    Basal-like:
        Precision: 0.20 - poor, only 20% of basal-like cases were correctly identified
        Recall: 0.09 - very poor, of the cases identified as basal-like only 9% were correct
        F1-score: 0.12 - poor, due to poor performances on precision and recall
    HER2-enriched:
        Precision: 0 - very poor, no HER2-enriched cases were correctly identified
        Recall: 0 - very poor, model failed to predict any HER2-enriched cases
        F1-score: 0 - very poor, due to very poor performances on precision and recall
    Luminal A:
        Precision: 0.46 - below average, 46% of Luminal A cases were correctly identified
        Recall: 0.85 - great, of the cases identified as Luminal A 85% were correct
        F1-score: 0.60 - above average, due to the below average performance on precision but excellent performance on recall
    Normal-like:
        Precision: 0 - very poor, no normal-like cases were correctly identified
        Recall: 0 - very poor, model failed to predict any normal-like cases
        F1-score: 0 - very poor, due to very poor performances on precision and recall

Confusion Matrix
    Basal-like (actual) vs. basal-like (predicted): 1 (correct)
    Basal-like (actual) vs. HER2-enriched (predicted): 1 (misclassification)
    Basal-like (actual) vs. Luminal A (predicted): 0 (misclassification)
    Basal-like (actual) vs. Normal-like (predicted): 0

    HER2-enriched (actual) vs. basal-like (predicted): 3 (misclassification)
    HER2-enriched (actual) vs. HER2-enriched (predicted): 0 (none correct)
    HER2-enriched (actual) vs. Luminal A (predicted): 7 (misclassification)
    HER2-enriched (actual) vs. Normal-like (predicted): 0

    Luminal A (actual) vs. basal-like (predicted): 1 (misclassification)
    Luminal A (actual) vs. HER2-enriched (predicted): 2 (misclassification)
    Luminal A (actual) vs. Luminal A (predicted): 17 (correct)
    Luminal A (actual) vs. Normal-like (predicted): 0

    Normal-like (actual) vs. basal-like (predicted): 0
    Normal-like (actual) vs. HER2-enriched (predicted): 0
    Normal-like (actual) vs. Luminal A (predicted): 4 (misclassification)
    Normal-like (actual) vs. Normal-like (predicted): 0 (none correct)

ROC AUC:
    AUC:
        Basal-like AUC: 0.29 - poor separability
        HER2-enriched AUC: 0.45 - below average separability
        Luminal A AUC: 0.45 - below average separability
        Normal-like AUC: 0.38 - below average separability
    Thresholds:
        Basal-like best threshold: 0.39
            - TPR ≈ 0.10 (10% of actual Basal-like cases are correctly identified)
            - FPR ≈ 0.00 (0% of non-Basal-like cases are wrongly labeled as Basal-like)
        HER2-enriched best threshold: 0.21
            - TPR ≈ 0.80 (80% of actual HER2-enriched cases are correctly identified)
            - FPR ≈ 0.70 (70% of non-HER2-enriched cases are wrongly labeled as HER2-enriched)
        Luminal A best threshold: 0.40
            - TPR ≈ 0.70 (70% of actual Luminal A cases are correctly identified)
            - FPR ≈ 0.60 (60% of non-Luminal A cases are wrongly labeled as Luminal A)
        Normal-like best threshold: 0.13
            - TPR ≈ 0.25 (25% of actual Normal-like cases are correctly identified)
            - FPR ≈ 0.15 (15% of non-Normal-like cases are wrongly labeled as Normal-like)

Summary:
    Overall the random forest model performed worse than the logisitic regression model.
    It was most effective at predicting Luminal A (majority) cases, as was the logisitic regression model.
    It fails to predict not only the Normal-like cases (the minority class) but also HER2-enriched cases.
"""
print(summary_rf)

print("-"*100)  # Separator

# HistGradientBoostingClassifier (Unbalanced)
print("---- HistGradientBoostingClassifier (Unbalanced) ---")

# Initialise the HistGradientBoostingClassifier model
model_hgbc = HistGradientBoostingClassifier(random_state=42)

# Fit the HistGradientBoostingClassifier model
model_hgbc.fit(X_train_selected, y_train)

# Predict on the test dataset
y_pred_hgbc = model_hgbc.predict(X_test_selected)

# HistGradientBoostingClassifier Model Evaluation
print("\nHistGradientBoostingClassifier (Unbalanced) Classification Report:\n")
print(classification_report(y_test, y_pred_hgbc, zero_division=0))

# HistGradientBoostingClassifier (Unbalanced) Classification Report:

#                precision    recall  f1-score   support

#    Basal-like       0.10      0.09      0.10        11
# HER2-enriched       0.50      0.20      0.29        10
#     Luminal A       0.47      0.70      0.56        20
#   Normal-like       0.00      0.00      0.00         4

#      accuracy                           0.38        45
#     macro avg       0.27      0.25      0.24        45
#  weighted avg       0.34      0.38      0.34        45

# HistGradientBoostingClassifier Confusion Matrix
cm_hgbc = confusion_matrix(y_test, y_pred_hgbc)
ax = plt.subplot()
sns.heatmap(cm_hgbc, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Figure 3e: HistGradientBoostingClassifier (Unbalanced) - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
ax.xaxis.set_ticklabels(ordered_labels)
plt.xticks(rotation=45, ha='right')
ax.yaxis.set_ticklabels(ordered_labels)
plt.yticks(rotation=0, ha='right')
plt.grid(visible=True, linestyle='--', linewidth=0.3, color='gray', alpha=0.3)
plt.tight_layout()
plt.savefig("figures/fig3e", dpi=300)
plt.show()

# HistGradientBoostingClassifier ROC AUC Curve 
y_prob_hgbc = model_hgbc.predict_proba(X_test_selected)       
    # Set empty dictionaries for ROC
fpr_hgbc = {}
tpr_hgbc = {}
thresholds_hgbc = {}
roc_auc_hgbc = {}
optimal_thresholds_hgbc = {}
j_best_indices_hgbc = {}
    # Calculate ROC curve and AUC for each subtype
for i, class_label in enumerate(class_names):
    fpr_hgbc[class_label], tpr_hgbc[class_label], thresholds_hgbc[class_label] = roc_curve(y_test_binarised[:, i], y_prob_hgbc[:, i])
    roc_auc_hgbc[class_label] = auc(fpr_hgbc[class_label], tpr_hgbc[class_label])
    
    # Also find the optimal thresholds using Youden’s J statistic (Youden's J = TPR - FPR)
    j_scores_hgbc = tpr_hgbc[class_label] - fpr_hgbc[class_label]
    j_best_index_hgbc = np.argmax(j_scores_hgbc)
    j_best_indices_hgbc[class_label] = j_best_index_hgbc
    optimal_thresholds_hgbc[class_label] = thresholds_hgbc[class_label][j_best_index_hgbc]

    # Plot the data
plt.figure(figsize=(10,7))
colors = ['red', 'blue', 'green', 'purple']
for i, class_label in enumerate(class_names):
    plt.plot(
        fpr_hgbc[class_label], 
        tpr_hgbc[class_label],
        color=colors[i],
        label=f"{class_label} (AUC = {roc_auc_hgbc[class_label]:.2f})"
        )
    # Add optimal threshold
    best_idx_hgbc = j_best_indices_hgbc[class_label]
    plt.scatter(
        fpr_hgbc[class_label][best_idx_hgbc],
        tpr_hgbc[class_label][best_idx_hgbc],
        color=colors[i],
        marker='o',
        edgecolors='black',
        label=f"Best threshold for {class_label} = {optimal_thresholds_hgbc[class_label]:.2f}"
    )
    # Add diagonal line for random chance
plt.plot([0,1], [0,1], 'k--', alpha=0.6)
plt.title("Figure 3f: Multiclass ROC Curve (HistGradientBoostingClassifier (Unbalanced))")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.grid(visible=True, linestyle='--', linewidth=0.3, color='gray', alpha=0.3)
plt.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig("figures/fig3f", dpi=300)
plt.show()

print("\nOptimal thresholds (Youden's J) for HistGradientBoostingClassifier (Unbalanced):")
for class_label in class_names:
    print(f"{class_label}: {optimal_thresholds_hgbc[class_label]:.3f}")

summary_hgbc = """
HistGradientBoostingClassifier (Unbalanced) - Conclusions

Classification Report
    Basal-like:
        Precision: 0.10 - very poor, only 10% of basal-like cases were correctly identified
        Recall: 0.09 - very poor, of the cases identified as basal-like only 9% were correct
        F1-score: 0.10 - very poor, due to very poor performances on precision and recall
    HER2-enriched:
        Precision: 0.50 - average, 50% of HER2-enriched cases were correctly identified
        Recall: 0.20 - poor, of the cases identified as HER2-enriched only 20% were correct
        F1-score: 0.29 - poor, due to poor performances on precision and recall
    Luminal A:
        Precision: 0.47 - below average, 47% of Luminal A cases were correctly identified
        Recall: 0.70 - great, of the cases identified as Luminal A 70% were correct
        F1-score: 0.56 - above average, due to the below average performance on precision but great performance on recall
    Normal-like:
        Precision: 0 - very poor, no normal-like cases were correctly identified
        Recall: 0 - very poor, model failed to predict any normal-like cases
        F1-score: 0 - very poor, due to very poor performances on precision and recall

Confusion Matrix
    Basal-like (actual) vs. basal-like (predicted): 1 (correct)
    Basal-like (actual) vs. HER2-enriched (predicted): 1 (misclassification)
    Basal-like (actual) vs. Luminal A (predicted): 9 (misclassification)
    Basal-like (actual) vs. Normal-like (predicted): 0

    HER2-enriched (actual) vs. basal-like (predicted): 4 (misclassification)
    HER2-enriched (actual) vs. HER2-enriched (predicted): 2 (correct)
    HER2-enriched (actual) vs. Luminal A (predicted): 4 (misclassification)
    HER2-enriched (actual) vs. Normal-like (predicted): 0

    Luminal A (actual) vs. basal-like (predicted): 5 (misclassification) 
    Luminal A (actual) vs. HER2-enriched (predicted): 0 
    Luminal A (actual) vs. Luminal A (predicted): 14 (correct)
    Luminal A (actual) vs. Normal-like (predicted): 1 (misclassification)

    Normal-like (actual) vs. basal-like (predicted): 0
    Normal-like (actual) vs. HER2-enriched (predicted): 1 (misclassification)
    Normal-like (actual) vs. Luminal A (predicted): 3 (misclassification)
    Normal-like (actual) vs. Normal-like (predicted): 0 (none correct)

ROC AUC:
    AUC:
        Basal-like AUC: 0.42 - below average separability
        HER2-enriched AUC: 0.61 - above average separability
        Luminal A AUC: 0.54 - above average separability
        Normal-like AUC: 0.45 - below average separability
    Thresholds:
        Basal-like best threshold: 0.005
            - TPR ≈ 0.90 (90% of actual Basal-like cases are correctly identified)
            - FPR ≈ 0.75 (75% of non-Basal-like cases are wrongly labeled as Basal-like)
        HER2-enriched best threshold: 0.24
            - TPR ≈ 0.40 (40% of actual HER2-enriched cases are correctly identified)
            - FPR ≈ 0.15 (15% of non-HER2-enriched cases are wrongly labeled as HER2-enriched)
        Luminal A best threshold: 0.81
            - TPR ≈ 0.55 (55% of actual Luminal A cases are correctly identified)
            - FPR ≈ 0.35 (35% of non-Luminal A cases are wrongly labeled as Luminal A)
        Normal-like best threshold: 0.000
            - TPR ≈ 0.75 (75% of actual Normal-like cases are correctly identified)
            - FPR ≈ 0.55 (55% of non-Normal-like cases are wrongly labeled as Normal-like)

Summary:
    Overall the HistGradientBoostingClassifier model performed worse than the logisitic regression model but better than random forest.
    It was able to correctly identify an equal porportion of Luminal A cases and HER2-enriched cases, 
        however it was more accurate in identifying actual Luminal A cases (better sensitivity)
    It fails to predict Normal-like cases (the minority class) and is very poor at identifying Basal-like cases.
"""
print(summary_hgbc)

print("-"*100)  # Separator

# Logistic Regression (Balanced)
print("---- Logistic Regression (Balanced) ---")

# Initialise the logistic regression model
model_lr = LogisticRegression(
    class_weight='balanced',
    random_state=42)

# Fit the logistic regression model
model_lr.fit(X_train_selected, y_train)

# Predict on the test dataset
y_pred_lr = model_lr.predict(X_test_selected)

# Logistic Regression Model Evaluation
print("\nLogistic Regression (Balanced) Classification Report:\n")
print(classification_report(y_test, y_pred_lr, zero_division=0))

# Logistic Regression (Balanced) Classification Report:

#                precision    recall  f1-score   support

#    Basal-like       0.36      0.45      0.40        11
# HER2-enriched       0.29      0.20      0.24        10
#     Luminal A       0.50      0.55      0.52        20
#   Normal-like       0.00      0.00      0.00         4

#      accuracy                           0.40        45
#     macro avg       0.29      0.30      0.29        45
#  weighted avg       0.37      0.40      0.38        45


# Logistic Regression Confusion Matrix
ordered_labels = ['Basal-like', 'HER2-enriched', 'Luminal A', 'Normal-like']
cm_lr = confusion_matrix(y_test, y_pred_lr)
ax = plt.subplot()
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Figure 3g: Logistic Regression (Balanced) - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
ax.xaxis.set_ticklabels(ordered_labels)
plt.xticks(rotation=45, ha='right')
ax.yaxis.set_ticklabels(ordered_labels)
plt.yticks(rotation=0, ha='right')
plt.grid(visible=True, linestyle='--', linewidth=0.3, color='gray', alpha=0.3)
plt.tight_layout()
plt.savefig("figures/fig3g", dpi=300)
plt.show()


# Logistic Regression ROC AUC Curve
    # As this is multi-class data, we will undertake a one vs. rest approach
    # Binarise multi-class labels
class_names = sorted(y.unique())                                # ['Basal-like', 'HER2-enriched', 'Luminal A', 'Normal-like']
y_test_binarised = label_binarize(y_test, classes=class_names)  # Outputs each subtype as [0 0 1 0] (for Luminal A, for example)
y_prob_lr = model_lr.predict_proba(X_test_selected)             # Outputs predicted probabilities for each subtype
    # Set empty dictionaries for ROC
fpr_lr = {}
tpr_lr = {}
thresholds_lr = {}
roc_auc_lr = {}
optimal_thresholds_lr = {}
j_best_indices_lr = {}
    # Calculate ROC curve and AUC for each subtype
for i, class_label in enumerate(class_names):
    fpr_lr[class_label], tpr_lr[class_label], thresholds_lr[class_label] = roc_curve(y_test_binarised[:, i], y_prob_lr[:, i])
    roc_auc_lr[class_label] = auc(fpr_lr[class_label], tpr_lr[class_label])
    
    # Also find the optimal thresholds using Youden’s J statistic (Youden's J = TPR - FPR)
    j_scores_lr = tpr_lr[class_label] - fpr_lr[class_label]
    j_best_index_lr = np.argmax(j_scores_lr)
    j_best_indices_lr[class_label] = j_best_index_lr
    optimal_thresholds_lr[class_label] = thresholds_lr[class_label][j_best_index_lr]

    # Plot the data
plt.figure(figsize=(10,7))
colors = ['red', 'blue', 'green', 'purple']
for i, class_label in enumerate(class_names):
    plt.plot(
        fpr_lr[class_label], 
        tpr_lr[class_label],
        color=colors[i],
        label=f"{class_label} (AUC = {roc_auc_lr[class_label]:.2f})"
        )
    # Add optimal threshold
    best_idx_lr = j_best_indices_lr[class_label]
    plt.scatter(
        fpr_lr[class_label][best_idx_lr],
        tpr_lr[class_label][best_idx_lr],
        color=colors[i],
        marker='o',
        edgecolors='black',
        label=f"Best threshold for {class_label} = {optimal_thresholds_lr[class_label]:.2f}"
    )
    # Add diagonal line for random chance
plt.plot([0,1], [0,1], 'k--', alpha=0.6)
plt.title("Figure 3h: Multiclass ROC Curve (Logistic Regression (Balanced))")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.grid(visible=True, linestyle='--', linewidth=0.3, color='gray', alpha=0.3)
plt.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig("figures/fig3h", dpi=300)
plt.show()

print("\nOptimal thresholds (Youden's J) for Logistic Regression (Balanced):")
for class_label in class_names:
    print(f"{class_label}: {optimal_thresholds_lr[class_label]:.3f}")


summary_lr = """
Logistic Regression (Balanced) - Conclusions

Classification Report
    Basal-like:
        Precision: 0.36 - poor, only 36% of basal-like cases were correctly identified
        Recall: 0.45 - below average, of the cases identified as basal-like only 45% were correct
        F1-score: 0.4 - below average, due to poor performance on precision and below average performance on recall
    HER2-enriched:
        Precision: 0.29 - poor, only 29% of HER2-enriched cases were correctly identified
        Recall: 0.20 - poor, of the cases identified as HER2-enriched only 20% were correct
        F1-score: 0.24 - poor, due to poor performances on precision and recall
    Luminal A:
        Precision: 0.50 - average, 50% of Luminal A cases were correctly identified
        Recall: 0.55 - above average, of the cases identified as Luminal A 55% were correct
        F1-score: 0.52 - average, due to average performances on precision and recall
    Normal-like:
        Precision: 0 - very poor, no normal-like cases were correctly identified
        Recall: 0 - very poor, model failed to predict any normal-like cases
        F1-score: 0 - very poor, due to very poor performances on precision and recall

Confusion Matrix
    Basal-like (actual) vs. basal-like (predicted): 5 (correct)
    Basal-like (actual) vs. HER2-enriched (predicted): 1 (misclassification)
    Basal-like (actual) vs. Luminal A (predicted): 5 (misclassification)
    Basal-like (actual) vs. Normal-like (predicted): 0

    HER2-enriched (actual) vs. basal-like (predicted): 5 (misclassification)
    HER2-enriched (actual) vs. HER2-enriched (predicted): 2 (correct)
    HER2-enriched (actual) vs. Luminal A (predicted): 3 (misclassification)
    HER2-enriched (actual) vs. Normal-like (predicted): 0

    Luminal A (actual) vs. basal-like (predicted): 4 (misclassification)
    Luminal A (actual) vs. HER2-enriched (predicted): 3 (misclassification)
    Luminal A (actual) vs. Luminal A (predicted): 11 (correct)
    Luminal A (actual) vs. Normal-like (predicted): 2 (misclassification)

    Normal-like (actual) vs. basal-like (predicted): 0
    Normal-like (actual) vs. HER2-enriched (predicted): 1 (misclassification)
    Normal-like (actual) vs. Luminal A (predicted): 3 (misclassification)
    Normal-like (actual) vs. Normal-like (predicted): 0 (none correct)

ROC AUC:
    AUC:
        Basal-like AUC: 0.53 - average separability, slightly better than random guessing
        HER2-enriched AUC: 0.45 - average separability, slightly worse than random guessing
        Luminal A AUC: 0.18 - poor separability
        Normal-like AUC: inf - poor separability
    Thresholds:
        Basal-like best threshold: 0.41
            - TPR ≈ 0.45 (45% of actual Basal-like cases are correctly identified)
            - FPR ≈ 0.25 (25% of non-Basal-like cases are wrongly labeled as Basal-like)
        HER2-enriched best threshold: 0.43
            - TPR ≈ 0.30 (30% of actual HER2-enriched cases are correctly identified)
            - FPR ≈ 0.10 (10% of non-HER2-enriched cases are wrongly labeled as HER2-enriched)
        Luminal A best threshold: 0.28
            - TPR ≈ 0.80 (80% of actual Luminal A cases are correctly identified)
            - FPR ≈ 0.55 (55% of non-Luminal A cases are wrongly labeled as Luminal A)
        Normal-like best threshold: inf
            - TPR ≈ 0 (0% of actual Normal-like cases are correctly identified)
            - FPR ≈ 0 (0% of non-Normal-like cases are wrongly labeled as Normal-like)

Summary:
    Overall the balanced model for logisitic regression performed similarly to the unbalanced model.
"""
print(summary_lr)

print("-"*100)  # Separator

# Random Forest
print("---- Random Forest (Balanced) ---")

# Initialise the random forest model
model_rf = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)

# Fit the random forest model
model_rf.fit(X_train_selected, y_train)

# Predict on the test dataset
y_pred_rf = model_rf.predict(X_test_selected)

# Random Forest Model Evaluation
print("\nRandom Forest (Balanced) Classification Report:\n")
print(classification_report(y_test, y_pred_rf, zero_division=0))

#                precision    recall  f1-score   support

#    Basal-like       0.33      0.09      0.14        11
# HER2-enriched       0.00      0.00      0.00        10
#     Luminal A       0.46      0.95      0.62        20
#   Normal-like       0.00      0.00      0.00         4

#      accuracy                           0.44        45
#     macro avg       0.20      0.26      0.19        45
#  weighted avg       0.29      0.44      0.31        45

# Random Forest Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
ax = plt.subplot()
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Figure 3i: Random Forest (Balanced) - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
ax.xaxis.set_ticklabels(ordered_labels)
plt.xticks(rotation=45, ha='right')
ax.yaxis.set_ticklabels(ordered_labels)
plt.yticks(rotation=0, ha='right')
plt.grid(visible=True, linestyle='--', linewidth=0.3, color='gray', alpha=0.3)
plt.tight_layout()
plt.savefig("figures/fig3i", dpi=300)
plt.show()

# Random Forest ROC AUC Curve
y_prob_rf = model_rf.predict_proba(X_test_selected)       
    # Set empty dictionaries for ROC
fpr_rf = {}
tpr_rf = {}
thresholds_rf = {}
roc_auc_rf = {}
optimal_thresholds_rf = {}
j_best_indices_rf = {}
    # Calculate ROC curve and AUC for each subtype
for i, class_label in enumerate(class_names):
    fpr_rf[class_label], tpr_rf[class_label], thresholds_rf[class_label] = roc_curve(y_test_binarised[:, i], y_prob_rf[:, i])
    roc_auc_rf[class_label] = auc(fpr_rf[class_label], tpr_rf[class_label])
    
    # Also find the optimal thresholds using Youden’s J statistic (Youden's J = TPR - FPR)
    j_scores_rf = tpr_rf[class_label] - fpr_rf[class_label]
    j_best_index_rf = np.argmax(j_scores_rf)
    j_best_indices_rf[class_label] = j_best_index_rf
    optimal_thresholds_rf[class_label] = thresholds_rf[class_label][j_best_index_rf]

    # Plot the data
plt.figure(figsize=(10,7))
colors = ['red', 'blue', 'green', 'purple']
for i, class_label in enumerate(class_names):
    plt.plot(
        fpr_rf[class_label], 
        tpr_rf[class_label],
        color=colors[i],
        label=f"{class_label} (AUC = {roc_auc_rf[class_label]:.2f})"
        )
    # Add optimal threshold
    best_idx_rf = j_best_indices_rf[class_label]
    plt.scatter(
        fpr_rf[class_label][best_idx_rf],
        tpr_rf[class_label][best_idx_rf],
        color=colors[i],
        marker='o',
        edgecolors='black',
        label=f"Best threshold for {class_label} = {optimal_thresholds_rf[class_label]:.2f}"
    )
    # Add diagonal line for random chance
plt.plot([0,1], [0,1], 'k--', alpha=0.6)
plt.title("Figure 3j: Multiclass ROC Curve (Random Forest (Balanced))")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.grid(visible=True, linestyle='--', linewidth=0.3, color='gray', alpha=0.3)
plt.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig("figures/fig3j", dpi=300)
plt.show()

print("\nOptimal thresholds (Youden's J) for Random Forest (Balanced):")
for class_label in class_names:
    print(f"{class_label}: {optimal_thresholds_rf[class_label]:.3f}")


summary_rf = """
Random Forest (Balanced) - Conclusions

Classification Report
    Basal-like:
        Precision: 0.33 - poor, only 33% of basal-like cases were correctly identified
        Recall: 0.09 - very poor, of the cases identified as basal-like only 9% were correct
        F1-score: 0.14 - poor, due to poor performances on precision and recall
    HER2-enriched:
        Precision: 0 - very poor, no HER2-enriched cases were correctly identified
        Recall: 0 - very poor, model failed to predict any HER2-enriched cases
        F1-score: 0 - very poor, due to very poor performances on precision and recall
    Luminal A:
        Precision: 0.46 - below average, 46% of Luminal A cases were correctly identified
        Recall: 0.95 - excellent, of the cases identified as Luminal A 95% were correct
        F1-score: 0.62 - above average, due to the below average performance on precision but excellent performance on recall
    Normal-like:
        Precision: 0 - very poor, no normal-like cases were correctly identified
        Recall: 0 - very poor, model failed to predict any normal-like cases
        F1-score: 0 - very poor, due to very poor performances on precision and recall

Confusion Matrix
    Basal-like (actual) vs. basal-like (predicted): 1 (correct)
    Basal-like (actual) vs. HER2-enriched (predicted): 0
    Basal-like (actual) vs. Luminal A (predicted): 10 (misclassification)
    Basal-like (actual) vs. Normal-like (predicted): 0

    HER2-enriched (actual) vs. basal-like (predicted): 2 (misclassification)
    HER2-enriched (actual) vs. HER2-enriched (predicted): 0 (none correct)
    HER2-enriched (actual) vs. Luminal A (predicted): 8 (misclassification)
    HER2-enriched (actual) vs. Normal-like (predicted): 0

    Luminal A (actual) vs. basal-like (predicted): 0 
    Luminal A (actual) vs. HER2-enriched (predicted): 1 (misclassification)
    Luminal A (actual) vs. Luminal A (predicted): 19 (correct)
    Luminal A (actual) vs. Normal-like (predicted): 0

    Normal-like (actual) vs. basal-like (predicted): 0
    Normal-like (actual) vs. HER2-enriched (predicted): 0
    Normal-like (actual) vs. Luminal A (predicted): 4 (misclassification)
    Normal-like (actual) vs. Normal-like (predicted): 0 (none correct)

ROC AUC:
    AUC:
        Basal-like AUC: 0.36 - poor separability
        HER2-enriched AUC: 0.35 - poor separability
        Luminal A AUC: 0.39 - poor separability
        Normal-like AUC: 0.41 - poor separability
    Thresholds:
        Basal-like best threshold: 0.42
            - TPR ≈ 0.10 (10% of actual Basal-like cases are correctly identified)
            - FPR ≈ 0.05 (5% of non-Basal-like cases are wrongly labeled as Basal-like)
        HER2-enriched best threshold: 0.16
            - TPR ≈ 0.90 (90% of actual HER2-enriched cases are correctly identified)
            - FPR ≈ 0.85 (85% of non-HER2-enriched cases are wrongly labeled as HER2-enriched)
        Luminal A best threshold: 0.41
            - TPR ≈ 0.85 (85% of actual Luminal A cases are correctly identified)
            - FPR ≈ 0.75 (75% of non-Luminal A cases are wrongly labeled as Luminal A)
        Normal-like best threshold: 0.03
            - TPR ≈ 1.00 (100% of actual Normal-like cases are correctly identified)
            - FPR ≈ 0.75 (75% of non-Normal-like cases are wrongly labeled as Normal-like)

Summary:
    Overall the balanced random forest model performed slightly better than the unbalanced model.
"""
print(summary_rf)

print("-"*100)  # Separator

# HistGradientBoostingClassifier (Balanced)
print("---- HistGradientBoostingClassifier (Balanced) ---")

# Initialise the HistGradientBoostingClassifier model
model_hgbc = HistGradientBoostingClassifier(
    class_weight='balanced',
    random_state=42
    )

# Fit the HistGradientBoostingClassifier model
model_hgbc.fit(X_train_selected, y_train)

# Predict on the test dataset
y_pred_hgbc = model_hgbc.predict(X_test_selected)

# HistGradientBoostingClassifier Model Evaluation
print("\nHistGradientBoostingClassifier (Balanced) Classification Report:\n")
print(classification_report(y_test, y_pred_hgbc, zero_division=0))

# HistGradientBoostingClassifier (Balanced) Classification Report:

#                precision    recall  f1-score   support

#    Basal-like       0.10      0.09      0.10        11
# HER2-enriched       0.33      0.10      0.15        10
#     Luminal A       0.42      0.65      0.51        20
#   Normal-like       0.00      0.00      0.00         4

#      accuracy                           0.33        45
#     macro avg       0.21      0.21      0.19        45
#  weighted avg       0.28      0.33      0.28        45

# HistGradientBoostingClassifier Confusion Matrix
cm_hgbc = confusion_matrix(y_test, y_pred_hgbc)
ax = plt.subplot()
sns.heatmap(cm_hgbc, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Figure 3j: HistGradientBoostingClassifier (Balanced) - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
ax.xaxis.set_ticklabels(ordered_labels)
plt.xticks(rotation=45, ha='right')
ax.yaxis.set_ticklabels(ordered_labels)
plt.yticks(rotation=0, ha='right')
plt.grid(visible=True, linestyle='--', linewidth=0.3, color='gray', alpha=0.3)
plt.tight_layout()
plt.savefig("figures/fig3k", dpi=300)
plt.show()

# HistGradientBoostingClassifier ROC AUC Curve 
y_prob_hgbc = model_hgbc.predict_proba(X_test_selected)       
    # Set empty dictionaries for ROC
fpr_hgbc = {}
tpr_hgbc = {}
thresholds_hgbc = {}
roc_auc_hgbc = {}
optimal_thresholds_hgbc = {}
j_best_indices_hgbc = {}
    # Calculate ROC curve and AUC for each subtype
for i, class_label in enumerate(class_names):
    fpr_hgbc[class_label], tpr_hgbc[class_label], thresholds_hgbc[class_label] = roc_curve(y_test_binarised[:, i], y_prob_hgbc[:, i])
    roc_auc_hgbc[class_label] = auc(fpr_hgbc[class_label], tpr_hgbc[class_label])
    
    # Also find the optimal thresholds using Youden’s J statistic (Youden's J = TPR - FPR)
    j_scores_hgbc = tpr_hgbc[class_label] - fpr_hgbc[class_label]
    j_best_index_hgbc = np.argmax(j_scores_hgbc)
    j_best_indices_hgbc[class_label] = j_best_index_hgbc
    optimal_thresholds_hgbc[class_label] = thresholds_hgbc[class_label][j_best_index_hgbc]

    # Plot the data
plt.figure(figsize=(10,7))
colors = ['red', 'blue', 'green', 'purple']
for i, class_label in enumerate(class_names):
    plt.plot(
        fpr_hgbc[class_label], 
        tpr_hgbc[class_label],
        color=colors[i],
        label=f"{class_label} (AUC = {roc_auc_hgbc[class_label]:.2f})"
        )
    # Add optimal threshold
    best_idx_hgbc = j_best_indices_hgbc[class_label]
    plt.scatter(
        fpr_hgbc[class_label][best_idx_hgbc],
        tpr_hgbc[class_label][best_idx_hgbc],
        color=colors[i],
        marker='o',
        edgecolors='black',
        label=f"Best threshold for {class_label} = {optimal_thresholds_hgbc[class_label]:.2f}"
    )
    # Add diagonal line for random chance
plt.plot([0,1], [0,1], 'k--', alpha=0.6)
plt.title("Figure 3k: Multiclass ROC Curve (HistGradientBoostingClassifier (Balanced))")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.grid(visible=True, linestyle='--', linewidth=0.3, color='gray', alpha=0.3)
plt.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig("figures/fig3l", dpi=300)
plt.show()

print("\nOptimal thresholds (Youden's J) for HistGradientBoostingClassifier (Balanced):")
for class_label in class_names:
    print(f"{class_label}: {optimal_thresholds_hgbc[class_label]:.3f}")

summary_hgbc = """
HistGradientBoostingClassifier (Balanced) - Conclusions

Classification Report
    Basal-like:
        Precision: 0.10 - very poor, only 10% of basal-like cases were correctly identified
        Recall: 0.09 - very poor, of the cases identified as basal-like only 9% were correct
        F1-score: 0.10 - very poor, due to very poor performances on precision and recall
    HER2-enriched:
        Precision: 0.33 - poor, 33% of HER2-enriched cases were correctly identified
        Recall: 0.10 - very poor, of the cases identified as HER2-enriched only 10% were correct
        F1-score: 0.15 - very poor, due to poor performances on precision and recall
    Luminal A:
        Precision: 0.42 - below average, 42% of Luminal A cases were correctly identified
        Recall: 0.65 - above average, of the cases identified as Luminal A 65% were correct
        F1-score: 0.51 - above average, due to the below average performance on precision but above average performance on recall
    Normal-like:
        Precision: 0 - very poor, no normal-like cases were correctly identified
        Recall: 0 - very poor, model failed to predict any normal-like cases
        F1-score: 0 - very poor, due to very poor performances on precision and recall

Confusion Matrix
    Basal-like (actual) vs. basal-like (predicted): 1 (correct)
    Basal-like (actual) vs. HER2-enriched (predicted): 1 (misclassification)
    Basal-like (actual) vs. Luminal A (predicted): 9 (misclassification)
    Basal-like (actual) vs. Normal-like (predicted): 0

    HER2-enriched (actual) vs. basal-like (predicted): 4 (misclassification)
    HER2-enriched (actual) vs. HER2-enriched (predicted): 1 (correct)
    HER2-enriched (actual) vs. Luminal A (predicted): 5 (misclassification)
    HER2-enriched (actual) vs. Normal-like (predicted): 0

    Luminal A (actual) vs. basal-like (predicted): 5 (misclassification) 
    Luminal A (actual) vs. HER2-enriched (predicted): 1 (misclassification) 
    Luminal A (actual) vs. Luminal A (predicted): 13 (correct)
    Luminal A (actual) vs. Normal-like (predicted): 1 (misclassification)

    Normal-like (actual) vs. basal-like (predicted): 0
    Normal-like (actual) vs. HER2-enriched (predicted): 0 
    Normal-like (actual) vs. Luminal A (predicted): 4 (misclassification)
    Normal-like (actual) vs. Normal-like (predicted): 0 (none correct)

ROC AUC:
    AUC:
        Basal-like AUC: 0.41 - below average separability
        HER2-enriched AUC: 0.57 - above average separability
        Luminal A AUC: 0.54 - above average separability
        Normal-like AUC: 0.47 - below average separability
    Thresholds:
        Basal-like best threshold: 0.002
            - TPR ≈ 1.00 (100% of actual Basal-like cases are correctly identified)
            - FPR ≈ 0.80 (80% of non-Basal-like cases are wrongly labeled as Basal-like)
        HER2-enriched best threshold: 0.057
            - TPR ≈ 0.70 (70% of actual HER2-enriched cases are correctly identified)
            - FPR ≈ 0.45 (45% of non-HER2-enriched cases are wrongly labeled as HER2-enriched)
        Luminal A best threshold: 0.83
            - TPR ≈ 0.50 (50% of actual Luminal A cases are correctly identified)
            - FPR ≈ 0.25 (25% of non-Luminal A cases are wrongly labeled as Luminal A)
        Normal-like best threshold: 0.000
            - TPR ≈ 0.75 (75% of actual Normal-like cases are correctly identified)
            - FPR ≈ 0.55 (55% of non-Normal-like cases are wrongly labeled as Normal-like)

Summary:
    Overall the balanced HistGradientBoostingClassifier model performed slightly worse than the unbalanced model.
"""
print(summary_hgbc)

print("-"*100)  # Separator

# 4. Hyperparamter Tuning
print("---------- 4. Hyperparamter Tuning ----------")

"""
Rationale:
- As this is a learning exercise using synthetic data which seems to be difficult for predictions, the subsequent sections 
    will serve as a proof-of-principle of a suggested workflow
- We shall use GridSearchCV to test several parameters to improve our model
- The model we shall test is the HistGradientBoostingClassifier (Unbalanced) model, as:
    - It has decent F1 scores comparatively (Basal-like: 0.10, HER2-enriched: 0.29, Luminal A: 0.56, Normal-like: 0.00)
    - It is more easily tunable than Logistic Regression
"""

# Re-Initialise the HistGradientBoostingClassifier model
model_hgbc = HistGradientBoostingClassifier(random_state=42)

# Re-Fit the HistGradientBoostingClassifier model
model_hgbc.fit(X_train_selected, y_train)

# Predict on the test dataset
y_pred_hgbc = model_hgbc.predict(X_test_selected)

# Set some features to be tested
param_grid = {
    'max_iter': [50, 100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [None, 3, 5, 10],
    'min_samples_leaf': [5, 10, 20]
}

for param, values in param_grid.items():
    print(f"    {param}: {values}")

# Initialise GridSearchCV
grid_search = GridSearchCV(
    estimator=model_hgbc,
    param_grid=param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='f1_macro',     # Over 'accuracy' due to imbalanced data
    verbose=1,
    n_jobs=-1
)
print(grid_search)

print("\n--- Starting Grid Search (this may take some time)... ---")
grid_search.fit(X_train_selected, y_train)
print("\n--- Grid Search Completed ---")

print(f"Best parameters found: {grid_search.best_params_}")
# Best parameters found: {'learning_rate': 0.1, 'max_depth': None, 'max_iter': 200, 'min_samples_leaf': 20}
print(f"Best cross-validation F1 (macro avg): {grid_search.best_score_:.4f}")   # 0.5638

best_hgbc_model = grid_search.best_estimator_
print(f"\nBest model (trained with best params): {best_hgbc_model}")    # HistGradientBoostingClassifier(max_iter=200, random_state=42)

y_pred_best_hgbc = best_hgbc_model.predict(X_test_selected)
test_f1_macro = f1_score(y_test, y_pred_best_hgbc, average='macro')
print(f"F1 (macro avg) on the test set: {test_f1_macro:.4f}")                   # 0.2268

summary_hgbc_hyperparameter = """
The model performed best with the following parameters:
- Learning rate: 0.1
- Max Depth: None
- Max Iterations: 200
- Minimum number of samples per leaf: 20

On the training data, the 'best' model achieved an accuracy of 0.65638.
On the unseen, test data, the 'best' model achieved an accuracy of 0.2268.

Overall, the model performs above average on the training data, but struggles on new, unseen data.

Despite hyperparameter tuning, this dataset is difficult to predict, due to:
- Its imbalanced nature.
- Its synthetic generation, meaning that true, meaningful relationships are absent and, instead, the data is random.
"""
print(summary_hgbc_hyperparameter)

print("-"*100)  # Separator

# 5. Saving and Loading
print("---------- 5. Saving and Loading ----------")

joblib.dump(best_hgbc_model, "model/best_hgbc_model.joblib")
# To load later:
# loaded_model = joblib.load("model/best_hgbc_model.joblib")

overall_summary = """
Due to this dataset being generated with random numbers, it does not truly reflect the patterns that would be captured with real world data.
As a result, it is very difficult to train any model on what is, effectively, noise.
Despite this, we have shown, as a comprehensive proof-of-principal for a machine learning workflow on a multi-class classification problem, the workflow that can be undertaken to:
- Generate random data
- Apply several models
- Tune the parameters of these models
- Save the best model
Conclusions:
 -Despite implementing standard techniques such as feature selection (SelectKBest), 
    managing class imbalance (through `stratify=y` during data splitting and `class_weight='balanced'` in models), 
    and advanced hyperparameter tuning (using `GridSearchCV` with `f1_macro` scoring and `StratifiedKFold` for 
    robust evaluation), the models consistently exhibited poor performance (low precision, recall, and F1-scores).
- The model's ability to consistently achieve the 'best' scores for Luminal A classification is, most likely, due to it being the dominant class.
In future:
- This workflow can be applied to a realistic dataset to see whether any of these models can accurately predict cancer subtypes
- Functions can be written to consolidate the amount of code required
"""
print(overall_summary)
