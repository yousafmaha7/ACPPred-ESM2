# ============================================================
# MLP + AdaBoost using ESM-2 Features for ACP Classification
# ============================================================
#
# Purpose:
# 1. Load the SAME 80/20 train-test datasets used in the
#    original AdaBoost-ESM2 experiment.
#
# 2. Reproduce the existing AdaBoost-ESM2 workflow:
#       - 5-fold GridSearchCV
#       - scoring = accuracy
#       - train only
#
# 3. Train an MLP classifier using the SAME ESM-2 features.
#
# 4. Perform 5-fold GridSearchCV for MLP hyperparameter
#    optimization using ONLY the training data.
#
# 5. Perform additional 10-fold CV for BOTH models using
#    identical folds.
#
# 6. Evaluate both final models on the untouched 20% test set.
#
# 7. Save Accuracy, F1, MCC, ROC-AUC, sensitivity,
#    specificity and fold-wise CV results.
#
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

import matplotlib.pyplot as plt


# ============================================================
# 2. SET RANDOM SEED
# ============================================================

RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)


# ============================================================
# 3. LOAD DATA
# ============================================================

TRAIN_FILE = "acp_train_esm2_features.csv"
TEST_FILE = "acp_test_esm2_features.csv"

print("\n==============================================")
print("Loading datasets")
print("==============================================")

train_data = pd.read_csv(TRAIN_FILE)
test_data = pd.read_csv(TEST_FILE)

print("Training dataset shape:", train_data.shape)
print("Testing dataset shape :", test_data.shape)


# ============================================================
# 4. SEPARATE LABELS AND FEATURES
# ============================================================
#
# Your original notebook uses:
#   column 3  -> Label
#   column 4+ -> ESM-2 features
#
# Python indexing:
#   iloc[:, 2] -> third column
#   iloc[:, 3:] -> fourth column onward
#
# We preserve this structure exactly.
# ============================================================

y_train = train_data.iloc[:, 2]
X_train = train_data.iloc[:, 3:]

y_test = test_data.iloc[:, 2]
X_test = test_data.iloc[:, 3:]


print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))
print("Number of ESM-2 features:", X_train.shape[1])

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())


# ============================================================
# 5. CHECK FOR MISSING VALUES
# ============================================================

if X_train.isnull().sum().sum() > 0:
    raise ValueError("Missing values detected in X_train.")

if X_test.isnull().sum().sum() > 0:
    raise ValueError("Missing values detected in X_test.")

if y_train.isnull().sum() > 0:
    raise ValueError("Missing labels detected in y_train.")

if y_test.isnull().sum() > 0:
    raise ValueError("Missing labels detected in y_test.")


# ============================================================
# 6. CHECK FEATURE MATCHING
# ============================================================

if list(X_train.columns) != list(X_test.columns):
    raise ValueError(
        "Training and testing feature columns do not match."
    )


# ============================================================
# 7. DEFINE 5-FOLD CV FOR HYPERPARAMETER OPTIMIZATION
# ============================================================
#
# IMPORTANT:
# This is the same role as the 5-fold GridSearchCV in your
# original AdaBoost experiment.
#
# Hyperparameter selection occurs ONLY inside X_train/y_train.
# The independent test set is NOT touched.
# ============================================================

cv_5fold = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)


# ============================================================
# 8. ADA BOOST MODEL
# ============================================================

print("\n==============================================")
print("ADA BOOST - HYPERPARAMETER OPTIMIZATION")
print("==============================================")

base_estimator = DecisionTreeClassifier(
    max_depth=1,
    random_state=RANDOM_STATE
)

adaboost_model = AdaBoostClassifier(
    estimator=base_estimator,
    random_state=RANDOM_STATE
)


# ------------------------------------------------------------
# Original AdaBoost hyperparameter grid from your notebook
# ------------------------------------------------------------

adaboost_param_grid = {
    "n_estimators": [
        50, 200, 400, 500, 600, 800, 1000
    ],

    "learning_rate": [
        0.01, 0.1, 1.0
    ],

    "algorithm": [
        "SAMME"
    ],

    "estimator__max_depth": [
        1, 3, 5, 7, 9
    ],

    "estimator__max_features": [
        "sqrt",
        "log2"
    ]
}


# ============================================================
# 9. ADA BOOST GRID SEARCH
# ============================================================

ada_grid = GridSearchCV(
    estimator=adaboost_model,
    param_grid=adaboost_param_grid,
    scoring="accuracy",
    cv=cv_5fold,
    n_jobs=-1,
    verbose=1
)

ada_grid.fit(X_train, y_train)


# ============================================================
# 10. BEST ADA BOOST MODEL
# ============================================================

best_adaboost_model = ada_grid.best_estimator_

print("\nBest AdaBoost parameters:")
print(ada_grid.best_params_)

print(
    "\nBest AdaBoost 5-fold CV accuracy:",
    ada_grid.best_score_
)


# Save AdaBoost model
joblib.dump(
    best_adaboost_model,
    "best_adaboost_esm2_model_new.pkl"
)


# Save AdaBoost hyperparameters
with open(
    "best_adaboost_esm2_parameters_new.txt",
    "w"
) as f:

    f.write("Best AdaBoost Hyperparameters\n")
    f.write("============================\n")

    for parameter, value in ada_grid.best_params_.items():
        f.write(
            f"{parameter}: {value}\n"
        )


# ============================================================
# 11. MLP MODEL
# ============================================================
#
# We use StandardScaler INSIDE the pipeline.
#
# This is important:
#
# Scaling is independently fitted inside each training fold.
# Therefore validation/test information cannot leak into the
# training process.
# ============================================================

print("\n==============================================")
print("MLP - HYPERPARAMETER OPTIMIZATION")
print("==============================================")


mlp_pipeline = Pipeline(
    [
        (
            "scaler",
            StandardScaler()
        ),

        (
            "mlp",
            MLPClassifier(
                random_state=RANDOM_STATE,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20
            )
        )
    ]
)


# ============================================================
# 12. MLP HYPERPARAMETER GRID
# ============================================================
#
# These are intentionally moderate architectures because the
# dataset is relatively small and ESM-2 already provides rich
# pretrained representations.
#
# Hidden layers:
#   (128,)
#   (256,)
#   (128,64)
#   (256,128)
#
# ============================================================

mlp_param_grid = {

    "mlp__hidden_layer_sizes": [
        (128,),
        (256,),
        (128, 64),
        (256, 128)
    ],

    "mlp__activation": [
        "relu"
    ],

    "mlp__alpha": [
        0.0001,
        0.001,
        0.01
    ],

    "mlp__learning_rate_init": [
        0.001,
        0.0001
    ],

    "mlp__batch_size": [
        16,
        32,
        64
    ]
}


# ============================================================
# 13. MLP GRID SEARCH
# ============================================================

mlp_grid = GridSearchCV(
    estimator=mlp_pipeline,
    param_grid=mlp_param_grid,
    scoring="accuracy",
    cv=cv_5fold,
    n_jobs=-1,
    verbose=1
)

mlp_grid.fit(X_train, y_train)


# ============================================================
# 14. BEST MLP MODEL
# ============================================================

best_mlp_model = mlp_grid.best_estimator_

print("\nBest MLP parameters:")
print(mlp_grid.best_params_)

print(
    "\nBest MLP 5-fold CV accuracy:",
    mlp_grid.best_score_
)


# Save MLP
joblib.dump(
    best_mlp_model,
    "best_mlp_esm2_model.pkl"
)


# Save MLP hyperparameters
with open(
    "best_mlp_esm2_parameters.txt",
    "w"
) as f:

    f.write("Best MLP Hyperparameters\n")
    f.write("=======================\n")

    for parameter, value in mlp_grid.best_params_.items():
        f.write(
            f"{parameter}: {value}\n"
        )


# ============================================================
# 15. DEFINE IDENTICAL 10-FOLD CV
# ============================================================
#
# IMPORTANT:
#
# Both models MUST use exactly the same folds.
#
# This is important because later we may perform paired
# statistical comparisons.
#
# shuffle=False reproduces sklearn's default behavior when
# cross_val_score(..., cv=10) is used with a classifier.
# ============================================================

cv_10fold = StratifiedKFold(
    n_splits=10,
    shuffle=False
)


# ============================================================
# 16. SCORING METRICS
# ============================================================

scoring_metrics = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}


# ============================================================
# 17. ADDITIONAL 10-FOLD CV - ADABOOST
# ============================================================

print("\n==============================================")
print("10-FOLD CROSS-VALIDATION - ADABOOST")
print("==============================================")


ada_cv_results = cross_validate(
    best_adaboost_model,
    X_train,
    y_train,
    cv=cv_10fold,
    scoring=scoring_metrics,
    n_jobs=-1,
    return_train_score=False
)


# ============================================================
# 18. ADDITIONAL 10-FOLD CV - MLP
# ============================================================

print("\n==============================================")
print("10-FOLD CROSS-VALIDATION - MLP")
print("==============================================")


mlp_cv_results = cross_validate(
    best_mlp_model,
    X_train,
    y_train,
    cv=cv_10fold,
    scoring=scoring_metrics,
    n_jobs=-1,
    return_train_score=False
)


# ============================================================
# 19. DISPLAY FOLD-WISE RESULTS
# ============================================================

print("\n==============================================")
print("FOLD-WISE RESULTS")
print("==============================================")


for metric in scoring_metrics.keys():

    print(f"\n{metric.upper()}")

    print(
        "AdaBoost:",
        np.round(
            ada_cv_results[
                "test_" + metric
            ],
            4
        )
    )

    print(
        "MLP     :",
        np.round(
            mlp_cv_results[
                "test_" + metric
            ],
            4
        )
    )


# ============================================================
# 20. SUMMARY OF 10-FOLD RESULTS
# ============================================================

summary_rows = []


for model_name, results in [
    ("AdaBoost-ESM2", ada_cv_results),
    ("MLP-ESM2", mlp_cv_results)
]:

    for metric in scoring_metrics.keys():

        values = results[
            "test_" + metric
        ]

        summary_rows.append(
            {
                "Model": model_name,
                "Metric": metric,
                "Mean": np.mean(values),
                "SD": np.std(values, ddof=1)
            }
        )


cv_summary = pd.DataFrame(
    summary_rows
)


print("\n==============================================")
print("10-FOLD CV SUMMARY")
print("==============================================")

print(
    cv_summary.to_string(
        index=False
    )
)


cv_summary.to_csv(
    "MLP_AdaBoost_10fold_CV_summary.csv",
    index=False
)


# ============================================================
# 21. SAVE FOLD-WISE RESULTS
# ============================================================

fold_results = pd.DataFrame(
    {
        "Fold": np.arange(1, 11),

        "AdaBoost_Accuracy":
            ada_cv_results["test_accuracy"],

        "MLP_Accuracy":
            mlp_cv_results["test_accuracy"],

        "AdaBoost_Precision":
            ada_cv_results["test_precision"],

        "MLP_Precision":
            mlp_cv_results["test_precision"],

        "AdaBoost_Recall":
            ada_cv_results["test_recall"],

        "MLP_Recall":
            mlp_cv_results["test_recall"],

        "AdaBoost_F1":
            ada_cv_results["test_f1"],

        "MLP_F1":
            mlp_cv_results["test_f1"],

        "AdaBoost_ROC_AUC":
            ada_cv_results["test_roc_auc"],

        "MLP_ROC_AUC":
            mlp_cv_results["test_roc_auc"]
    }
)


fold_results.to_csv(
    "MLP_AdaBoost_10fold_foldwise_results.csv",
    index=False
)


# ============================================================
# 22. FUNCTION FOR TEST-SET EVALUATION
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    model_name
):

    print("\n")
    print("==============================================")
    print(model_name)
    print("TEST SET PERFORMANCE")
    print("==============================================")


    # Predictions
    y_pred = model.predict(X_test)

    # Probability scores
    y_prob = model.predict_proba(
        X_test
    )[:, 1]


    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    tn, fp, fn, tp = cm.ravel()


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    mcc = matthews_corrcoef(
        y_test,
        y_pred
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )


    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        f"Accuracy    : {accuracy:.4f}"
    )

    print(
        f"Precision   : {precision:.4f}"
    )

    print(
        f"Recall      : {recall:.4f}"
    )

    print(
        f"F1 Score    : {f1:.4f}"
    )

    print(
        f"MCC         : {mcc:.4f}"
    )

    print(
        f"ROC-AUC     : {roc_auc:.4f}"
    )

    print(
        f"Sensitivity : {sensitivity:.4f}"
    )

    print(
        f"Specificity : {specificity:.4f}"
    )

    print("\nConfusion Matrix:")
    print(cm)


    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            digits=4,
            zero_division=0
        )
    )


    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MCC": mcc,
        "ROC_AUC": roc_auc,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }, y_pred, y_prob


# ============================================================
# 23. FINAL TEST-SET EVALUATION
# ============================================================
#
# IMPORTANT:
#
# The test set is used ONLY here.
#
# It was not used during:
#   - AdaBoost GridSearchCV
#   - MLP GridSearchCV
#   - 10-fold CV
#
# ============================================================

ada_test_results, ada_pred, ada_prob = evaluate_model(
    best_adaboost_model,
    X_test,
    y_test,
    "AdaBoost-ESM2"
)


mlp_test_results, mlp_pred, mlp_prob = evaluate_model(
    best_mlp_model,
    X_test,
    y_test,
    "MLP-ESM2"
)


# ============================================================
# 24. SAVE TEST RESULTS
# ============================================================

test_results = pd.DataFrame(
    [
        ada_test_results,
        mlp_test_results
    ]
)

test_results.to_csv(
    "MLP_AdaBoost_test_results.csv",
    index=False
)


# ============================================================
# 25. SAVE TEST PREDICTIONS
# ============================================================

test_predictions = pd.DataFrame(
    {
        "True_Label": y_test.values,

        "AdaBoost_Prediction":
            ada_pred,

        "AdaBoost_Probability":
            ada_prob,

        "MLP_Prediction":
            mlp_pred,

        "MLP_Probability":
            mlp_prob
    }
)


# If an ID column exists, preserve it
if "ID" in test_data.columns:

    test_predictions.insert(
        0,
        "ID",
        test_data["ID"].values
    )


test_predictions.to_csv(
    "MLP_AdaBoost_test_predictions.csv",
    index=False
)


# ============================================================
# 26. ROC CURVES
# ============================================================

fpr_ada, tpr_ada, _ = roc_curve(
    y_test,
    ada_prob
)

fpr_mlp, tpr_mlp, _ = roc_curve(
    y_test,
    mlp_prob
)


auc_ada = auc(
    fpr_ada,
    tpr_ada
)

auc_mlp = auc(
    fpr_mlp,
    tpr_mlp
)


plt.figure(
    figsize=(8, 6)
)

plt.plot(
    fpr_ada,
    tpr_ada,
    label=f"AdaBoost-ESM2 (AUC = {auc_ada:.3f})"
)

plt.plot(
    fpr_mlp,
    tpr_mlp,
    label=f"MLP-ESM2 (AUC = {auc_mlp:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curves: AdaBoost-ESM2 vs MLP-ESM2"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "MLP_AdaBoost_ROC_comparison.png",
    dpi=300
)

plt.close()


# ============================================================
# 27. FINAL SUMMARY
# ============================================================

print("\n")
print("==========================================================")
print("FINAL COMPARISON")
print("==========================================================")

print(
    test_results[
        [
            "Model",
            "Accuracy",
            "F1",
            "MCC",
            "ROC_AUC",
            "Sensitivity",
            "Specificity"
        ]
    ].to_string(index=False)
)


print("\n==========================================================")
print("FILES GENERATED")
print("==========================================================")

print(
    "1. best_adaboost_esm2_model_new.pkl"
)

print(
    "2. best_mlp_esm2_model.pkl"
)

print(
    "3. best_adaboost_esm2_parameters_new.txt"
)

print(
    "4. best_mlp_esm2_parameters.txt"
)

print(
    "5. MLP_AdaBoost_10fold_CV_summary.csv"
)

print(
    "6. MLP_AdaBoost_10fold_foldwise_results.csv"
)

print(
    "7. MLP_AdaBoost_test_results.csv"
)

print(
    "8. MLP_AdaBoost_test_predictions.csv"
)

print(
    "9. MLP_AdaBoost_ROC_comparison.png"
)

print("\nAnalysis completed successfully.")
