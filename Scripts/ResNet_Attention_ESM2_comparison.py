# ============================================================
# RESNET-ESM2 + ATTENTION-ESM2 ACP CLASSIFICATION
# ============================================================
#
# Purpose:
#
# 1. Load the SAME train/test datasets used in the original
#    AdaBoost-ESM2 and MLP-ESM2 experiments.
#
# 2. Keep the SAME 80/20 independent train-test split.
#
# 3. Train a lightweight Residual Neural Network classifier
#    using ESM-2 embeddings.
#
# 4. Train an Attention-based classifier using the same
#    ESM-2 embeddings.
#
# 5. Hyperparameter optimization is performed ONLY on the
#    training dataset.
#
# 6. 5-fold stratified CV is used for model selection.
#
# 7. An additional 10-fold CV is performed on the training
#    data for comparison with the existing analysis.
#
# 8. The untouched 20% independent test set is used only
#    for final evaluation.
#
# 9. Report:
#       Accuracy
#       Precision
#       Recall / Sensitivity
#       Specificity
#       F1
#       MCC
#       ROC-AUC
#
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import random
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# 2. PYTORCH
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import (
    TensorDataset,
    DataLoader
)


# ============================================================
# 3. SCIKIT-LEARN
# ============================================================

from sklearn.model_selection import (
    StratifiedKFold,
    ParameterGrid
)

from sklearn.preprocessing import StandardScaler

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


# ============================================================
# 4. RANDOM SEEDS
# ============================================================

RANDOM_STATE = 42

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\n================================================")
print("DEVICE")
print("================================================")

print(DEVICE)


# ============================================================
# 5. DATA FILES
# ============================================================

TRAIN_FILE = "acp_train_esm2_features.csv"
TEST_FILE = "acp_test_esm2_features.csv"


# ============================================================
# 6. LOAD DATA
# ============================================================

print("\n================================================")
print("LOADING DATA")
print("================================================")

train_data = pd.read_csv(TRAIN_FILE)
test_data = pd.read_csv(TEST_FILE)

print(
    "Training dataset shape:",
    train_data.shape
)

print(
    "Testing dataset shape :",
    test_data.shape
)


# ============================================================
# 7. EXTRACT LABELS AND ESM-2 FEATURES
# ============================================================
#
# Same column structure as your original workflow:
#
# Third column  = label
# Fourth onward = ESM-2 features
#
# ============================================================

y_train = train_data.iloc[:, 2].astype(int).values

X_train = train_data.iloc[:, 3:].astype(float).values

y_test = test_data.iloc[:, 2].astype(int).values

X_test = test_data.iloc[:, 3:].astype(float).values


print("\nNumber of training samples:", len(X_train))
print("Number of test samples:", len(X_test))

print(
    "Number of ESM-2 features:",
    X_train.shape[1]
)


# ============================================================
# 8. DATA VALIDATION
# ============================================================

if np.isnan(X_train).any():

    raise ValueError(
        "NaN values detected in training features."
    )


if np.isnan(X_test).any():

    raise ValueError(
        "NaN values detected in test features."
    )


if np.isinf(X_train).any():

    raise ValueError(
        "Infinite values detected in training features."
    )


if np.isinf(X_test).any():

    raise ValueError(
        "Infinite values detected in test features."
    )


if X_train.shape[1] != X_test.shape[1]:

    raise ValueError(
        "Training and testing feature dimensions do not match."
    )


print("\nTraining class distribution:")

unique, counts = np.unique(
    y_train,
    return_counts=True
)

for u, c in zip(unique, counts):

    print(
        f"Class {u}: {c}"
    )


print("\nTesting class distribution:")

unique, counts = np.unique(
    y_test,
    return_counts=True
)

for u, c in zip(unique, counts):

    print(
        f"Class {u}: {c}"
    )


# ============================================================
# 9. RESIDUAL BLOCK
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(
        self,
        dimension,
        dropout
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Linear(
                dimension,
                dimension
            ),

            nn.BatchNorm1d(
                dimension
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                dimension,
                dimension
            ),

            nn.BatchNorm1d(
                dimension
            )
        )

        self.activation = nn.ReLU()


    def forward(self, x):

        residual = x

        output = self.block(x)

        output = output + residual

        output = self.activation(
            output
        )

        return output


# ============================================================
# 10. RESNET-STYLE MODEL
# ============================================================

class ResNetESM2(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        num_blocks=2,
        dropout=0.3
    ):

        super().__init__()


        self.input_layer = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.BatchNorm1d(
                hidden_dim
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            )
        )


        self.residual_blocks = nn.Sequential(

            *[
                ResidualBlock(
                    hidden_dim,
                    dropout
                )

                for _ in range(num_blocks)
            ]
        )


        self.classifier = nn.Sequential(

            nn.Linear(
                hidden_dim,
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                64,
                1
            )
        )


    def forward(self, x):

        x = self.input_layer(x)

        x = self.residual_blocks(x)

        x = self.classifier(x)

        return x.squeeze(1)


# ============================================================
# 11. ATTENTION MODEL
# ============================================================
#
# The fixed ESM-2 embedding is divided into feature tokens.
#
# Each token contains a group of ESM-2 dimensions.
#
# Multi-head self-attention is then applied across these
# feature tokens.
#
# ============================================================

class AttentionESM2(nn.Module):

    def __init__(
        self,
        input_dim,
        token_dim=64,
        num_heads=4,
        num_layers=1,
        dropout=0.3
    ):

        super().__init__()


        if input_dim % token_dim != 0:

            raise ValueError(
                "input_dim must be divisible by token_dim."
            )


        self.input_dim = input_dim

        self.token_dim = token_dim

        self.num_tokens = (
            input_dim // token_dim
        )


        encoder_layer = nn.TransformerEncoderLayer(

            d_model=token_dim,

            nhead=num_heads,

            dim_feedforward=token_dim * 4,

            dropout=dropout,

            activation="relu",

            batch_first=True
        )


        self.encoder = nn.TransformerEncoder(

            encoder_layer,

            num_layers=num_layers
        )


        self.classifier = nn.Sequential(

            nn.Linear(
                token_dim,
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                64,
                1
            )
        )


    def forward(self, x):

        batch_size = x.shape[0]


        # ----------------------------------------------------
        # Convert flat ESM-2 vector into feature tokens
        # ----------------------------------------------------

        x = x.reshape(
            batch_size,
            self.num_tokens,
            self.token_dim
        )


        # ----------------------------------------------------
        # Self-attention
        # ----------------------------------------------------

        x = self.encoder(x)


        # ----------------------------------------------------
        # Mean pooling over tokens
        # ----------------------------------------------------

        x = x.mean(
            dim=1
        )


        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        x = self.classifier(x)

        return x.squeeze(1)


# ============================================================
# 12. METRIC FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    y_prob
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    mcc = matthews_corrcoef(
        y_true,
        y_pred
    )

    roc_auc = roc_auc_score(
        y_true,
        y_prob
    )


    cm = confusion_matrix(
        y_true,
        y_pred
    )


    tn, fp, fn, tp = cm.ravel()


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


    return {

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1": f1,

        "MCC": mcc,

        "ROC_AUC": roc_auc,

        "Sensitivity": sensitivity,

        "Specificity": specificity
    }


# ============================================================
# 13. MODEL TRAINING FUNCTION
# ============================================================

def train_model(
    model,
    X_tr,
    y_tr,
    X_val,
    y_val,
    learning_rate,
    batch_size,
    epochs,
    patience=20
):

    model = model.to(DEVICE)


    # --------------------------------------------------------
    # Standardization
    # --------------------------------------------------------
    #
    # VERY IMPORTANT:
    #
    # The scaler is fitted ONLY on the current training fold.
    #
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_tr_scaled = scaler.fit_transform(
        X_tr
    )

    X_val_scaled = scaler.transform(
        X_val
    )


    # --------------------------------------------------------
    # Convert to tensors
    # --------------------------------------------------------

    X_tr_tensor = torch.tensor(
        X_tr_scaled,
        dtype=torch.float32
    )

    y_tr_tensor = torch.tensor(
        y_tr,
        dtype=torch.float32
    )


    X_val_tensor = torch.tensor(
        X_val_scaled,
        dtype=torch.float32
    )

    y_val_tensor = torch.tensor(
        y_val,
        dtype=torch.float32
    )


    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    train_dataset = TensorDataset(
        X_tr_tensor,
        y_tr_tensor
    )


    train_loader = DataLoader(

        train_dataset,

        batch_size=batch_size,

        shuffle=True
    )


    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.BCEWithLogitsLoss()


    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = optim.Adam(

        model.parameters(),

        lr=learning_rate,

        weight_decay=1e-4
    )


    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    best_val_loss = np.inf

    best_state = None

    patience_counter = 0


    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        epochs
    ):


        model.train()


        for batch_X, batch_y in train_loader:

            batch_X = batch_X.to(
                DEVICE
            )

            batch_y = batch_y.to(
                DEVICE
            )


            optimizer.zero_grad()


            logits = model(
                batch_X
            )


            loss = criterion(
                logits,
                batch_y
            )


            loss.backward()


            optimizer.step()


        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()


        with torch.no_grad():

            val_logits = model(
                X_val_tensor.to(
                    DEVICE
                )
            )


            val_loss = criterion(

                val_logits,

                y_val_tensor.to(
                    DEVICE
                )
            ).item()


        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_state = {
                k: v.cpu().clone()

                for k, v
                in model.state_dict().items()
            }

            patience_counter = 0

        else:

            patience_counter += 1


        if patience_counter >= patience:

            break


    # ========================================================
    # RESTORE BEST MODEL
    # ========================================================

    if best_state is not None:

        model.load_state_dict(
            best_state
        )


    model = model.to(
        DEVICE
    )


    return model, scaler


# ============================================================
# 14. PREDICTION FUNCTION
# ============================================================

def predict_model(
    model,
    scaler,
    X
):

    X_scaled = scaler.transform(
        X
    )


    X_tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32
    ).to(
        DEVICE
    )


    model.eval()


    with torch.no_grad():

        logits = model(
            X_tensor
        )

        probabilities = torch.sigmoid(
            logits
        ).cpu().numpy()


    predictions = (
        probabilities >= 0.5
    ).astype(int)


    return predictions, probabilities


# ============================================================
# 15. MODEL FACTORY
# ============================================================

INPUT_DIM = X_train.shape[1]


def create_resnet(params):

    return ResNetESM2(

        input_dim=INPUT_DIM,

        hidden_dim=params[
            "hidden_dim"
        ],

        num_blocks=params[
            "num_blocks"
        ],

        dropout=params[
            "dropout"
        ]
    )


def create_attention(params):

    return AttentionESM2(

        input_dim=INPUT_DIM,

        token_dim=params[
            "token_dim"
        ],

        num_heads=params[
            "num_heads"
        ],

        num_layers=params[
            "num_layers"
        ],

        dropout=params[
            "dropout"
        ]
    )


# ============================================================
# 16. HYPERPARAMETER GRIDS
# ============================================================

resnet_param_grid = {

    "hidden_dim": [
        128,
        256
    ],

    "num_blocks": [
        1,
        2
    ],

    "dropout": [
        0.2,
        0.4
    ],

    "learning_rate": [
        0.001,
        0.0001
    ],

    "batch_size": [
        16,
        32
    ]
}


attention_param_grid = {

    "token_dim": [
        32,
        64
    ],

    "num_heads": [
        2,
        4
    ],

    "num_layers": [
        1,
        2
    ],

    "dropout": [
        0.2,
        0.4
    ],

    "learning_rate": [
        0.001,
        0.0001
    ],

    "batch_size": [
        16,
        32
    ]
}


# ============================================================
# 17. 5-FOLD CV
# ============================================================
#
# Same fundamental role as your original GridSearchCV:
#
#   ONLY TRAINING DATA
#
# Test data remains untouched.
#
# ============================================================

cv5 = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=RANDOM_STATE
)


# ============================================================
# 18. RESNET HYPERPARAMETER SEARCH
# ============================================================

print("\n")
print("==========================================================")
print("RESNET-ESM2 HYPERPARAMETER SEARCH")
print("==========================================================")


best_resnet_score = -np.inf

best_resnet_params = None


resnet_results = []


for params in ParameterGrid(
    resnet_param_grid
):

    print(
        "\nTesting ResNet parameters:",
        params
    )


    fold_scores = []


    for fold, (
        train_idx,
        val_idx
    ) in enumerate(
        cv5.split(
            X_train,
            y_train
        ),
        start=1
    ):


        model = create_resnet(
            params
        )


        model, scaler = train_model(

            model,

            X_train[
                train_idx
            ],

            y_train[
                train_idx
            ],

            X_train[
                val_idx
            ],

            y_train[
                val_idx
            ],

            learning_rate=params[
                "learning_rate"
            ],

            batch_size=params[
                "batch_size"
            ],

            epochs=200,

            patience=20
        )


        predictions, probabilities = predict_model(

            model,

            scaler,

            X_train[
                val_idx
            ]
        )


        score = accuracy_score(

            y_train[
                val_idx
            ],

            predictions
        )


        fold_scores.append(
            score
        )


        print(
            f"Fold {fold}: {score:.4f}"
        )


    mean_score = np.mean(
        fold_scores
    )

    std_score = np.std(
        fold_scores,
        ddof=1
    )


    print(
        f"Mean Accuracy: {mean_score:.4f}"
    )

    print(
        f"SD: {std_score:.4f}"
    )


    resnet_results.append(

        {

            **params,

            "Mean_Accuracy":
                mean_score,

            "SD_Accuracy":
                std_score

        }
    )


    if mean_score > best_resnet_score:

        best_resnet_score = mean_score

        best_resnet_params = params.copy()


# ============================================================
# 19. SAVE RESNET GRID RESULTS
# ============================================================

resnet_grid_results = pd.DataFrame(
    resnet_results
)

resnet_grid_results.to_csv(
    "ResNet_ESM2_5fold_grid_results.csv",
    index=False
)


print("\n")
print("Best ResNet parameters:")

print(
    best_resnet_params
)

print(
    "Best ResNet 5-fold accuracy:",
    best_resnet_score
)


# ============================================================
# 20. ATTENTION HYPERPARAMETER SEARCH
# ============================================================

print("\n")
print("==========================================================")
print("ATTENTION-ESM2 HYPERPARAMETER SEARCH")
print("==========================================================")


best_attention_score = -np.inf

best_attention_params = None


attention_results = []


for params in ParameterGrid(
    attention_param_grid
):


    print(
        "\nTesting Attention parameters:",
        params
    )


    fold_scores = []


    for fold, (
        train_idx,
        val_idx
    ) in enumerate(
        cv5.split(
            X_train,
            y_train
        ),
        start=1
    ):


        model = create_attention(
            params
        )


        model, scaler = train_model(

            model,

            X_train[
                train_idx
            ],

            y_train[
                train_idx
            ],

            X_train[
                val_idx
            ],

            y_train[
                val_idx
            ],

            learning_rate=params[
                "learning_rate"
            ],

            batch_size=params[
                "batch_size"
            ],

            epochs=200,

            patience=20
        )


        predictions, probabilities = predict_model(

            model,

            scaler,

            X_train[
                val_idx
            ]
        )


        score = accuracy_score(

            y_train[
                val_idx
            ],

            predictions
        )


        fold_scores.append(
            score
        )


        print(
            f"Fold {fold}: {score:.4f}"
        )


    mean_score = np.mean(
        fold_scores
    )

    std_score = np.std(
        fold_scores,
        ddof=1
    )


    print(
        f"Mean Accuracy: {mean_score:.4f}"
    )

    print(
        f"SD: {std_score:.4f}"
    )


    attention_results.append(

        {

            **params,

            "Mean_Accuracy":
                mean_score,

            "SD_Accuracy":
                std_score

        }
    )


    if mean_score > best_attention_score:

        best_attention_score = mean_score

        best_attention_params = params.copy()


# ============================================================
# 21. SAVE ATTENTION GRID RESULTS
# ============================================================

attention_grid_results = pd.DataFrame(
    attention_results
)

attention_grid_results.to_csv(
    "Attention_ESM2_5fold_grid_results.csv",
    index=False
)


print("\n")
print("Best Attention parameters:")

print(
    best_attention_params
)

print(
    "Best Attention 5-fold accuracy:",
    best_attention_score
)


# ============================================================
# 22. FINAL 10-FOLD CV FUNCTION
# ============================================================

def perform_10fold_cv(
    model_type,
    best_params
):


    cv10 = StratifiedKFold(

        n_splits=10,

        shuffle=False
    )


    fold_results = []


    for fold, (
        train_idx,
        val_idx
    ) in enumerate(
        cv10.split(
            X_train,
            y_train
        ),
        start=1
    ):


        if model_type == "ResNet":

            model = create_resnet(
                best_params
            )

        elif model_type == "Attention":

            model = create_attention(
                best_params
            )

        else:

            raise ValueError(
                "Unknown model type."
            )


        model, scaler = train_model(

            model,

            X_train[
                train_idx
            ],

            y_train[
                train_idx
            ],

            X_train[
                val_idx
            ],

            y_train[
                val_idx
            ],

            learning_rate=best_params[
                "learning_rate"
            ],

            batch_size=best_params[
                "batch_size"
            ],

            epochs=200,

            patience=20
        )


        predictions, probabilities = predict_model(

            model,

            scaler,

            X_train[
                val_idx
            ]
        )


        metrics = calculate_metrics(

            y_train[
                val_idx
            ],

            predictions,

            probabilities
        )


        metrics[
            "Fold"
        ] = fold


        fold_results.append(
            metrics
        )


    return pd.DataFrame(
        fold_results
    )


# ============================================================
# 23. RESNET 10-FOLD CV
# ============================================================

print("\n")
print("==========================================================")
print("RESNET-ESM2 10-FOLD CV")
print("==========================================================")


resnet_10fold = perform_10fold_cv(

    "ResNet",

    best_resnet_params
)


print(
    resnet_10fold
)


resnet_10fold.to_csv(

    "ResNet_ESM2_10fold_results.csv",

    index=False
)


# ============================================================
# 24. ATTENTION 10-FOLD CV
# ============================================================

print("\n")
print("==========================================================")
print("ATTENTION-ESM2 10-FOLD CV")
print("==========================================================")


attention_10fold = perform_10fold_cv(

    "Attention",

    best_attention_params
)


print(
    attention_10fold
)


attention_10fold.to_csv(

    "Attention_ESM2_10fold_results.csv",

    index=False
)


# ============================================================
# 25. TRAIN FINAL RESNET ON COMPLETE TRAINING DATA
# ============================================================

print("\n")
print("==========================================================")
print("TRAINING FINAL RESNET")
print("==========================================================")


# ------------------------------------------------------------
# For final training, we create a small validation subset
# from the training data solely for early stopping.
#
# The independent test set remains untouched.
# ------------------------------------------------------------

from sklearn.model_selection import train_test_split


X_resnet_train, X_resnet_val, y_resnet_train, y_resnet_val = (

    train_test_split(

        X_train,

        y_train,

        test_size=0.1,

        stratify=y_train,

        random_state=RANDOM_STATE
    )
)


final_resnet = create_resnet(
    best_resnet_params
)


final_resnet, final_resnet_scaler = train_model(

    final_resnet,

    X_resnet_train,

    y_resnet_train,

    X_resnet_val,

    y_resnet_val,

    learning_rate=best_resnet_params[
        "learning_rate"
    ],

    batch_size=best_resnet_params[
        "batch_size"
    ],

    epochs=200,

    patience=20
)


# ============================================================
# 26. SAVE FINAL RESNET
# ============================================================

torch.save(

    {

        "model_state_dict":
            final_resnet.state_dict(),

        "input_dim":
            INPUT_DIM,

        "parameters":
            best_resnet_params,

        "scaler_mean":
            final_resnet_scaler.mean_,

        "scaler_scale":
            final_resnet_scaler.scale_

    },

    "best_resnet_esm2_model.pt"
)


# ============================================================
# 27. TRAIN FINAL ATTENTION MODEL
# ============================================================

print("\n")
print("==========================================================")
print("TRAINING FINAL ATTENTION MODEL")
print("==========================================================")


X_attention_train, X_attention_val, y_attention_train, y_attention_val = (

    train_test_split(

        X_train,

        y_train,

        test_size=0.1,

        stratify=y_train,

        random_state=RANDOM_STATE
    )
)


final_attention = create_attention(
    best_attention_params
)


final_attention, final_attention_scaler = train_model(

    final_attention,

    X_attention_train,

    y_attention_train,

    X_attention_val,

    y_attention_val,

    learning_rate=best_attention_params[
        "learning_rate"
    ],

    batch_size=best_attention_params[
        "batch_size"
    ],

    epochs=200,

    patience=20
)


# ============================================================
# 28. SAVE FINAL ATTENTION MODEL
# ============================================================

torch.save(

    {

        "model_state_dict":
            final_attention.state_dict(),

        "input_dim":
            INPUT_DIM,

        "parameters":
            best_attention_params,

        "scaler_mean":
            final_attention_scaler.mean_,

        "scaler_scale":
            final_attention_scaler.scale_

    },

    "best_attention_esm2_model.pt"
)


# ============================================================
# 29. TEST SET EVALUATION
# ============================================================

print("\n")
print("==========================================================")
print("INDEPENDENT TEST SET EVALUATION")
print("==========================================================")


# ------------------------------------------------------------
# ResNet
# ------------------------------------------------------------

resnet_pred, resnet_prob = predict_model(

    final_resnet,

    final_resnet_scaler,

    X_test
)


resnet_test_metrics = calculate_metrics(

    y_test,

    resnet_pred,

    resnet_prob
)


# ------------------------------------------------------------
# Attention
# ------------------------------------------------------------

attention_pred, attention_prob = predict_model(

    final_attention,

    final_attention_scaler,

    X_test
)


attention_test_metrics = calculate_metrics(

    y_test,

    attention_pred,

    attention_prob
)


# ============================================================
# 30. DISPLAY TEST RESULTS
# ============================================================

print("\n")
print("----------------------------------------------------------")
print("ResNet-ESM2")
print("----------------------------------------------------------")


for metric, value in resnet_test_metrics.items():

    print(
        f"{metric}: {value:.6f}"
    )


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        resnet_pred
    )
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        resnet_pred,
        digits=4,
        zero_division=0
    )
)


print("\n")
print("----------------------------------------------------------")
print("Attention-ESM2")
print("----------------------------------------------------------")


for metric, value in attention_test_metrics.items():

    print(
        f"{metric}: {value:.6f}"
    )


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        attention_pred
    )
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        attention_pred,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# 31. SAVE TEST RESULTS
# ============================================================

test_results = pd.DataFrame(

    [

        {
            "Model":
                "ResNet-ESM2",

            **resnet_test_metrics
        },

        {
            "Model":
                "Attention-ESM2",

            **attention_test_metrics
        }

    ]
)


test_results.to_csv(

    "ResNet_Attention_ESM2_test_results.csv",

    index=False
)


# ============================================================
# 32. SAVE TEST PREDICTIONS
# ============================================================

test_predictions = pd.DataFrame(

    {

        "True_Label":
            y_test,

        "ResNet_Prediction":
            resnet_pred,

        "ResNet_Probability":
            resnet_prob,

        "Attention_Prediction":
            attention_pred,

        "Attention_Probability":
            attention_prob

    }
)


test_predictions.to_csv(

    "ResNet_Attention_ESM2_test_predictions.csv",

    index=False
)


# ============================================================
# 33. ROC CURVES
# ============================================================

fpr_resnet, tpr_resnet, _ = roc_curve(

    y_test,

    resnet_prob
)


fpr_attention, tpr_attention, _ = roc_curve(

    y_test,

    attention_prob
)


auc_resnet = auc(

    fpr_resnet,

    tpr_resnet
)


auc_attention = auc(

    fpr_attention,

    tpr_attention
)


import matplotlib.pyplot as plt


plt.figure(
    figsize=(8, 6)
)


plt.plot(

    fpr_resnet,

    tpr_resnet,

    label=(
        f"ResNet-ESM2 "
        f"(AUC = {auc_resnet:.3f})"
    )
)


plt.plot(

    fpr_attention,

    tpr_attention,

    label=(
        f"Attention-ESM2 "
        f"(AUC = {auc_attention:.3f})"
    )
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
    "ROC Curves: Deep Learning ESM-2 Models"
)


plt.legend()


plt.tight_layout()


plt.savefig(

    "ResNet_Attention_ESM2_ROC.png",

    dpi=300
)


plt.close()


# ============================================================
# 34. COMBINE 10-FOLD RESULTS
# ============================================================

resnet_combined = resnet_10fold.copy()

resnet_combined.insert(
    0,
    "Model",
    "ResNet-ESM2"
)


attention_combined = attention_10fold.copy()

attention_combined.insert(
    0,
    "Model",
    "Attention-ESM2"
)


combined_10fold = pd.concat(

    [

        resnet_combined,

        attention_combined

    ],

    ignore_index=True
)


combined_10fold.to_csv(

    "ResNet_Attention_ESM2_10fold_combined.csv",

    index=False
)


# ============================================================
# 35. 10-FOLD MEAN ± SD
# ============================================================

summary_rows = []


for model_name, df in [

    (
        "ResNet-ESM2",
        resnet_10fold
    ),

    (
        "Attention-ESM2",
        attention_10fold
    )

]:


    for metric in [

        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "MCC",
        "ROC_AUC",
        "Sensitivity",
        "Specificity"

    ]:


        values = df[
            metric
        ].values


        summary_rows.append(

            {

                "Model":
                    model_name,

                "Metric":
                    metric,

                "Mean":
                    np.mean(values),

                "SD":
                    np.std(
                        values,
                        ddof=1
                    )

            }
        )


cv_summary = pd.DataFrame(
    summary_rows
)


print("\n")
print("==========================================================")
print("10-FOLD CV SUMMARY")
print("==========================================================")


print(
    cv_summary.to_string(
        index=False
    )
)


cv_summary.to_csv(

    "ResNet_Attention_ESM2_10fold_summary.csv",

    index=False
)


# ============================================================
# 36. FINAL SUMMARY
# ============================================================

print("\n")
print("==========================================================")
print("FINAL TEST SET COMPARISON")
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

    ].to_string(
        index=False
    )
)


# ============================================================
# 37. GENERATED FILES
# ============================================================

print("\n")
print("==========================================================")
print("FILES GENERATED")
print("==========================================================")


files_generated = [

    "ResNet_ESM2_5fold_grid_results.csv",

    "Attention_ESM2_5fold_grid_results.csv",

    "ResNet_ESM2_10fold_results.csv",

    "Attention_ESM2_10fold_results.csv",

    "ResNet_Attention_ESM2_10fold_combined.csv",

    "ResNet_Attention_ESM2_10fold_summary.csv",

    "best_resnet_esm2_model.pt",

    "best_attention_esm2_model.pt",

    "ResNet_Attention_ESM2_test_results.csv",

    "ResNet_Attention_ESM2_test_predictions.csv",

    "ResNet_Attention_ESM2_ROC.png"

]


for file in files_generated:

    print(
        file
    )


print("\n")
print("==========================================================")
print("ANALYSIS COMPLETED")
print("==========================================================")
