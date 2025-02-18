# **Bayesian Optimization for XGBoost Hyperparameter Tuning Using recall**
import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from bayes_opt import BayesianOptimization
from sklearn.metrics import make_scorer, f1_score, recall_score

# Load the feature-selected dataset
dataset_path = r"C:\Users\vagel\Desktop\C.R\Dataset\final_credit_risk_dataset.csv"
data = pd.read_csv(dataset_path)

#Separate Features & Target
X = data.drop(columns=["Credit_Status"])  # Features
y = data["Credit_Status"]  # Target

# Split the dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Define the function to optimize
def xgb_evaluate(learning_rate, n_estimators, max_depth, subsample, colsample_bytree, gamma):
    model = XGBClassifier(
        learning_rate=learning_rate,
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
    
    # Use Stratified K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
    recall_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring=make_scorer(recall_score))

    return recall_scores.mean()  # Maximize recall

# Define the parameter space
param_bounds = {
    "learning_rate": (0.01, 0.8),
    "n_estimators": (10, 500),
    "max_depth": (1, 20),
    "subsample": (0.1, 1.0),
    "colsample_bytree": (0.1, 1.0),
    "gamma": (0, 10)
}

# Run Bayesian Optimization
optimizer = BayesianOptimization(f=xgb_evaluate,
                                 pbounds=param_bounds,
                                 random_state=42)
optimizer.maximize(init_points=0, n_iter=40)

# Extract the best parameters
best_params = optimizer.max["params"]
best_params["n_estimators"] = int(best_params["n_estimators"])
best_params["max_depth"] = int(best_params["max_depth"])

# Save the best parameters to a JSON file
with open("xgb_best_params.json", "w") as f:
    json.dump(best_params, f, indent=4)

print("Best Parameters Saved in xgb_best_params.json")
