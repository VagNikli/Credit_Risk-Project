import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, recall_score
from bayes_opt import BayesianOptimization


# Load the dataset
dataset_path = r"C:\Users\vagel\Desktop\C.R\Dataset\final_credit_risk_dataset.csv"
data = pd.read_csv(dataset_path)

# Define features (X) and target (y)
X = data.drop(columns=["Credit_Status"])
y = data["Credit_Status"]

# Split the dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Define the function to optimize
def rf_evaluate(n_estimators, max_depth, min_samples_split, min_samples_leaf):
    model = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        random_state=42,
        n_jobs=-1
    )
    
    # Use Stratified K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    recall_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring=make_scorer(recall_score))

    return recall_scores.mean()  # Maximize recall

# Define the parameter space
param_bounds = {
    "n_estimators": (50, 500),
    "max_depth": (3, 20),
    "min_samples_split": (2, 10),
    "min_samples_leaf": (1, 10)
}

# Run Bayesian Optimization
optimizer = BayesianOptimization(f=rf_evaluate, pbounds=param_bounds, random_state=42)
optimizer.maximize(init_points=0, n_iter=40)

# Extract the best parameters
best_params = optimizer.max["params"]
best_params["n_estimators"] = int(best_params["n_estimators"])
best_params["max_depth"] = int(best_params["max_depth"])
best_params["min_samples_split"] = int(best_params["min_samples_split"])
best_params["min_samples_leaf"] = int(best_params["min_samples_leaf"])

# Save the best parameters to a JSON file
with open("rf_best_params.json", "w") as f:
    json.dump(best_params, f, indent=4)

print("Best Parameters Saved in rf_best_params.json")
