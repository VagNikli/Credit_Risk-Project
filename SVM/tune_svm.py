import json
import numpy as np
import pandas as pd
from sklearn.svm import SVC
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define the function to optimize
def svm_evaluate(C, gamma):
    model = SVC(
        C=C,
        gamma=gamma,
        kernel="rbf",
        probability=True,
        random_state=42
    )
    
    # Use Stratified K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    recall_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring=make_scorer(recall_score))

    return recall_scores.mean()  # Maximize recall

# Define the parameter space
param_bounds = {
    "C": (0.01, 10),  # Regularization parameter
    "gamma": (0.001, 1)  # Kernel coefficient
}

# Run Bayesian Optimization
optimizer = BayesianOptimization(f=svm_evaluate, pbounds=param_bounds, random_state=42)
optimizer.maximize(init_points=0, n_iter=2)

# Extract the best parameters
best_params = optimizer.max["params"]
best_params["C"] = float(best_params["C"])
best_params["gamma"] = float(best_params["gamma"])

# Save the best parameters to a JSON file
with open("svm_best_params.json", "w") as f:
    json.dump(best_params, f, indent=4)

print("Best Parameters Saved in svm_best_params.json")
