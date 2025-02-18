import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from bayes_opt import BayesianOptimization
from sklearn.metrics import make_scorer, f1_score, recall_score

# Load the dataset
dataset_path = r"C:\Users\vagel\Desktop\C.R\Dataset\final_credit_risk_dataset.csv"
data = pd.read_csv(dataset_path)

# Define features (X) and target (y)
X = data.drop(columns=["Credit_Status"])
y = data["Credit_Status"]

# Split the dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y)

# Define the function to optimize
def logreg_evaluate(C):
    model = LogisticRegression(
        C=C,
        solver="liblinear",
        random_state=42
    )
    
    # Use Stratified K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    recall_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring=make_scorer(recall_score))

    return recall_scores.mean()  # Maximize recall

# Define the parameter space
param_bounds = {
    "C": (0.01, 10)
}

# Run Bayesian Optimization
optimizer = BayesianOptimization(f=logreg_evaluate,
                                 pbounds=param_bounds,
                                 random_state=42)
optimizer.maximize(init_points=0, n_iter=20)

# Extract the best parameters
best_params = optimizer.max["params"]
best_params["C"] = float(best_params["C"])

# Save the best parameters to a JSON file
with open("logreg_best_params.json", "w") as f:
    json.dump(best_params, f, indent=4)

print("Best Parameters Saved in logreg_best_params.json")
