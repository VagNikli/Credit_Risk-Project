import json
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, recall_score
from bayes_opt import BayesianOptimization


class SVMBayesianOptimizer:
    """Performs Bayesian Optimization to find the best hyperparameters for SVM."""
    
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.best_params = None
        self._load_data()

    def _load_data(self):
        """Loads dataset and splits into training and testing sets."""
        data = pd.read_csv(self.dataset_path)
        self.X = data.drop(columns=["Credit_Status"])
        self.y = data["Credit_Status"]
        self.X_train, _, self.y_train, _ = train_test_split(
            self.X, self.y, test_size=0.2, stratify=self.y, random_state=42
        )

    def _svm_evaluate(self, C, gamma):
        """Objective function for Bayesian Optimization."""
        model = SVC(
            C=C,
            gamma=gamma,
            kernel="rbf",
            probability=True,
            random_state=42
        )

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        recall_scores = cross_val_score(model, self.X_train, self.y_train, cv=skf, scoring=make_scorer(recall_score))
        return recall_scores.mean()  # Maximize recall

    def optimize(self, n_iter=10):
        """Runs Bayesian Optimization to find the best hyperparameters."""
        param_bounds = {
            "C": (0.01, 10),  # Regularization parameter
            "gamma": (0.001, 1)  # Kernel coefficient
        }

        optimizer = BayesianOptimization(f=self._svm_evaluate, pbounds=param_bounds, random_state=42)
        optimizer.maximize(init_points=0, n_iter=n_iter)

        self.best_params = optimizer.max["params"]
        self.best_params["C"] = float(self.best_params["C"])
        self.best_params["gamma"] = float(self.best_params["gamma"])

    def save_best_params(self, json_path="svm_best_params.json"):
        """Saves the best hyperparameters to a JSON file."""
        if self.best_params:
            with open(json_path, "w") as f:
                json.dump(self.best_params, f, indent=4)
            print(f"Best Parameters Saved in {json_path}")
        else:
            print("No best parameters found. Run optimization first.")


if __name__ == "__main__":
    dataset_path = r"C:\Users\vagel\Desktop\CR_Risk Project\Dataset\final_credit_risk_dataset.csv"
    json_path = r"C:\Users\vagel\Desktop\CR_Risk Project\svm_best_params.json"

    optimizer = SVMBayesianOptimizer(dataset_path)
    optimizer.optimize(n_iter=10)
    optimizer.save_best_params(json_path)
