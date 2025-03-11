import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, recall_score
from bayes_opt import BayesianOptimization



class RFBayesianOptimizer:
    """Performs Bayesian Optimization to find the best hyperparameters for Random Forest."""
    
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

    def _rf_evaluate(self, n_estimators, max_depth, min_samples_split, min_samples_leaf):
        """Objective function for Bayesian Optimization."""
        model = RandomForestClassifier(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            min_samples_split=int(min_samples_split),
            min_samples_leaf=int(min_samples_leaf),
            random_state=42,
            n_jobs=-1
        )

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        recall_scores = cross_val_score(model, self.X_train, self.y_train, cv=skf, scoring=make_scorer(recall_score))
        return recall_scores.mean()  # Maximize recall

    def optimize(self, n_iter=40):
        """Runs Bayesian Optimization to find the best hyperparameters."""
        param_bounds = {
            "n_estimators": (50, 500),
            "max_depth": (3, 20),
            "min_samples_split": (2, 10),
            "min_samples_leaf": (1, 10)
        }

        optimizer = BayesianOptimization(f=self._rf_evaluate, pbounds=param_bounds, random_state=42)
        optimizer.maximize(init_points=2, n_iter=n_iter)

        self.best_params = optimizer.max["params"]
        self.best_params["n_estimators"] = int(self.best_params["n_estimators"])
        self.best_params["max_depth"] = int(self.best_params["max_depth"])
        self.best_params["min_samples_split"] = int(self.best_params["min_samples_split"])
        self.best_params["min_samples_leaf"] = int(self.best_params["min_samples_leaf"])

    def save_best_params(self, json_path="rf_best_params.json"):
        """Saves the best hyperparameters to a JSON file."""
        if self.best_params:
            with open(json_path, "w") as f:
                json.dump(self.best_params, f, indent=4)
            print(f"Best Parameters Saved in {json_path}")
        else:
            print("No best parameters found. Run optimization first.")


if __name__ == "__main__":
    dataset_path = r"C:\Users\vagel\Desktop\CR_Risk Project\Dataset\final_credit_risk_dataset.csv"
    json_path = r"C:\Users\vagel\Desktop\CR_Risk Project\rf_best_params.json"

    optimizer = RFBayesianOptimizer(dataset_path)
    optimizer.optimize(n_iter=40)
    optimizer.save_best_params(json_path)