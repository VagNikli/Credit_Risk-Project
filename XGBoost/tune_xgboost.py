# **Bayesian Optimization for XGBoost Hyperparameter Tuning Using recall**
import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from bayes_opt import BayesianOptimization
from sklearn.metrics import make_scorer, f1_score, recall_score

class XGBBayesianOptimizer:
    """Performs Bayesian Optimization to find the best hyperparameters for XGBoost."""
    
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

    def _xgb_evaluate(self, learning_rate, n_estimators, max_depth, subsample, colsample_bytree, gamma):
        """Objective function for Bayesian Optimization."""
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

        skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
        recall_scores = cross_val_score(model, self.X_train, self.y_train, cv=skf, scoring=make_scorer(recall_score))
        return recall_scores.mean()

    def optimize(self, n_iter=40):
        """Runs Bayesian Optimization to find the best hyperparameters."""
        param_bounds = {
            "learning_rate": (0.01, 0.8),
            "n_estimators": (10, 500),
            "max_depth": (1, 20),
            "subsample": (0.1, 1.0),
            "colsample_bytree": (0.1, 1.0),
            "gamma": (0, 10)
        }

        optimizer = BayesianOptimization(f=self._xgb_evaluate, pbounds=param_bounds, random_state=42)
        optimizer.maximize(init_points=0, n_iter=n_iter)

        self.best_params = optimizer.max["params"]
        self.best_params["n_estimators"] = int(self.best_params["n_estimators"])
        self.best_params["max_depth"] = int(self.best_params["max_depth"])

    def save_best_params(self, json_path="xgb_best_params.json"):
        """Saves the best hyperparameters to a JSON file."""
        if self.best_params:
            with open(json_path, "w") as f:
                json.dump(self.best_params, f, indent=4)
            print(f"Best Parameters Saved in {json_path}")
        else:
            print("No best parameters found. Run optimization first.")

# ========== RUNNING THE PIPELINE ==========
if __name__ == "__main__":
    dataset_path = r"C:\Users\vagel\Desktop\CR_Risk Project\Dataset\final_credit_risk_dataset.csv"
    json_path = r"C:\Users\vagel\Desktop\CR_Risk Project\xgboost_best_params.json"

    optimizer = XGBBayesianOptimizer(dataset_path)
    optimizer.optimize(n_iter=40)
    optimizer.save_best_params(json_path)
