"""
Baseline Machine Learning Models for Elliptic Bitcoin Scam Detection.
Implements Logistic Regression and Random Forest models with class imbalance handling.
"""

from pathlib import Path
from typing import Optional, Union, Dict, Any
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.utils.logger import get_logger

logger = get_logger("baselines")


class LogisticRegressionBaseline:
    """
    Logistic Regression baseline classifier.
    Handles class imbalance using balanced class weights.
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: Optional[Union[str, Dict[int, float]]] = "balanced",
        solver: str = "lbfgs",
        random_state: int = 42,
    ):
        """
        Initialize Logistic Regression model.

        Args:
            C: Inverse regularization strength.
            max_iter: Maximum number of iterations for solver.
            class_weight: Weights associated with classes ('balanced' or dict).
            solver: Optimization algorithm solver.
            random_state: Seed used by random number generator.
        """
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.solver = solver
        self.random_state = random_state

        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            solver=self.solver,
            random_state=self.random_state,
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionBaseline":
        """
        Fit Logistic Regression model on training features and labels.

        Args:
            X: Feature matrix of shape (N, D).
            y: Target binary labels of shape (N,).

        Returns:
            Self instance.
        """
        logger.info(f"Fitting Logistic Regression baseline on {X.shape[0]} samples with {X.shape[1]} features...")
        self.model.fit(X, y)
        self.is_fitted = True
        logger.info("Logistic Regression training complete.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary class labels for input features.

        Args:
            X: Feature matrix of shape (N, D).

        Returns:
            Predicted binary labels array of shape (N,).
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() before predict().")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for positive class (class 1: Illicit).

        Args:
            X: Feature matrix of shape (N, D).

        Returns:
            Probability of class 1 array of shape (N,).
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() before predict_proba().")
        probas = self.model.predict_proba(X)
        if probas.shape[1] == 2:
            return probas[:, 1]
        return probas[:, 0]

    def save(self, filepath: Union[str, Path]) -> Path:
        """
        Save trained model to disk.

        Args:
            filepath: Destination file path.

        Returns:
            Path of saved model.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        logger.info(f"Saved Logistic Regression model checkpoint to {filepath}")
        return filepath

    def load(self, filepath: Union[str, Path]) -> "LogisticRegressionBaseline":
        """
        Load model checkpoint from disk.

        Args:
            filepath: Source file path.

        Returns:
            Loaded self instance.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {filepath}")
        self.model = joblib.load(filepath)
        self.is_fitted = True
        logger.info(f"Loaded Logistic Regression model checkpoint from {filepath}")
        return self


class RandomForestBaseline:
    """
    Random Forest baseline classifier.
    Handles class imbalance using balanced class weights or balanced subsampling.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 15,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        class_weight: Optional[Union[str, Dict[int, float]]] = "balanced",
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        """
        Initialize Random Forest model.

        Args:
            n_estimators: Number of trees in the forest.
            max_depth: Maximum depth of the trees.
            min_samples_split: Minimum samples required to split an internal node.
            min_samples_leaf: Minimum samples required at a leaf node.
            class_weight: Weights associated with classes ('balanced', 'balanced_subsample', or dict).
            random_state: Seed used by random number generator.
            n_jobs: Number of parallel jobs for fitting and predicting (-1 uses all cores).
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestBaseline":
        """
        Fit Random Forest model on training features and labels.

        Args:
            X: Feature matrix of shape (N, D).
            y: Target binary labels of shape (N,).

        Returns:
            Self instance.
        """
        logger.info(f"Fitting Random Forest baseline ({self.n_estimators} trees, max_depth={self.max_depth}) on {X.shape[0]} samples...")
        self.model.fit(X, y)
        self.is_fitted = True
        logger.info("Random Forest training complete.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary class labels for input features.

        Args:
            X: Feature matrix of shape (N, D).

        Returns:
            Predicted binary labels array of shape (N,).
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() before predict().")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for positive class (class 1: Illicit).

        Args:
            X: Feature matrix of shape (N, D).

        Returns:
            Probability of class 1 array of shape (N,).
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() before predict_proba().")
        probas = self.model.predict_proba(X)
        if probas.shape[1] == 2:
            return probas[:, 1]
        return probas[:, 0]

    def save(self, filepath: Union[str, Path]) -> Path:
        """
        Save trained model to disk.

        Args:
            filepath: Destination file path.

        Returns:
            Path of saved model.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        logger.info(f"Saved Random Forest model checkpoint to {filepath}")
        return filepath

    def load(self, filepath: Union[str, Path]) -> "RandomForestBaseline":
        """
        Load model checkpoint from disk.

        Args:
            filepath: Source file path.

        Returns:
            Loaded self instance.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {filepath}")
        self.model = joblib.load(filepath)
        self.is_fitted = True
        logger.info(f"Loaded Random Forest model checkpoint from {filepath}")
        return self
