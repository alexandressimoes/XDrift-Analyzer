import logging
import numpy as np
from typing import Dict, List, Optional, Union, Any, Protocol
import warnings
import sklearn
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance
import shap

from xadapt_drift.adapters.base import BaseAdapter

# Configure logging
logger = logging.getLogger(__name__)


class SklearnCompatible(Protocol):
    """Protocol for sklearn-compatible models."""
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        ...
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SklearnCompatible':
        """Fit the model."""
        ...

class SklearnAdapter(BaseAdapter):
    """Adapter for scikit-learn compatible models.
    
    This adapter provides a standardized interface for scikit-learn models,
    enabling drift detection and explainability analysis. It supports both
    classifiers and regressors with automatic type detection.
    
    Examples:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model = RandomForestClassifier()
        >>> adapter = SklearnAdapter(model, feature_names=['feature1', 'feature2'])
    """
    
    def __init__(self, model: SklearnCompatible, feature_names: Optional[List[str]] = None, 
                 target_names: Optional[List[str]] = None, validate_model: bool = True):
        """Initialize the sklearn adapter.
        
        Args:
            model: Scikit-learn compatible model that implements predict() and fit()
            feature_names: Names of input features. If None, will be inferred from model
            target_names: Names of target classes (for classification)
            validate_model: Whether to validate model compatibility
            
        Raises:
            TypeError: If model is not sklearn compatible
            ValueError: If feature names don't match model expectations
        """
        if validate_model:
            self._validate_model(model)
            
        # Call parent constructor to get common functionality
        super().__init__(model, feature_names)
        
        self.target_names = target_names
        
        # Determine if classifier or regressor
        self.is_classifier = hasattr(model, "classes_")
        
        # Validate feature names if model is fitted
        if hasattr(model, "n_features_in_") and feature_names is not None:
            if len(feature_names) != model.n_features_in_:
                raise ValueError(
                    f"Number of feature names ({len(feature_names)}) doesn't match "
                    f"model's expected features ({model.n_features_in_})"
                )
        
        logger.info(f"Initialized SklearnAdapter for {type(model).__name__} "
                   f"({'classifier' if self.is_classifier else 'regressor'})")
    
    def _validate_model(self, model: Any) -> None:
        """Validate that the model is sklearn compatible.
        
        Args:
            model: Model to validate
            
        Raises:
            TypeError: If model doesn't implement required methods
        """
        required_methods = ["predict"]
        for method in required_methods:
            if not hasattr(model, method):
                raise TypeError(f"Model must implement '{method}' method")
        
        if not callable(getattr(model, "predict")):
            raise TypeError("Model's 'predict' method must be callable")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the model.
        
        Args:
            X: Input features for prediction
            
        Returns:
            Model predictions
            
        Raises:
            ValueError: If input is invalid
            RuntimeError: If model prediction fails
        """
        # Use parent class validation
        X = self.validate_input(X, "X")
            
        try:
            return self.model.predict(X)
        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            raise RuntimeError(f"Prediction failed: {e}") from e
    
    def explain(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                method: str = "shap", **kwargs) -> Dict[str, np.ndarray]:
        """Generate feature importance explanations.
        
        Args:
            X: Features to explain
            y: Ground truth labels (for permutation importance)
            method: "shap" or "permutation"
            
        Returns:
            Dictionary with feature importances
        """
        if method.lower() == "shap":
            return self._explain_shap(X, **kwargs)
        elif method.lower() == "permutation":
            return self._explain_permutation(X, y, **kwargs)
        else:
            raise ValueError(f"Unknown explanation method: {method}")
    
    def _explain_shap(self, X: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """Generate SHAP explanations.
        
        Args:
            X: Features to explain
            **kwargs: Additional arguments for SHAP explainer
            
        Returns:
            Dictionary with feature importances
            
        Raises:
            ImportError: If SHAP is not available
            RuntimeError: If SHAP explanation fails
        """
        try:
            import shap
        except ImportError as e:
            raise ImportError("SHAP library is required for SHAP explanations. "
                            "Install with: pip install shap") from e
        
        # Use parent class validation
        X = self.validate_input(X, "X")
        
        try:
            # Create explainer based on model type
            if hasattr(self.model, "predict_proba"):
                explainer = shap.Explainer(self.model.predict_proba, X)
            else:
                explainer = shap.Explainer(self.model, X)
                
            # Calculate SHAP values
            shap_values = explainer(X)
            
            # Convert to dictionary format
            feature_importances = {}
            for i, feature in enumerate(self.feature_names):
                # Get absolute mean SHAP value for each feature
                feature_importances[feature] = np.abs(shap_values.values[:, i]).mean()
                
            return feature_importances
            
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            raise RuntimeError(f"SHAP explanation failed: {e}") from e
    
    def _explain_permutation(self, X: np.ndarray, y: np.ndarray, 
                           n_repeats: int = 10, random_state: int = 42, 
                           **kwargs) -> Dict[str, np.ndarray]:
        """Generate permutation importance explanations.
        
        Args:
            X: Features to explain
            y: Ground truth labels
            n_repeats: Number of times to permute each feature
            random_state: Random state for reproducibility
            **kwargs: Additional arguments for permutation_importance
            
        Returns:
            Dictionary with feature importances
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If permutation importance calculation fails
        """
        # Use parent class validation
        X = self.validate_input(X, "X")
        
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)
            
        if X.shape[0] != len(y):
            raise ValueError(f"X and y must have same number of samples: {X.shape[0]} vs {len(y)}")
            
        try:
            perm_importance = permutation_importance(
                self.model, X, y, n_repeats=n_repeats, 
                random_state=random_state, **kwargs
            )
            
            # Convert to dictionary format
            feature_importances = {}
            for i, feature in enumerate(self.feature_names):
                feature_importances[feature] = perm_importance.importances_mean[i]
                
            return feature_importances
            
        except Exception as e:
            logger.error(f"Permutation importance calculation failed: {e}")
            raise RuntimeError(f"Permutation importance calculation failed: {e}") from e
    @property
    def feature_names(self) -> List[str]:
        """Get feature names used by the model."""
        if self._feature_names is not None:
            return self._feature_names
        
        # Try to infer feature names from model
        if hasattr(self.model, "feature_names_in_"):
            return list(self.model.feature_names_in_)
        
        # Default to indices as strings
        n_features = self.model.n_features_in_ if hasattr(self.model, "n_features_in_") else 0
        return [f"feature_{i}" for i in range(n_features)]
