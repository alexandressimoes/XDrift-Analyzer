import numpy as np
from typing import Dict, List, Optional, Union, Any
import sklearn
from sklearn.inspection import permutation_importance
import shap

from xadapt_drift.adapters.base import BaseAdapter

class SklearnAdapter(BaseAdapter):
    """Adapter for scikit-learn compatible models."""
    
    def __init__(self, model: Any, feature_names: Optional[List[str]] = None, 
                 target_names: Optional[List[str]] = None):
        """
        Args:
            model: Scikit-learn compatible model
            feature_names: Names of input features
            target_names: Names of target classes
        """
        self.model = model
        self._feature_names = feature_names
        self.target_names = target_names
        
        # Determine if classifier or regressor
        self.is_classifier = hasattr(model, "classes_")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the model."""
        return self.model.predict(X)
    
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
        """Generate SHAP explanations."""
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
    
    def _explain_permutation(self, X: np.ndarray, y: np.ndarray, 
                           n_repeats: int = 10, random_state: int = 42, 
                           **kwargs) -> Dict[str, np.ndarray]:
        """Generate permutation importance explanations."""
        perm_importance = permutation_importance(
            self.model, X, y, n_repeats=n_repeats, 
            random_state=random_state, **kwargs
        )
        
        # Convert to dictionary format
        feature_importances = {}
        for i, feature in enumerate(self.feature_names):
            feature_importances[feature] = perm_importance.importances_mean[i]
            
        return feature_importances
    
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
