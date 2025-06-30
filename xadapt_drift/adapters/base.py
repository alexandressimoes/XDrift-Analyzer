from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Union, List, Optional

class BaseAdapter(ABC):
    """Base adapter interface for ML models.
    
    This adapter pattern allows XAdapt-Drift to work with different ML libraries
    by providing a standardized interface for model interaction.
    """
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the wrapped model.
        
        Args:
            X: Features to predict on
            
        Returns:
            Model predictions
        """
        pass
    
    @abstractmethod
    def explain(self, X: np.ndarray, method: str = "shap") -> Dict[str, np.ndarray]:
        """Generate feature importance explanations.
        
        Args:
            X: Features to explain
            method: Explanation method ("shap" or "permutation")
            
        Returns:
            Dictionary mapping feature names to importance values
        """
        pass
    
    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        """Get model feature names."""
        pass
