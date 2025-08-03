from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Union, List, Optional, Protocol, runtime_checkable
import logging

logger = logging.getLogger(__name__)


@runtime_checkable
class MLModel(Protocol):
    """Protocol defining the minimum interface for ML models."""
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on input data."""
        ...


class BaseAdapter(ABC):
    """Base adapter interface for ML models.
    
    This adapter pattern allows XAdapt-Drift to work with different ML libraries
    (scikit-learn, TensorFlow, PyTorch, XGBoost, etc.) by providing a standardized
    interface for model interaction, drift detection, and explainability analysis.
    
    The adapter pattern provides several key benefits:
    1. **Unified Interface**: All ML models expose the same API regardless of framework
    2. **Extensibility**: Easy to add support for new ML frameworks
    3. **Type Safety**: Consistent type hints and validation across implementations
    4. **Maintainability**: Changes to the interface propagate to all implementations
    5. **Testing**: Standardized testing patterns for all adapters
    
    Examples:
        >>> # Different frameworks, same interface
        >>> sklearn_adapter = SklearnAdapter(sklearn_model)
        >>> tensorflow_adapter = TensorFlowAdapter(tf_model)  # Future implementation
        >>> 
        >>> # Polymorphic usage
        >>> adapters = [sklearn_adapter, tensorflow_adapter]
        >>> for adapter in adapters:
        ...     predictions = adapter.predict(X)
        ...     explanations = adapter.explain(X)
    """
    
    def __init__(self, model: MLModel, feature_names: Optional[List[str]] = None):
        """Initialize the base adapter.
        
        Args:
            model: The ML model to wrap
            feature_names: Optional list of feature names
        """
        self.model = model
        self._feature_names = feature_names
        logger.debug(f"Initialized {self.__class__.__name__}")
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the wrapped model.
        
        This method must be implemented by all adapter subclasses to provide
        a consistent prediction interface across different ML frameworks.
        
        Args:
            X: Input features for prediction, shape (n_samples, n_features)
            
        Returns:
            Model predictions, shape (n_samples,) for regression or 
            (n_samples,) for classification
            
        Raises:
            ValueError: If input data is invalid
            RuntimeError: If prediction fails
        """
        pass
    
    @abstractmethod
    def explain(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                method: str = "shap", **kwargs) -> Dict[str, np.ndarray]:
        """Generate feature importance explanations.
        
        This method provides explainability analysis for model predictions,
        supporting multiple explanation methods depending on the framework.
        
        Args:
            X: Features to explain, shape (n_samples, n_features)
            y: Optional ground truth labels for some explanation methods
            method: Explanation method ("shap", "permutation", "lime", etc.)
            **kwargs: Additional method-specific parameters
            
        Returns:
            Dictionary mapping feature names to importance values
            
        Raises:
            ValueError: If explanation method is not supported
            ImportError: If required explanation library is not available
        """
        pass
    
    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        """Get model feature names.
        
        Returns:
            List of feature names used by the model
        """
        pass
    
    # Concrete methods that provide common functionality
    
    def validate_input(self, X: np.ndarray, name: str = "X") -> np.ndarray:
        """Validate input data format.
        
        Args:
            X: Input data to validate
            name: Name of the input for error messages
            
        Returns:
            Validated numpy array
            
        Raises:
            ValueError: If input format is invalid
        """
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
            logger.debug(f"Converted {name} to numpy array")
            
        if X.ndim != 2:
            raise ValueError(f"{name} must be 2D array, got shape {X.shape}")
            
        if X.size == 0:
            raise ValueError(f"{name} cannot be empty")
            
        return X
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the wrapped model.
        
        Returns:
            Dictionary with model metadata
        """
        info = {
            "adapter_type": self.__class__.__name__,
            "model_type": type(self.model).__name__,
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names,
        }
        
        # Add model-specific info if available
        if hasattr(self.model, "get_params"):
            info["model_params"] = self.model.get_params()
            
        return info
    
    def __repr__(self) -> str:
        """String representation of the adapter."""
        return (f"{self.__class__.__name__}("
                f"model={type(self.model).__name__}, "
                f"features={len(self.feature_names)})")
    
    def __eq__(self, other: object) -> bool:
        """Check equality with another adapter."""
        if not isinstance(other, BaseAdapter):
            return NotImplemented
        return (type(self) == type(other) and 
                self.model == other.model and 
                self.feature_names == other.feature_names)
