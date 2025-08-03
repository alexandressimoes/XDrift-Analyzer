"""
Example demonstrating the value of BaseAdapter pattern.

This module shows how the BaseAdapter enables polymorphism and extensibility
by implementing multiple adapters that work seamlessly together.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from abc import ABC

from xadapt_drift.adapters.base import BaseAdapter


class MockTensorFlowAdapter(BaseAdapter):
    """Mock TensorFlow adapter to demonstrate polymorphism."""
    
    def __init__(self, model: Any, feature_names: Optional[List[str]] = None):
        """Initialize the TensorFlow adapter."""
        super().__init__(model, feature_names)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Mock TensorFlow prediction."""
        X = self.validate_input(X)
        # Simulate TensorFlow model prediction
        return np.random.rand(X.shape[0])
    
    def explain(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                method: str = "integrated_gradients", **kwargs) -> Dict[str, np.ndarray]:
        """Mock TensorFlow explanation using Integrated Gradients."""
        X = self.validate_input(X)
        
        if method == "integrated_gradients":
            # Simulate integrated gradients
            importances = {}
            for i, feature in enumerate(self.feature_names):
                importances[feature] = np.random.rand()
            return importances
        else:
            raise ValueError(f"TensorFlow adapter doesn't support method: {method}")
    
    @property
    def feature_names(self) -> List[str]:
        """Get feature names."""
        if self._feature_names is not None:
            return self._feature_names
        # Default feature names for demo
        return [f"tf_feature_{i}" for i in range(10)]


class MockXGBoostAdapter(BaseAdapter):
    """Mock XGBoost adapter to demonstrate polymorphism."""
    
    def __init__(self, model: Any, feature_names: Optional[List[str]] = None):
        """Initialize the XGBoost adapter."""
        super().__init__(model, feature_names)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Mock XGBoost prediction."""
        X = self.validate_input(X)
        # Simulate XGBoost model prediction
        return np.random.rand(X.shape[0])
    
    def explain(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                method: str = "tree_shap", **kwargs) -> Dict[str, np.ndarray]:
        """Mock XGBoost explanation using TreeSHAP."""
        X = self.validate_input(X)
        
        if method == "tree_shap":
            # Simulate TreeSHAP values
            importances = {}
            for i, feature in enumerate(self.feature_names):
                importances[feature] = np.random.rand()
            return importances
        elif method == "feature_importance":
            # Simulate built-in feature importance
            importances = {}
            for i, feature in enumerate(self.feature_names):
                importances[feature] = np.random.rand()
            return importances
        else:
            raise ValueError(f"XGBoost adapter doesn't support method: {method}")
    
    @property
    def feature_names(self) -> List[str]:
        """Get feature names."""
        if self._feature_names is not None:
            return self._feature_names
        # Default feature names for demo
        return [f"xgb_feature_{i}" for i in range(8)]


def demonstrate_polymorphism():
    """Demonstrate how BaseAdapter enables polymorphism."""
    
    # Create mock models
    sklearn_model = object()  # Mock sklearn model
    tf_model = object()       # Mock TensorFlow model
    xgb_model = object()      # Mock XGBoost model
    
    # Create different adapters
    from xadapt_drift.adapters.sklearn_adapter import SklearnAdapter
    
    # Note: Using mock for sklearn since we need a real model for full demo
    adapters = [
        MockTensorFlowAdapter(tf_model, feature_names=[f"feature_{i}" for i in range(5)]),
        MockXGBoostAdapter(xgb_model, feature_names=[f"feature_{i}" for i in range(5)])
    ]
    
    # Generate sample data
    X = np.random.rand(100, 5)
    
    print("🔄 Demonstrating Polymorphism with BaseAdapter")
    print("=" * 50)
    
    # Use adapters polymorphically
    for i, adapter in enumerate(adapters, 1):
        print(f"\n{i}. Using {adapter.__class__.__name__}")
        print(f"   Model Info: {adapter.get_model_info()['model_type']}")
        
        # Same interface, different implementations
        predictions = adapter.predict(X)
        print(f"   Predictions shape: {predictions.shape}")
        
        # Different explanation methods for different frameworks
        if isinstance(adapter, MockTensorFlowAdapter):
            explanations = adapter.explain(X, method="integrated_gradients")
        elif isinstance(adapter, MockXGBoostAdapter):
            explanations = adapter.explain(X, method="tree_shap")
        
        print(f"   Explanation features: {list(explanations.keys())}")
        print(f"   Sample importance: {list(explanations.values())[0]:.4f}")


def demonstrate_extensibility():
    """Demonstrate how easy it is to extend with new adapters."""
    
    print("\n🚀 Demonstrating Extensibility")
    print("=" * 50)
    
    # Easy to add new ML framework support
    class MockPyTorchAdapter(BaseAdapter):
        """New PyTorch adapter - easy to add!"""
        
        def predict(self, X: np.ndarray) -> np.ndarray:
            X = self.validate_input(X)
            return np.random.rand(X.shape[0])
        
        def explain(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
                    method: str = "captum", **kwargs) -> Dict[str, np.ndarray]:
            X = self.validate_input(X)
            importances = {}
            for i, feature in enumerate(self.feature_names):
                importances[feature] = np.random.rand()
            return importances
        
        @property
        def feature_names(self) -> List[str]:
            return self._feature_names or [f"pytorch_feature_{i}" for i in range(6)]
    
    # Instantly works with existing infrastructure
    pytorch_adapter = MockPyTorchAdapter(
        model=object(), 
        feature_names=[f"feature_{i}" for i in range(5)]
    )
    
    X = np.random.rand(50, 5)
    predictions = pytorch_adapter.predict(X)
    explanations = pytorch_adapter.explain(X, method="captum")
    
    print(f"✅ New PyTorch adapter created and working!")
    print(f"   Predictions: {predictions.shape}")
    print(f"   Explanations: {len(explanations)} features")


def demonstrate_validation():
    """Demonstrate shared validation logic."""
    
    print("\n🛡️ Demonstrating Shared Validation")
    print("=" * 50)
    
    adapter = MockTensorFlowAdapter(object())
    
    try:
        # This will fail validation in the base class
        invalid_data = np.array([1, 2, 3])  # 1D instead of 2D
        adapter.predict(invalid_data)
    except ValueError as e:
        print(f"✅ Validation caught error: {e}")
    
    try:
        # This will pass validation
        valid_data = np.random.rand(10, 5)
        predictions = adapter.predict(valid_data)
        print(f"✅ Valid data processed successfully: {predictions.shape}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    demonstrate_polymorphism()
    demonstrate_extensibility() 
    demonstrate_validation()
    
    print("\n" + "=" * 50)
    print("🎯 Benefits of BaseAdapter Pattern:")
    print("   1. Polymorphism - Same interface for all ML frameworks")
    print("   2. Extensibility - Easy to add new framework support")
    print("   3. Validation - Shared input validation logic")
    print("   4. Maintainability - Centralized interface changes")
    print("   5. Type Safety - Consistent type hints")
    print("   6. Testing - Standardized test patterns")
