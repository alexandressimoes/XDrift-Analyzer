import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_classification, make_regression
from unittest.mock import patch, MagicMock

from xadapt_drift.adapters.sklearn_adapter import SklearnAdapter


class TestSklearnAdapter:
    """Test suite for SklearnAdapter following best practices."""
    
    @pytest.fixture
    def classification_data(self):
        """Generate sample classification data."""
        X, y = make_classification(
            n_samples=100, n_features=5, n_classes=2, random_state=42
        )
        return X, y
    
    @pytest.fixture  
    def regression_data(self):
        """Generate sample regression data."""
        X, y = make_regression(
            n_samples=100, n_features=5, random_state=42
        )
        return X, y
    
    @pytest.fixture
    def classifier_model(self, classification_data):
        """Trained classifier model."""
        X, y = classification_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model
    
    @pytest.fixture
    def regressor_model(self, regression_data):
        """Trained regressor model."""
        X, y = regression_data
        model = LinearRegression()
        model.fit(X, y)
        return model
    
    def test_adapter_initialization_with_classifier(self, classifier_model):
        """Test adapter initialization with a classifier."""
        feature_names = [f"feature_{i}" for i in range(5)]
        adapter = SklearnAdapter(
            model=classifier_model,
            feature_names=feature_names
        )
        
        assert adapter.model == classifier_model
        assert adapter.is_classifier == True
        assert adapter.feature_names == feature_names
    
    def test_adapter_initialization_with_regressor(self, regressor_model):
        """Test adapter initialization with a regressor."""
        feature_names = [f"feature_{i}" for i in range(5)]
        adapter = SklearnAdapter(
            model=regressor_model,
            feature_names=feature_names
        )
        
        assert adapter.model == regressor_model
        assert adapter.is_classifier == False
        assert adapter.feature_names == feature_names
    
    def test_model_validation_failure(self):
        """Test that invalid models are rejected."""
        invalid_model = MagicMock()
        del invalid_model.predict  # Remove predict method
        
        with pytest.raises(TypeError, match="Model must implement 'predict' method"):
            SklearnAdapter(invalid_model)
    
    def test_feature_names_mismatch(self, classifier_model):
        """Test validation of feature names count."""
        wrong_feature_names = ["feature_0", "feature_1"]  # Only 2 names for 5 features
        
        with pytest.raises(ValueError, match="Number of feature names"):
            SklearnAdapter(classifier_model, feature_names=wrong_feature_names)
    
    def test_predict_success(self, classifier_model, classification_data):
        """Test successful prediction."""
        X, _ = classification_data
        adapter = SklearnAdapter(classifier_model)
        
        predictions = adapter.predict(X)
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X)
    
    def test_predict_input_validation(self, classifier_model):
        """Test prediction input validation."""
        adapter = SklearnAdapter(classifier_model)
        
        # Test invalid dimensions
        with pytest.raises(ValueError, match="Input must be 2D array"):
            adapter.predict(np.array([1, 2, 3]))  # 1D array
    
    def test_explain_shap_method(self, classifier_model, classification_data):
        """Test SHAP explanation method."""
        X, _ = classification_data
        feature_names = [f"feature_{i}" for i in range(5)]
        adapter = SklearnAdapter(classifier_model, feature_names=feature_names)
        
        with patch('xadapt_drift.adapters.sklearn_adapter.shap') as mock_shap:
            # Mock SHAP explainer and values
            mock_explainer = MagicMock()
            mock_shap_values = MagicMock()
            mock_shap_values.values = np.random.rand(len(X), 5)
            
            mock_shap.Explainer.return_value = mock_explainer
            mock_explainer.return_value = mock_shap_values
            
            result = adapter.explain(X, method="shap")
            
            assert isinstance(result, dict)
            assert len(result) == 5  # Number of features
            assert all(feature in result for feature in feature_names)
    
    def test_explain_permutation_method(self, classifier_model, classification_data):
        """Test permutation importance explanation method."""
        X, y = classification_data
        feature_names = [f"feature_{i}" for i in range(5)]
        adapter = SklearnAdapter(classifier_model, feature_names=feature_names)
        
        with patch('xadapt_drift.adapters.sklearn_adapter.permutation_importance') as mock_perm:
            # Mock permutation importance result
            mock_result = MagicMock()
            mock_result.importances_mean = np.random.rand(5)
            mock_perm.return_value = mock_result
            
            result = adapter.explain(X, y, method="permutation")
            
            assert isinstance(result, dict)
            assert len(result) == 5
            assert all(feature in result for feature in feature_names)
    
    def test_explain_invalid_method(self, classifier_model, classification_data):
        """Test error handling for invalid explanation method."""
        X, _ = classification_data
        adapter = SklearnAdapter(classifier_model)
        
        with pytest.raises(ValueError, match="Unknown explanation method"):
            adapter.explain(X, method="invalid_method")
    
    def test_feature_names_inference(self, classification_data):
        """Test automatic feature name inference."""
        X, y = classification_data
        
        # Create model with feature_names_in_ attribute
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        adapter = SklearnAdapter(model)  # No explicit feature names
        
        # Should infer from model or create default names
        feature_names = adapter.feature_names
        assert len(feature_names) == X.shape[1]
        assert all(isinstance(name, str) for name in feature_names)
    
    def test_logging_integration(self, classifier_model, caplog):
        """Test that adapter logs initialization properly."""
        import logging
        
        with caplog.at_level(logging.INFO):
            adapter = SklearnAdapter(classifier_model)
            
        assert "Initialized SklearnAdapter" in caplog.text
        assert "classifier" in caplog.text
