"""Unit tests for HuggingFace model wrappers."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.base import clone

from openadmet.models.architecture.huggingface import (
    HuggingFaceClassifier,
    HuggingFaceClassifierModel,
    HuggingFaceModel,
    HuggingFaceModelBase,
    HuggingFaceRegressor,
    HuggingFaceRegressorModel,
)
from openadmet.models.drivers import DriverType


# Test data
@pytest.fixture
def smiles_data():
    """Create simple SMILES data for testing."""
    X = np.array([
        "CCO",  # ethanol
        "CC(=O)O",  # acetic acid
        "c1ccccc1",  # benzene
        "CC(C)O",  # isopropanol
        "CCCC",  # butane
        "CC=O",  # acetaldehyde
    ])
    return X


@pytest.fixture
def regression_targets():
    """Create regression targets."""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


@pytest.fixture
def classification_targets():
    """Create classification targets."""
    return np.array([0, 1, 0, 1, 0, 1])


# Mock fixtures
@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": [[1, 2, 3]],
        "attention_mask": [[1, 1, 1]],
    }
    return tokenizer


@pytest.fixture
def mock_model():
    """Create a mock HuggingFace model."""
    model = MagicMock()
    model.return_value.logits = MagicMock()
    model.return_value.logits.squeeze.return_value.cpu.return_value.numpy.return_value = np.array([1.0])  # noqa: E501
    return model


class TestHuggingFaceModelBase:
    """Tests for HuggingFaceModelBase."""

    def test_driver_type(self):
        """Test that HuggingFaceModelBase uses sklearn driver."""

        class ConcreteModel(HuggingFaceModelBase):
            def build(self):
                pass

            def train(self, X, y):
                pass

            def predict(self, X):
                pass

        model = ConcreteModel()
        assert model._driver_type == DriverType.SKLEARN

    def test_make_new(self):
        """Test make_new creates a copy without estimator."""

        class ConcreteModel(HuggingFaceModelBase):
            test_param: str = "value"

            def build(self):
                self.estimator = "some_estimator"

            def train(self, X, y):
                pass

            def predict(self, X):
                pass

        model = ConcreteModel(test_param="custom")
        model.build()
        new_model = model.make_new()

        assert new_model.test_param == "custom"
        assert new_model.estimator is None


class TestHuggingFaceRegressor:
    """Tests for HuggingFaceRegressor sklearn wrapper."""

    def test_init_params(self):
        """Test that init params are stored correctly."""
        regressor = HuggingFaceRegressor(
            model_id="test-model",
            learning_rate=1e-4,
            num_train_epochs=5,
            per_device_train_batch_size=16,
            weight_decay=0.1,
            warmup_ratio=0.2,
            early_stopping_patience=5,
            accelerator="cpu",
        )
        assert regressor.model_id == "test-model"
        assert regressor.learning_rate == 1e-4
        assert regressor.num_train_epochs == 5
        assert regressor.per_device_train_batch_size == 16
        assert regressor.weight_decay == 0.1
        assert regressor.warmup_ratio == 0.2
        assert regressor.early_stopping_patience == 5
        assert regressor.accelerator == "cpu"

    def test_sklearn_clone(self):
        """Test that HuggingFaceRegressor can be cloned with sklearn."""
        regressor = HuggingFaceRegressor(
            model_id="test-model",
            learning_rate=1e-4,
        )
        cloned = clone(regressor)

        assert cloned.model_id == "test-model"
        assert cloned.learning_rate == 1e-4
        assert cloned is not regressor

    def test_get_params(self):
        """Test sklearn get_params compatibility."""
        regressor = HuggingFaceRegressor(learning_rate=1e-4)
        params = regressor.get_params()

        assert "learning_rate" in params
        assert params["learning_rate"] == 1e-4
        assert "model_id" in params
        assert "early_stopping_patience" in params

    def test_set_params(self):
        """Test sklearn set_params compatibility."""
        regressor = HuggingFaceRegressor()
        regressor.set_params(learning_rate=1e-3, num_train_epochs=20)

        assert regressor.learning_rate == 1e-3
        assert regressor.num_train_epochs == 20

    def test_predict_before_fit_raises(self):
        """Test that predict before fit raises an error."""
        pytest.importorskip("transformers")
        regressor = HuggingFaceRegressor()
        with pytest.raises(ValueError, match="Model not fitted"):
            regressor.predict(np.array(["CCO"]))

    def test_prepare_labels_normalization(self):
        """Test that _prepare_labels normalizes targets."""
        regressor = HuggingFaceRegressor()
        y = np.array([10.0, 20.0, 30.0])

        y_processed, num_labels = regressor._prepare_labels(y)

        assert num_labels == 1
        assert np.isclose(y_processed.mean(), 0.0, atol=1e-10)
        assert np.isclose(y_processed.std(), 1.0, atol=1e-10)
        assert regressor._y_mean == 20.0
        assert np.isclose(regressor._y_std, 8.1649658, atol=1e-5)

    def test_postprocess_predictions_denormalization(self):
        """Test that _postprocess_predictions denormalizes."""
        regressor = HuggingFaceRegressor()
        regressor._y_mean = 20.0
        regressor._y_std = 10.0

        logits = np.array([[0.0], [1.0], [-1.0]])
        preds = regressor._postprocess_predictions(logits)

        np.testing.assert_allclose(preds, [20.0, 30.0, 10.0])


class TestHuggingFaceClassifier:
    """Tests for HuggingFaceClassifier sklearn wrapper."""

    def test_init_params(self):
        """Test that init params are stored correctly."""
        classifier = HuggingFaceClassifier(
            model_id="test-model",
            learning_rate=1e-4,
        )
        assert classifier.model_id == "test-model"
        assert classifier.learning_rate == 1e-4

    def test_sklearn_clone(self):
        """Test that HuggingFaceClassifier can be cloned with sklearn."""
        classifier = HuggingFaceClassifier(
            model_id="test-model",
            learning_rate=1e-4,
        )
        cloned = clone(classifier)

        assert cloned.model_id == "test-model"
        assert cloned.learning_rate == 1e-4
        assert cloned is not classifier

    def test_prepare_labels_encoding(self):
        """Test that _prepare_labels encodes class labels."""
        classifier = HuggingFaceClassifier()
        y = np.array(["cat", "dog", "cat", "bird"])

        y_processed, num_labels = classifier._prepare_labels(y)

        assert num_labels == 3
        assert set(y_processed) == {0, 1, 2}
        assert len(classifier._classes) == 3

    def test_postprocess_predictions_decoding(self):
        """Test that _postprocess_predictions decodes to class labels."""
        classifier = HuggingFaceClassifier()
        classifier._prepare_labels(np.array(["cat", "dog", "bird"]))

        logits = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]])
        preds = classifier._postprocess_predictions(logits)

        assert len(preds) == 2
        assert preds[0] in classifier._classes
        assert preds[1] in classifier._classes


class TestHuggingFaceRegressorModel:
    """Tests for HuggingFaceRegressorModel OpenADMET wrapper."""

    def test_default_params(self):
        """Test default parameter values."""
        model = HuggingFaceRegressorModel()

        assert model.model_id == "ibm/MoLFormer-XL-both-10pct"
        assert model.learning_rate == 5e-5
        assert model.num_train_epochs == 200
        assert model.per_device_train_batch_size == 32
        assert model.weight_decay == 0.01
        assert model.warmup_ratio == 0.1
        assert model.dropout == 0.1
        assert model.lr_scheduler_type == "linear"
        assert model.gradient_accumulation_steps == 1
        assert model.max_grad_norm == 1.0
        assert model.adam_beta1 == 0.9
        assert model.adam_beta2 == 0.999
        assert model.adam_epsilon == 1e-8
        assert model.label_smoothing_factor == 0.0
        assert model.early_stopping_patience == 5
        assert model.accelerator == "auto"
        assert model.trust_remote_code is True

    def test_custom_params(self):
        """Test custom parameter values."""
        model = HuggingFaceRegressorModel(
            model_id="custom-model",
            learning_rate=1e-4,
            num_train_epochs=20,
            early_stopping_patience=5,
        )

        assert model.model_id == "custom-model"
        assert model.learning_rate == 1e-4
        assert model.num_train_epochs == 20
        assert model.early_stopping_patience == 5

    def test_accelerator_validation(self):
        """Test accelerator parameter validation."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            HuggingFaceRegressorModel(accelerator="invalid")

    def test_driver_type(self):
        """Test that model uses sklearn driver."""
        model = HuggingFaceRegressorModel()
        assert model._driver_type == DriverType.SKLEARN

    def test_type_class_var(self):
        """Test type class variable is set correctly."""
        assert HuggingFaceRegressorModel.type == "HuggingFaceRegressorModel"

    def test_build_creates_estimator(self):
        """Test that build creates the sklearn estimator."""
        model = HuggingFaceRegressorModel(
            model_id="test-model",
            learning_rate=1e-4,
        )
        model.build()

        assert model.estimator is not None
        assert isinstance(model.estimator, HuggingFaceRegressor)
        assert model.estimator.model_id == "test-model"
        assert model.estimator.learning_rate == 1e-4

    def test_build_idempotent(self):
        """Test that build doesn't recreate estimator if already exists."""
        model = HuggingFaceRegressorModel()
        model.build()
        first_estimator = model.estimator

        model.build()
        assert model.estimator is first_estimator

    def test_predict_without_train_raises(self):
        """Test that predict without training raises an error."""
        model = HuggingFaceRegressorModel()
        with pytest.raises(ValueError, match="Model not trained"):
            model.predict(np.array(["CCO"]))

    def test_make_new(self):
        """Test make_new creates copy without estimator."""
        model = HuggingFaceRegressorModel(learning_rate=1e-4)
        model.build()
        new_model = model.make_new()

        assert new_model.learning_rate == 1e-4
        assert new_model.estimator is None


class TestHuggingFaceClassifierModel:
    """Tests for HuggingFaceClassifierModel OpenADMET wrapper."""

    def test_default_params(self):
        """Test default parameter values."""
        model = HuggingFaceClassifierModel()
        assert model.model_id == "ibm/MoLFormer-XL-both-10pct"

    def test_type_class_var(self):
        """Test type class variable is set correctly."""
        assert HuggingFaceClassifierModel.type == "HuggingFaceClassifierModel"

    def test_build_creates_classifier_estimator(self):
        """Test that build creates the classifier estimator."""
        model = HuggingFaceClassifierModel()
        model.build()

        assert model.estimator is not None
        assert isinstance(model.estimator, HuggingFaceClassifier)


class TestModelRegistry:
    """Tests for model registry integration."""

    def test_regressor_registered(self):
        """Test that HuggingFaceRegressorModel is registered."""
        from openadmet.models.architecture.model_base import models

        assert "HuggingFaceRegressorModel" in models.keys()

    def test_classifier_registered(self):
        """Test that HuggingFaceClassifierModel is registered."""
        from openadmet.models.architecture.model_base import models

        assert "HuggingFaceClassifierModel" in models.keys()

    def test_get_regressor_class(self):
        """Test getting regressor class from registry."""
        from openadmet.models.architecture.model_base import get_mod_class

        cls = get_mod_class("HuggingFaceRegressorModel")
        assert cls == HuggingFaceRegressorModel

    def test_get_classifier_class(self):
        """Test getting classifier class from registry."""
        from openadmet.models.architecture.model_base import get_mod_class

        cls = get_mod_class("HuggingFaceClassifierModel")
        assert cls == HuggingFaceClassifierModel


class TestSerialization:
    """Tests for model serialization."""

    def test_regressor_model_dump(self):
        """Test that model can be dumped to dict."""
        model = HuggingFaceRegressorModel(
            model_id="test-model",
            learning_rate=1e-4,
        )
        dump = model.model_dump()

        assert dump["model_id"] == "test-model"
        assert dump["learning_rate"] == 1e-4
        assert "estimator" not in dump or dump["estimator"] is None

    def test_regressor_model_dump_json(self):
        """Test that model can be dumped to JSON."""
        model = HuggingFaceRegressorModel(
            model_id="test-model",
            learning_rate=1e-4,
        )
        json_str = model.model_dump_json()
        parsed = json.loads(json_str)

        assert parsed["model_id"] == "test-model"
        assert parsed["learning_rate"] == 1e-4


class TestOptunaCompatibility:
    """Tests for Optuna/sklearn compatibility."""

    def test_regressor_param_distributions_format(self):
        """Test that param distributions work with Optuna format."""
        # Verify default_param_distributions returns expected format
        default_params = HuggingFaceRegressorModel.default_param_distributions()

        # Verify all params exist in the model
        model = HuggingFaceRegressorModel()
        for param_name in default_params:
            assert hasattr(model, param_name), f"Missing param: {param_name}"

        # Verify expected params are included
        assert "learning_rate" in default_params
        assert "weight_decay" in default_params
        assert "warmup_ratio" in default_params
        assert "dropout" in default_params
        assert "lr_scheduler_type" in default_params
        assert "max_grad_norm" in default_params

        # Verify types
        assert default_params["learning_rate"]["type"] == "float"
        assert default_params["lr_scheduler_type"]["type"] == "categorical"

    def test_estimator_clonable_for_cv(self):
        """Test that estimator can be cloned for cross-validation."""
        model = HuggingFaceRegressorModel()
        model.build()

        # Should be clonable for use in OptunaSearchCV
        cloned = clone(model.estimator)
        assert cloned is not model.estimator
        assert cloned.learning_rate == model.estimator.learning_rate


class TestDeviceConfiguration:
    """Tests for device configuration."""

    def test_cpu_accelerator(self):
        """Test CPU accelerator configuration."""
        regressor = HuggingFaceRegressor(accelerator="cpu")
        use_cpu, use_bf16 = regressor._get_device_config()

        assert use_cpu is True
        assert use_bf16 is False

    def test_auto_accelerator_no_cuda(self):
        """Test auto accelerator when CUDA not available."""
        regressor = HuggingFaceRegressor(accelerator="auto")

        with patch("torch.cuda.is_available", return_value=False):
            use_cpu, use_bf16 = regressor._get_device_config()

        assert use_cpu is True
        assert use_bf16 is False

    def test_auto_accelerator_with_cuda(self):
        """Test auto accelerator when CUDA is available."""
        regressor = HuggingFaceRegressor(accelerator="auto")

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.is_bf16_supported", return_value=True):
                use_cpu, use_bf16 = regressor._get_device_config()

        assert use_cpu is False
        assert use_bf16 is True


class TestHuggingFaceE2EWithHPO:
    """End-to-end tests for HuggingFace models with HPO via Anvil workflow."""

    @pytest.mark.slow
    def test_molformer_nested_optuna_regression(self, tmp_path):
        """Test end-to-end nested CV with Optuna for HuggingFace regressor.

        This test verifies the complete integration with the Anvil API:
        - Creating an Anvil recipe YAML with SKLearnOptunaTrainer
        - Using HuggingFaceRegressorModel with MoLFormer
        - Running nested CV with Optuna through the workflow
        - Generating evaluation metrics

        Uses minimal settings for fast execution:
        - 2 epochs max with early stopping patience 1
        - 2 HPO trials
        - 2 outer CV splits, 2 inner CV splits
        """
        pytest.importorskip("transformers")

        import yaml

        from openadmet.models.anvil.specification import AnvilSpecification

        # Create test data with simple SMILES
        smiles_list = [
            "CCO",  # ethanol
            "CC(=O)O",  # acetic acid
            "c1ccccc1",  # benzene
            "CC(C)O",  # isopropanol
            "CCCC",  # butane
            "CC=O",  # acetaldehyde
            "CCN",  # ethylamine
            "CCC",  # propane
            "CCCO",  # propanol
            "CC(C)C",  # isobutane
            "CCOCC",  # diethyl ether
            "CC(=O)C",  # acetone
            "c1ccc(O)cc1",  # phenol
            "CCCBr",  # 1-bromopropane
            "CCCC(=O)O",  # butyric acid
            "c1ccc(C)cc1",  # toluene
        ]
        targets = [1.0, 2.0, 3.0, 1.5, 2.5, 3.5, 1.2, 2.2, 3.2, 1.8, 2.8, 3.8,
                   1.1, 2.1, 3.1, 1.6]

        import pandas as pd

        df = pd.DataFrame({
            "smiles": smiles_list,
            "target": targets,
        })
        csv_path = tmp_path / "test_data.csv"
        df.to_csv(csv_path, index=False)

        # Create Anvil recipe with nested Optuna trainer and HuggingFace model
        recipe = {
            "metadata": {
                "version": "v1",
                "name": "huggingface-molformer-nested-optuna-test",
                "build_number": 0,
                "description": "E2E test for HuggingFace with nested CV and Optuna",
                "tag": "test-huggingface-optuna",
                "authors": "Test",
                "email": "test@test.com",
                "date_created": "2024-01-01",
                "biotargets": ["TEST"],
                "tags": ["test", "huggingface", "molformer", "optuna"],
            },
            "data": {
                "type": "csv",
                "resource": str(csv_path),
                "input_col": "smiles",
                "target_cols": ["target"],
            },
            "procedure": {
                "split": {
                    "type": "ShuffleSplitter",
                    "params": {"train_size": 0.8, "random_state": 42},
                },
                "feat": {
                    "type": "IdentityFeaturizer",
                    "params": {},
                },
                "model": {
                    "type": "HuggingFaceRegressorModel",
                    "params": {
                        "model_id": "ibm/MoLFormer-XL-both-10pct",
                        "num_train_epochs": 2,
                        "early_stopping_patience": 1,
                        "per_device_train_batch_size": 4,
                        "accelerator": "cpu",
                    },
                },
                "train": {
                    "type": "SKLearnOptunaTrainer",
                    "params": {
                        "outer_n_splits": 2,
                        "outer_repeats": 1,
                        "inner_cv": 2,
                        "n_trials": 2,
                        "sampler_seed": 42,
                        "param_distributions": {
                            "learning_rate": {
                                "type": "float",
                                "low": 1e-5,
                                "high": 1e-4,
                                "log": True,
                            },
                            "weight_decay": {
                                "type": "float",
                                "low": 0.0,
                                "high": 0.1,
                            },
                        },
                    },
                },
            },
            "report": {
                "eval": [
                    {"type": "RegressionMetrics"},
                ]
            },
        }

        recipe_path = tmp_path / "huggingface_optuna_recipe.yaml"
        with open(recipe_path, "w") as f:
            yaml.dump(recipe, f)

        # Run workflow through Anvil API
        output_dir = tmp_path / "output"
        anvil_spec = AnvilSpecification.from_recipe(recipe_path)
        anvil_workflow = anvil_spec.to_workflow()
        anvil_workflow.run(output_dir=output_dir)

        # Verify expected outputs exist
        from pathlib import Path

        assert Path(output_dir / "model.json").exists()
        assert Path(output_dir / "regression_metrics.json").exists()
        assert Path(output_dir / "anvil_recipe.yaml").exists()

        # Verify regression metrics were computed
        with open(output_dir / "regression_metrics.json") as f:
            metrics = json.load(f)
            # Should have standard regression metrics - check nested under 'target'
            target_metrics = metrics.get("target", {})
            assert "mse" in target_metrics, f"Expected 'mse' in metrics: {metrics}"
            assert "mae" in target_metrics
            assert "r2" in target_metrics
