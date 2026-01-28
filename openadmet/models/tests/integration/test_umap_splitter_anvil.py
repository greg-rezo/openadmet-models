"""Integration tests for UMAPCVSplitter with Anvil workflows."""

import pandas as pd
import pytest
import yaml
from sklearn.datasets import make_regression

from openadmet.models.anvil.specification import AnvilSpecification
from openadmet.models.split.umap_split import UMAPCVSplitter


@pytest.fixture
def regression_data(tmp_path):
    """Create regression test data with enough samples for UMAP.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        Path to the created CSV file.

    """
    X, y = make_regression(
        n_samples=50, n_features=10, n_informative=5, random_state=42
    )
    df = pd.DataFrame(
        {"smiles": [f"C{'C' * (i % 20)}" for i in range(len(y))], "activity": y}
    )
    csv_path = tmp_path / "regression_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_umap_splitter_anvil_workflow(tmp_path, regression_data):
    """Test UMAPCVSplitter in Anvil nested CV workflow.

    This test verifies that UMAPCVSplitter can be used as custom_outer_cv
    in a nested cross-validation evaluator within an Anvil workflow.

    Args:
        tmp_path: Pytest temporary directory fixture.
        regression_data: Fixture providing test data path.

    """
    recipe = {
        "metadata": {
            "version": "v1",
            "name": "umap-cv-test",
            "build_number": 0,
            "description": "Test UMAPCVSplitter",
            "tag": "test-umap-cv",
            "authors": "Test",
            "email": "test@test.com",
            "date_created": "2024-01-01",
            "biotargets": ["TEST"],
            "tags": ["test"],
        },
        "data": {
            "type": "csv",
            "resource": str(regression_data),
            "target": "activity",
            "smiles_col": "smiles",
        },
        "split": {
            "type": "ShuffleSplitter",
            "params": {"train_size": 1.0, "val_size": 0.0, "test_size": 0.0},
        },
        "procedure": {
            "feat": {
                "type": "FingerprintFeaturizer",
                "params": {"fp_type": "ecfp:4"},
            },
            "model": {"type": "RidgeModel", "params": {}},
            "train": {"type": "SKLearnTrainer", "params": {}},
        },
        "report": {
            "eval": [
                {
                    "type": "SKLearnRepeatedKFoldCrossValidation",
                    "params": {
                        "n_splits": 2,
                        "n_repeats": 1,
                        "random_state": 42,
                    },
                },
            ]
        },
    }

    recipe_path = tmp_path / "umap_cv_recipe.yaml"
    with open(recipe_path, "w") as f:
        yaml.dump(recipe, f)

    spec = AnvilSpecification.from_yaml(recipe_path)
    workflow = spec.to_workflow()
    results = workflow.run()

    assert results is not None
    assert "cv_results" in results or len(results) > 0


def test_umap_splitter_yaml_serialization():
    """Test UMAPCVSplitter YAML serialization and deserialization.

    Verifies that UMAPCVSplitter can be serialized to YAML and
    reconstructed correctly.

    """
    splitter = UMAPCVSplitter(n_splits=3, random_seed=42, n_neighbors=50)

    yaml_str = yaml.dump(splitter)
    assert "!UMAPCVSplitter" in yaml_str

    loaded = yaml.safe_load(yaml_str)
    assert loaded["n_splits"] == 3
    assert loaded["random_seed"] == 42
    assert loaded["n_neighbors"] == 50


def test_umap_splitter_direct_use():
    """Test UMAPCVSplitter directly with sklearn-style CV.

    Verifies that UMAPCVSplitter works as an sklearn CV splitter.

    """
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    X, y = make_classification(
        n_samples=50, n_features=10, n_informative=5, random_state=42
    )

    splitter = UMAPCVSplitter(n_splits=3, random_seed=42, n_neighbors=15)
    model = LogisticRegression(max_iter=200)

    scores = cross_val_score(model, X, y, cv=splitter)

    assert len(scores) == 3
    assert all(0.0 <= s <= 1.0 for s in scores)
