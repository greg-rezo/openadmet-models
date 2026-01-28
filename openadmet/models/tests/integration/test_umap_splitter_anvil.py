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
    """Test Anvil workflow with CV evaluation runs correctly.

    This test verifies that a basic Anvil workflow with CV evaluation
    completes successfully. UMAPCVSplitter is tested separately in
    test_umap_splitter_direct_use which tests it with sklearn's
    cross_val_score function.

    Args:
        tmp_path: Pytest temporary directory fixture.
        regression_data: Fixture providing test data path.

    """
    from pathlib import Path

    recipe = {
        "metadata": {
            "version": "v1",
            "name": "umap-cv-test",
            "build_number": 0,
            "description": "Test CV workflow",
            "tag": "test-cv",
            "authors": "Test",
            "email": "test@test.com",
            "date_created": "2024-01-01",
            "biotargets": ["TEST"],
            "tags": ["test"],
        },
        "data": {
            "type": "csv",
            "resource": str(regression_data),
            "input_col": "smiles",
            "target_cols": ["activity"],
        },
        "procedure": {
            "split": {
                "type": "ShuffleSplitter",
                "params": {"train_size": 0.8, "random_state": 42},
            },
            "feat": {
                "type": "FingerprintFeaturizer",
                "params": {"fp_type": "ecfp:4"},
            },
            "model": {"type": "RidgeModel", "params": {}},
            "train": {"type": "SKLearnBasicTrainer"},
        },
        "report": {
            "eval": [
                {"type": "RegressionMetrics"},
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

    recipe_path = tmp_path / "cv_recipe.yaml"
    with open(recipe_path, "w") as f:
        yaml.dump(recipe, f)

    output_dir = tmp_path / "output"
    spec = AnvilSpecification.from_recipe(recipe_path)
    workflow = spec.to_workflow()
    workflow.run(output_dir=output_dir)

    # Verify outputs exist
    assert Path(output_dir / "model.json").exists()
    assert Path(output_dir / "regression_metrics.json").exists()
    assert Path(output_dir / "anvil_recipe.yaml").exists()


def test_umap_splitter_yaml_serialization():
    """Test UMAPCVSplitter YAML serialization and deserialization.

    Verifies that UMAPCVSplitter can be serialized to YAML and
    reconstructed correctly.

    """
    splitter = UMAPCVSplitter(n_splits=3, random_seed=42, n_neighbors=50)

    yaml_str = yaml.dump(splitter)
    assert "!UMAPCVSplitter" in yaml_str

    loaded = yaml.safe_load(yaml_str)
    # safe_load returns UMAPCVSplitter object because constructor is registered
    assert loaded.n_splits == 3
    assert loaded.random_seed == 42
    assert loaded.n_neighbors == 50


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
