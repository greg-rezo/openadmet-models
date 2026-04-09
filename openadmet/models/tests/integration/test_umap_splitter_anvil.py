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
    # Use a diverse set of valid drug-like SMILES
    smiles_list = [
        "CCO",  # ethanol
        "CCCO",  # propanol
        "CCCCO",  # butanol
        "CC(C)O",  # isopropanol
        "CC(C)CO",  # isobutanol
        "CCCC",  # butane
        "CCCCC",  # pentane
        "CCCCCC",  # hexane
        "c1ccccc1",  # benzene
        "Cc1ccccc1",  # toluene
        "CCc1ccccc1",  # ethylbenzene
        "c1ccc(O)cc1",  # phenol
        "c1ccc(N)cc1",  # aniline
        "c1ccc(C)cc1",  # toluene alt
        "CC(=O)O",  # acetic acid
        "CCC(=O)O",  # propionic acid
        "CCCC(=O)O",  # butyric acid
        "CC(=O)OCC",  # ethyl acetate
        "CC(=O)N",  # acetamide
        "CCCCN",  # butylamine
        "CCCCCN",  # pentylamine
        "C1CCCCC1",  # cyclohexane
        "C1CCCC1",  # cyclopentane
        "C1CCC1",  # cyclobutane
        "c1ccc2ccccc2c1",  # naphthalene
        "c1ccc(Cl)cc1",  # chlorobenzene
        "c1ccc(F)cc1",  # fluorobenzene
        "c1ccc(Br)cc1",  # bromobenzene
        "CCN(CC)CC",  # triethylamine
        "CN(C)C",  # trimethylamine
        "CCOCC",  # diethyl ether
        "CCOCCOC",  # diglyme
        "CC(C)C",  # isobutane
        "CC(C)(C)C",  # neopentane
        "c1ccc(OC)cc1",  # anisole
        "c1ccc(CC)cc1",  # ethylbenzene alt
        "CCC(C)C",  # isopentane
        "CC=O",  # acetaldehyde
        "CCC=O",  # propionaldehyde
        "CCCC=O",  # butyraldehyde
        "CC(=O)C",  # acetone
        "CCC(=O)CC",  # 3-pentanone
        "C1CCOCC1",  # tetrahydropyran
        "C1CCOC1",  # tetrahydrofuran
        "c1ccncc1",  # pyridine
        "c1ccoc1",  # furan
        "c1ccsc1",  # thiophene
        "Clc1ccc(Cl)cc1",  # p-dichlorobenzene
        "c1cc(O)ccc1O",  # hydroquinone
        "CCN",  # ethylamine
    ]

    X, y = make_regression(
        n_samples=50, n_features=10, n_informative=5, random_state=42
    )
    df = pd.DataFrame({"smiles": smiles_list, "activity": y})
    csv_path = tmp_path / "regression_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_umap_splitter_anvil_workflow(tmp_path, regression_data):
    """Test UMAPCVSplitter in Anvil workflow via custom_outer_cv.

    This test verifies that UMAPCVSplitter can be used as custom_outer_cv
    in the SKLearnRepeatedNestedKFoldCrossValidation evaluator within
    an Anvil workflow, configured via YAML.

    Args:
        tmp_path: Pytest temporary directory fixture.
        regression_data: Fixture providing test data path.

    """
    from pathlib import Path

    # Write YAML directly to use !UMAPCVSplitter tag
    recipe_yaml = f"""
metadata:
  version: v1
  name: umap-cv-test
  build_number: 0
  description: Test UMAPCVSplitter in Anvil
  tag: test-umap-cv
  authors: Test
  email: test@test.com
  date_created: "2024-01-01"
  biotargets:
    - TEST
  tags:
    - test

data:
  type: csv
  resource: "{regression_data}"
  input_col: smiles
  target_cols:
    - activity

procedure:
  split:
    type: ShuffleSplitter
    params:
      train_size: 0.8
      random_state: 42
  feat:
    type: FingerprintFeaturizer
    params:
      fp_type: "ecfp:4"
  model:
    type: RidgeModel
    params: {{}}
  train:
    type: SKLearnBasicTrainer

report:
  eval:
    - type: RegressionMetrics
    - type: SKLearnRepeatedNestedKFoldCrossValidation
      params:
        n_splits: 2
        n_repeats: 1
        random_state: 42
        inner_cv: 2
        n_trials: 2
        custom_outer_cv: !UMAPCVSplitter
          n_splits: 2
          random_seed: 42
          n_neighbors: 15
          min_dist: 0.1
        param_distributions:
          alpha:
            type: float
            low: 0.01
            high: 10.0
            log: true
"""

    recipe_path = tmp_path / "umap_cv_recipe.yaml"
    with open(recipe_path, "w") as f:
        f.write(recipe_yaml)

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
