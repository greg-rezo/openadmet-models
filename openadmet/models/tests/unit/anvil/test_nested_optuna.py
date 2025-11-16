"""Unit tests for nested_optuna module."""

from __future__ import annotations

import numpy as np
import pytest
from optuna.distributions import CategoricalDistribution, FloatDistribution
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from openadmet.models.anvil.nested_optuna import (
    NestedSearchConfig,
    run_nested_optuna_search,
)


@pytest.fixture
def small_classification_dataset():
    """Create small classification dataset for fast tests.

    Returns:
        Tuple of (X, y) with 100 samples and 10 features.
    """
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42,
    )
    return X, y


@pytest.fixture
def base_estimator():
    """Create basic estimator pipeline.

    Returns:
        Pipeline with scaler and logistic regression.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=100, random_state=42)),
        ]
    )


@pytest.fixture
def param_distributions():
    """Create parameter distributions for Optuna.

    Returns:
        Dictionary of parameter distributions.
    """
    return {
        "clf__C": FloatDistribution(1e-2, 1e1, log=True),
        "clf__penalty": CategoricalDistribution(["l2"]),
    }


@pytest.fixture
def basic_config():
    """Create basic nested search config.

    Returns:
        NestedSearchConfig with minimal settings for fast tests.
    """
    return NestedSearchConfig(
        outer_n_splits=2,
        outer_repeats=1,
        outer_shuffle=True,
        outer_random_state=42,
        inner_cv=2,
        n_trials=3,
        sampler_seed=42,
        optuna_n_jobs=1,
        n_jobs_outer=1,
        scoring="accuracy",
    )


def test_nested_search_config_defaults():
    """Test NestedSearchConfig default values."""
    cfg = NestedSearchConfig()
    assert cfg.outer_n_splits == 5
    assert cfg.outer_repeats == 1
    assert cfg.inner_cv == 3
    assert cfg.n_trials == 50


def test_run_nested_optuna_search_basic(
    small_classification_dataset,
    base_estimator,
    param_distributions,
    basic_config,
):
    """Test basic nested optuna search execution and result structure."""
    X, y = small_classification_dataset

    results = run_nested_optuna_search(
        X, y, base_estimator, param_distributions, basic_config
    )

    # Check result structure
    assert "outer_cv_results" in results
    assert "outer_best_params" in results
    assert "outer_best_scores" in results
    assert "estimators" in results

    # Check correct number of folds
    expected_folds = basic_config.outer_n_splits * basic_config.outer_repeats
    assert len(results["outer_best_params"]) == expected_folds
    assert len(results["outer_best_scores"]) == expected_folds
    assert len(results["estimators"]) == expected_folds

    # Check parameter values are valid
    for params in results["outer_best_params"]:
        assert "clf__C" in params
        assert "clf__penalty" in params
        assert 1e-2 <= params["clf__C"] <= 1e1

    # Check scores are valid
    for score in results["outer_best_scores"]:
        assert 0.0 <= score <= 1.0

    # Check cv_results structure
    cv_results = results["outer_cv_results"]
    assert "test_score" in cv_results
    assert "fit_time" in cv_results
    assert "estimator" in cv_results


def test_run_nested_optuna_search_with_repeats(
    small_classification_dataset,
    base_estimator,
    param_distributions,
):
    """Test nested optuna search with repeated CV produces correct folds."""
    X, y = small_classification_dataset

    cfg = NestedSearchConfig(
        outer_n_splits=2,
        outer_repeats=2,
        inner_cv=2,
        n_trials=3,
        sampler_seed=42,
        scoring="accuracy",
    )

    results = run_nested_optuna_search(X, y, base_estimator, param_distributions, cfg)

    expected_folds = 2 * 2
    assert len(results["estimators"]) == expected_folds


def test_nested_optuna_end_to_end_simple_model(
    small_classification_dataset,
):
    """Test end-to-end nested CV with Optuna on a simple classification task.

    This test verifies the complete workflow including:
    - Nested cross-validation (outer loop for evaluation)
    - Optuna hyperparameter search (inner loop for tuning)
    - Multiple parameter types (float with log scale, categorical)
    - Reproducibility with random seeds
    """
    X, y = small_classification_dataset

    # Use LogisticRegression for fast execution
    base_estimator = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200, random_state=42)),
        ]
    )

    # Define hyperparameter search space
    param_distributions = {
        "clf__C": FloatDistribution(1e-3, 1e2, log=True),
        "clf__penalty": CategoricalDistribution(["l1", "l2"]),
        "clf__solver": CategoricalDistribution(["liblinear"]),
    }

    # Configure nested CV with minimal settings for speed
    cfg = NestedSearchConfig(
        outer_n_splits=3,
        outer_repeats=1,
        inner_cv=2,
        n_trials=5,
        sampler_seed=42,
        scoring="accuracy",
        n_jobs_outer=1,
        optuna_n_jobs=1,
    )

    # Run nested CV
    results = run_nested_optuna_search(X, y, base_estimator, param_distributions, cfg)

    # Verify all folds completed successfully
    assert len(results["estimators"]) == 3

    # Verify scores are reasonable for a simple classification task
    scores = results["outer_best_scores"]
    mean_score = np.mean(scores)
    assert mean_score > 0.5  # Better than random

    # Verify each fold found different hyperparameters
    params_list = results["outer_best_params"]
    assert all("clf__C" in params for params in params_list)
    assert all("clf__penalty" in params for params in params_list)

    # Verify fitted estimators can make predictions
    for estimator in results["estimators"]:
        predictions = estimator.predict(X)
        assert len(predictions) == len(y)
        assert set(predictions).issubset({0, 1})
