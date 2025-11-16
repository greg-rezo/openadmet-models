"""Unit tests for anvil_nested_optuna module."""

from __future__ import annotations

import numpy as np
import pytest
from optuna.distributions import CategoricalDistribution, FloatDistribution
from optuna.integration import OptunaSearchCV
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from anvil_nested_optuna import (
    NestedSearchConfig,
    _make_optuna_search,
    _make_outer_cv,
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
        outer_n_splits=3,
        outer_repeats=1,
        outer_shuffle=True,
        outer_random_state=42,
        inner_cv=2,
        n_trials=5,
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
    assert cfg.outer_shuffle is True
    assert cfg.outer_random_state == 42
    assert cfg.inner_cv == 3
    assert cfg.n_trials == 50
    assert cfg.timeout_per_trial_s is None
    assert cfg.sampler_seed is None
    assert cfg.optuna_n_jobs == 1
    assert cfg.scoring is None
    assert cfg.n_jobs_outer == 1


def test_make_outer_cv_stratified_kfold(basic_config):
    """Test _make_outer_cv creates StratifiedKFold for single repeat."""
    from sklearn.model_selection import StratifiedKFold

    cfg = basic_config
    cfg.outer_repeats = 1
    outer_cv = _make_outer_cv(cfg)

    assert isinstance(outer_cv, StratifiedKFold)
    assert outer_cv.n_splits == cfg.outer_n_splits
    assert outer_cv.shuffle == cfg.outer_shuffle
    assert outer_cv.random_state == cfg.outer_random_state


def test_make_outer_cv_repeated_stratified_kfold(
    basic_config, small_classification_dataset
):
    """Test _make_outer_cv creates RepeatedStratifiedKFold for repeats."""
    from sklearn.model_selection import RepeatedStratifiedKFold

    cfg = basic_config
    cfg.outer_repeats = 2
    outer_cv = _make_outer_cv(cfg)

    assert isinstance(outer_cv, RepeatedStratifiedKFold)
    assert outer_cv.n_repeats == cfg.outer_repeats
    assert outer_cv.random_state == cfg.outer_random_state

    # Verify it produces correct number of splits when used
    X, y = small_classification_dataset
    n_splits = sum(1 for _ in outer_cv.split(X, y))
    assert n_splits == cfg.outer_n_splits * cfg.outer_repeats


def test_make_optuna_search(base_estimator, param_distributions, basic_config):  # noqa: E501
    """Test _make_optuna_search creates OptunaSearchCV object."""
    search = _make_optuna_search(base_estimator, param_distributions, basic_config)

    assert isinstance(search, OptunaSearchCV)
    assert search.n_trials == basic_config.n_trials
    assert search.cv == basic_config.inner_cv
    assert search.scoring == basic_config.scoring
    assert search.n_jobs == basic_config.optuna_n_jobs


def test_make_optuna_search_with_seed(
    base_estimator, param_distributions, basic_config
):
    """Test _make_optuna_search uses TPESampler with seed."""
    from optuna.samplers import TPESampler

    cfg = basic_config
    cfg.sampler_seed = 123

    search = _make_optuna_search(base_estimator, param_distributions, cfg)

    # Verify TPESampler is used (seed is internal implementation detail)
    assert isinstance(search.study.sampler, TPESampler)


def test_make_optuna_search_without_seed(
    base_estimator, param_distributions, basic_config
):
    """Test _make_optuna_search creates TPESampler without seed."""
    from optuna.samplers import TPESampler

    cfg = basic_config
    cfg.sampler_seed = None

    search = _make_optuna_search(base_estimator, param_distributions, cfg)

    assert isinstance(search.study.sampler, TPESampler)


def test_run_nested_optuna_search_returns_expected_keys(
    small_classification_dataset,
    base_estimator,
    param_distributions,
    basic_config,
):
    """Test run_nested_optuna_search returns correct result keys."""
    X, y = small_classification_dataset

    results = run_nested_optuna_search(
        X, y, base_estimator, param_distributions, basic_config
    )

    assert "outer_cv_results" in results
    assert "outer_best_params" in results
    assert "outer_best_scores" in results
    assert "estimators" in results


def test_run_nested_optuna_search_correct_number_of_folds(
    small_classification_dataset,
    base_estimator,
    param_distributions,
    basic_config,
):
    """Test run_nested_optuna_search creates correct number of folds."""
    X, y = small_classification_dataset

    results = run_nested_optuna_search(
        X, y, base_estimator, param_distributions, basic_config
    )

    expected_folds = basic_config.outer_n_splits * basic_config.outer_repeats
    assert len(results["outer_best_params"]) == expected_folds
    assert len(results["outer_best_scores"]) == expected_folds
    assert len(results["estimators"]) == expected_folds


def test_run_nested_optuna_search_estimators_are_fitted(
    small_classification_dataset,
    base_estimator,
    param_distributions,
    basic_config,
):
    """Test run_nested_optuna_search returns fitted estimators."""
    from optuna.integration import OptunaSearchCV

    X, y = small_classification_dataset

    results = run_nested_optuna_search(
        X, y, base_estimator, param_distributions, basic_config
    )

    for estimator in results["estimators"]:
        assert isinstance(estimator, OptunaSearchCV)
        assert hasattr(estimator, "best_params_")
        assert hasattr(estimator, "best_score_")


def test_run_nested_optuna_search_best_params_correct_format(
    small_classification_dataset,
    base_estimator,
    param_distributions,
    basic_config,
):
    """Test run_nested_optuna_search best_params have expected keys."""
    X, y = small_classification_dataset

    results = run_nested_optuna_search(
        X, y, base_estimator, param_distributions, basic_config
    )

    for params in results["outer_best_params"]:
        assert params is not None
        assert "clf__C" in params
        assert "clf__penalty" in params
        assert params["clf__penalty"] == "l2"
        assert 1e-2 <= params["clf__C"] <= 1e1


def test_run_nested_optuna_search_scores_are_valid(
    small_classification_dataset,
    base_estimator,
    param_distributions,
    basic_config,
):
    """Test run_nested_optuna_search scores are in valid range."""
    X, y = small_classification_dataset

    results = run_nested_optuna_search(
        X, y, base_estimator, param_distributions, basic_config
    )

    for score in results["outer_best_scores"]:
        assert score is not None
        assert 0.0 <= score <= 1.0


def test_run_nested_optuna_search_with_repeats(
    small_classification_dataset,
    base_estimator,
    param_distributions,
    basic_config,
):
    """Test run_nested_optuna_search with repeated CV."""
    X, y = small_classification_dataset

    cfg = basic_config
    cfg.outer_n_splits = 2
    cfg.outer_repeats = 2

    results = run_nested_optuna_search(X, y, base_estimator, param_distributions, cfg)

    expected_folds = 2 * 2
    assert len(results["estimators"]) == expected_folds


def test_run_nested_optuna_search_cv_results_structure(
    small_classification_dataset,
    base_estimator,
    param_distributions,
    basic_config,
):
    """Test outer_cv_results has expected structure from cross_validate."""
    X, y = small_classification_dataset

    results = run_nested_optuna_search(
        X, y, base_estimator, param_distributions, basic_config
    )

    cv_results = results["outer_cv_results"]
    assert "test_score" in cv_results
    assert "fit_time" in cv_results
    assert "score_time" in cv_results
    assert "estimator" in cv_results
    assert len(cv_results["test_score"]) == basic_config.outer_n_splits
