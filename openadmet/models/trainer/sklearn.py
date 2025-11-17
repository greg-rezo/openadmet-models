"""Trainers for sklearn models."""

from typing import Any

from loguru import logger
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from sklearn.model_selection import GridSearchCV

from openadmet.models.anvil.nested_optuna import (
    NestedSearchConfig,
    run_nested_optuna_search,
)
from openadmet.models.drivers import DriverType
from openadmet.models.trainer.trainer_base import TrainerBase, trainers


class SKLearnTrainer(TrainerBase):
    """Base trainer for sklearn models."""

    _driver_type: DriverType = DriverType.SKLEARN


@trainers.register("SKLearnBasicTrainer")
class SKlearnBasicTrainer(SKLearnTrainer):
    """Basic trainer for sklearn models."""

    def train(self, X: Any, y: Any):
        """
        Train the model.

        Parameters
        ----------
        X : Any
            Feature data.
        y : Any
            Target data.

        Returns
        -------
        ModelBase
            The trained model.

        """
        sklearn_model = self.model.estimator
        sklearn_model.fit(X, y)
        self.model.estimator = sklearn_model
        return self.model

    def build(self):
        """Unused method for sklearn models."""
        pass


class SKLearnSearchTrainer(SKLearnTrainer):
    """
    Trainer for sklearn models with search.

    Attributes
    ----------
    search : Any
        The search object (e.g., GridSearchCV).

    """

    _search: Any

    @property
    def search(self):
        """Return search object (e.g., GridSearchCV)."""
        return self._search

    @search.setter
    def search(self, value):
        """Set search object (e.g., GridSearchCV)."""
        self._search = value

    def build(self):
        """Unused method for sklearn models."""
        pass


@trainers.register("SKLearnGridSearchTrainer")
class SKLearnGridSearchTrainer(SKLearnSearchTrainer):
    """
    Trainer for sklearn models with grid search.

    Attributes
    ----------
    param_grid : dict
        The parameter grid for grid search.

    """

    param_grid: dict = {}

    def train(self, X: Any, y: Any):
        """
        Train the model.

        Parameters
        ----------
        X : Any
            Featurized data.
        y : Any
            Target data.

        Returns
        -------
        ModelBase
            The trained model.

        """
        # Handle imputation if the model has a fitted imputer
        # This ensures data is imputed before GridSearchCV cloning
        if hasattr(self.model, "_imputer") and self.model._imputer is not None:
            logger.info("Imputing NaN values before GridSearchCV")
            X = self.model._imputer.fit_transform(X)

        sklearn_model = self.model.estimator
        self.search = GridSearchCV(sklearn_model, param_grid=self.param_grid)
        self.search.fit(X, y)

        # Set the params and model to the best found
        self.model.estimator = self.search.best_estimator_
        self.model.__dict__.update(self.model.estimator.get_params())

        logger.info(f"Best params: {self.model.estimator.get_params()}")
        return self.model


@trainers.register("SKLearnOptunaTrainer")
class SKLearnOptunaTrainer(SKLearnSearchTrainer):
    """
    Trainer for sklearn models with nested CV using Optuna.

    Performs nested cross-validation with Optuna hyperparameter search
    in the inner loop.

    Attributes
    ----------
    param_distributions : dict
        Parameter distributions for Optuna search. Each key is a parameter
        name and each value is a dict with 'type' and distribution args.
        Example:
            {
                "learning_rate": {"type": "float", "low": 0.01, "high": 0.3,
                                  "log": True},
                "n_estimators": {"type": "int", "low": 10, "high": 100}
            }
    outer_n_splits : int
        Number of outer CV splits (default: 5).
    outer_repeats : int
        Number of outer CV repeats (default: 1).
    inner_cv : int
        Number of inner CV splits for Optuna (default: 3).
    n_trials : int
        Number of Optuna trials (default: 50).
    sampler_seed : int | None
        Random seed for Optuna sampler (default: None).
    scoring : str | None
        Scoring metric for evaluation (default: None).
    n_jobs_outer : int
        Number of parallel jobs for outer CV (default: 1).

    """

    param_distributions: dict[str, dict[str, Any]] = {}
    outer_n_splits: int = 5
    outer_repeats: int = 1
    inner_cv: int = 3
    n_trials: int = 50
    sampler_seed: int | None = None
    scoring: str | None = None
    n_jobs_outer: int = 1

    def _convert_param_distributions(
        self, param_dists: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Convert parameter distributions from dict format to Optuna objects.

        Args:
            param_dists: Parameter distributions in dict format.

        Returns:
            Dictionary with Optuna distribution objects.

        """
        converted = {}
        for param_name, dist_config in param_dists.items():
            dist_type = dist_config["type"]
            if dist_type == "float":
                converted[param_name] = FloatDistribution(
                    low=dist_config["low"],
                    high=dist_config["high"],
                    log=dist_config.get("log", False),
                )
            elif dist_type == "int":
                converted[param_name] = IntDistribution(
                    low=dist_config["low"],
                    high=dist_config["high"],
                    log=dist_config.get("log", False),
                )
            elif dist_type == "categorical":
                converted[param_name] = CategoricalDistribution(
                    choices=dist_config["choices"]
                )
            else:
                raise ValueError(f"Unknown distribution type: {dist_type}")
        return converted

    def train(self, X: Any, y: Any):
        """
        Train the model using nested CV with Optuna.

        Parameters
        ----------
        X : Any
            Featurized data.
        y : Any
            Target data.

        Returns
        -------
        ModelBase
            The trained model with best estimator from first outer fold.

        """
        # Handle imputation if the model has a fitted imputer
        # This ensures data is imputed before Optuna search
        if hasattr(self.model, "_imputer") and self.model._imputer is not None:
            logger.info("Imputing NaN values before Optuna search")
            X = self.model._imputer.fit_transform(X)

        sklearn_model = self.model.estimator

        # Convert param distributions from dict to Optuna objects
        optuna_dists = self._convert_param_distributions(self.param_distributions)

        # Configure nested search
        cfg = NestedSearchConfig(
            outer_n_splits=self.outer_n_splits,
            outer_repeats=self.outer_repeats,
            inner_cv=self.inner_cv,
            n_trials=self.n_trials,
            sampler_seed=self.sampler_seed,
            scoring=self.scoring,
            n_jobs_outer=self.n_jobs_outer,
        )

        # Run nested CV
        results = run_nested_optuna_search(X, y, sklearn_model, optuna_dists, cfg)

        # Use the best estimator from the first outer fold
        # (alternative: could retrain on full data with best params)
        self.search = results["estimators"][0]
        self.model.estimator = self.search.best_estimator_
        self.model.__dict__.update(self.model.estimator.get_params())

        logger.info(
            f"Nested CV mean score: "
            f"{sum(results['outer_best_scores']) / len(results['outer_best_scores']):.4f}"  # noqa: E501
        )
        logger.info(f"Best params (fold 0): {self.search.best_params_}")

        return self.model
