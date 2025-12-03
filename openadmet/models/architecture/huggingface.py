"""HuggingFace transformer model implementations.

This module provides sklearn-compatible wrappers for HuggingFace transformers,
enabling integration with OpenADMET's sklearn driver, Optuna HPO, and nested CV.

Example usage:
    model = HuggingFaceRegressorModel(
        model_id="ibm/MoLFormer-XL-both-10pct",
        learning_rate=5e-5,
        num_train_epochs=20,
    )
    model.build()
    model.train(X_smiles, y_targets)
    predictions = model.predict(X_test)
"""

import json
import tempfile
from abc import abstractmethod
from os import PathLike
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import torch
from loguru import logger
from pydantic import field_validator
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from openadmet.models.architecture.model_base import ModelBase, models
from openadmet.models.drivers import DriverType


class HuggingFaceModelBase(ModelBase):
    """Base class for HuggingFace models using sklearn driver.

    Uses HuggingFace's native save_pretrained/from_pretrained for serialization
    instead of joblib pickle.
    """

    _model_save_name: ClassVar[str] = "model"
    _driver_type: DriverType = DriverType.SKLEARN

    def save(self, path: PathLike):
        """Save the model using HuggingFace's native method."""
        if self.estimator is None:
            raise ValueError("Model is not built, cannot save")
        self.estimator.save(path)

    def load(self, path: PathLike):
        """Load the model using HuggingFace's native method."""
        self.estimator.load(path)

    def make_new(self) -> "HuggingFaceModelBase":
        """Copy parameters to a new model instance without the estimator."""
        return self.__class__(**self.model_dump(exclude={"estimator"}))

    def serialize(
        self,
        param_path: PathLike = "model.json",
        serial_path: PathLike = "model",
    ):
        """Save config as JSON and model using HF's save_pretrained."""
        with open(param_path, "w") as f:
            f.write(self.model_dump_json(indent=2))
        self.save(serial_path)

    @classmethod
    def deserialize(
        cls,
        param_path: PathLike = "model.json",
        serial_path: PathLike = "model",
    ):
        """Load config from JSON and model using HF's from_pretrained."""
        with open(param_path) as f:
            mod_params = json.load(f)
        instance = cls(**mod_params)
        instance.build()
        instance.load(serial_path)
        return instance


class HuggingFaceEstimatorBase(BaseEstimator):
    """Base sklearn-compatible wrapper for HuggingFace transformers.

    This wrapper enables HuggingFace models to work with sklearn's
    cross-validation, GridSearchCV, and Optuna's OptunaSearchCV.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier (e.g., "ibm/MoLFormer-XL-both-10pct")
    learning_rate : float
        Learning rate for fine-tuning
    num_train_epochs : int
        Maximum number of training epochs
    per_device_train_batch_size : int
        Batch size per device during training
    weight_decay : float
        Weight decay for AdamW optimizer
    warmup_ratio : float
        Ratio of total steps for learning rate warmup
    early_stopping_patience : int
        Number of epochs with no improvement before stopping.
        Set to 0 to disable early stopping.
    accelerator : str
        Device to use ("cpu", "cuda", "auto")
    trust_remote_code : bool
        Whether to trust remote code from HuggingFace Hub
    """

    # Subclasses must define problem_type
    _problem_type: ClassVar[str]

    def __init__(
        self,
        model_id="ibm/MoLFormer-XL-both-10pct",
        learning_rate=5e-5,
        num_train_epochs=200,
        per_device_train_batch_size=32,
        weight_decay=0.01,
        warmup_ratio=0.1,
        dropout=0.1,
        lr_scheduler_type="linear",
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        label_smoothing_factor=0.0,
        early_stopping_patience=5,
        accelerator="auto",
        trust_remote_code=True,
    ):
        # sklearn requirement: attribute names must match __init__ params
        # and values must be stored without modification
        self.model_id = model_id
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.dropout = dropout
        self.lr_scheduler_type = lr_scheduler_type
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon
        self.label_smoothing_factor = label_smoothing_factor
        self.early_stopping_patience = early_stopping_patience
        self.accelerator = accelerator
        self.trust_remote_code = trust_remote_code

        # Internal state (not sklearn params - prefixed with underscore)
        self._model = None
        self._tokenizer = None
        self._label_encoder = None

    def _get_device_config(self) -> tuple[bool, bool]:
        """Determine device configuration.

        Returns
        -------
        tuple[bool, bool]
            (use_cpu, use_bf16)
        """
        if self.accelerator == "auto":
            use_cpu = not torch.cuda.is_available()
        else:
            use_cpu = self.accelerator == "cpu"

        use_bf16 = not use_cpu and torch.cuda.is_bf16_supported()
        return use_cpu, use_bf16

    def _create_tokenizer(self):
        """Create and return the tokenizer."""
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=self.trust_remote_code
        )

    def _create_model(self, num_labels: int):
        """Create and return the model.

        Parameters
        ----------
        num_labels : int
            Number of output labels (1 for regression, n_classes for
            classification)
        """
        from transformers import (
            AutoConfig,
            AutoModelForSequenceClassification,
        )

        config = AutoConfig.from_pretrained(
            self.model_id,
            num_labels=num_labels,
            problem_type=self._problem_type,
            trust_remote_code=self.trust_remote_code,
        )
        # Apply dropout to both hidden layers and attention
        if hasattr(config, "hidden_dropout_prob"):
            config.hidden_dropout_prob = self.dropout
        if hasattr(config, "attention_probs_dropout_prob"):
            config.attention_probs_dropout_prob = self.dropout
        if hasattr(config, "classifier_dropout"):
            config.classifier_dropout = self.dropout

        return AutoModelForSequenceClassification.from_pretrained(
            self.model_id,
            config=config,
            trust_remote_code=self.trust_remote_code,
        )

    def _tokenize_dataset(self, dataset):
        """Tokenize a HuggingFace dataset.

        Parameters
        ----------
        dataset : datasets.Dataset
            Dataset with 'text' column

        Returns
        -------
        datasets.Dataset
            Tokenized dataset
        """

        def tokenize_fn(examples):
            return self._tokenizer(
                examples["text"],
                padding=True,
                truncation=True,
            )

        return dataset.map(tokenize_fn, batched=True)

    def _create_trainer(
        self,
        train_dataset,
        eval_dataset=None,
        compute_metrics=None,
        output_dir=None,
    ):
        """Create and return a HuggingFace Trainer.

        Parameters
        ----------
        train_dataset : datasets.Dataset
            Training dataset
        eval_dataset : datasets.Dataset, optional
            Evaluation dataset for early stopping
        compute_metrics : callable, optional
            Function to compute metrics
        output_dir : str, optional
            Output directory for checkpoints

        Returns
        -------
        transformers.Trainer
            Configured trainer
        """
        from transformers import (
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )

        use_cpu, use_bf16 = self._get_device_config()

        # Determine eval strategy based on early stopping
        use_early_stopping = (
            self.early_stopping_patience > 0 and eval_dataset is not None
        )

        args = TrainingArguments(
            output_dir=output_dir or tempfile.mkdtemp(),
            learning_rate=self.learning_rate,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_train_batch_size,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            lr_scheduler_type=self.lr_scheduler_type,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            max_grad_norm=self.max_grad_norm,
            adam_beta1=self.adam_beta1,
            adam_beta2=self.adam_beta2,
            adam_epsilon=self.adam_epsilon,
            label_smoothing_factor=self.label_smoothing_factor,
            use_cpu=use_cpu,
            bf16=use_bf16,
            report_to="none",
            logging_strategy="no",
            eval_strategy="epoch" if use_early_stopping else "no",
            save_strategy="epoch" if use_early_stopping else "no",
            load_best_model_at_end=use_early_stopping,
            metric_for_best_model="loss",
            greater_is_better=False,
            save_total_limit=1,
        )

        callbacks = []
        if use_early_stopping:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.early_stopping_patience
                )
            )

        return Trainer(
            model=self._model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorWithPadding(self._tokenizer),
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )

    @abstractmethod
    def _prepare_labels(self, y: np.ndarray) -> tuple[np.ndarray, int]:
        """Prepare labels for training.

        Parameters
        ----------
        y : np.ndarray
            Raw target values

        Returns
        -------
        tuple[np.ndarray, int]
            (processed_labels, num_labels)
        """
        pass

    @abstractmethod
    def _postprocess_predictions(self, logits: np.ndarray) -> np.ndarray:
        """Post-process model predictions.

        Parameters
        ----------
        logits : np.ndarray
            Raw model outputs

        Returns
        -------
        np.ndarray
            Processed predictions
        """
        pass

    def fit(self, X, y):
        """Train on SMILES strings (X) and targets (y).

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            SMILES strings
        y : array-like of shape (n_samples,)
            Target values

        Returns
        -------
        self : HuggingFaceEstimatorBase
            Fitted estimator
        """
        from datasets import Dataset
        from sklearn.model_selection import train_test_split

        # Convert to numpy
        X = np.asarray(X).ravel()
        y = np.asarray(y).ravel()

        # Initialize tokenizer
        self._tokenizer = self._create_tokenizer()

        # Prepare labels (normalization for regression, encoding for
        # classification)
        y_processed, num_labels = self._prepare_labels(y)

        # Build model
        self._model = self._create_model(num_labels)

        # Create dataset
        if self.early_stopping_patience > 0:
            # Split for early stopping validation
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_processed, test_size=0.1, random_state=42
            )
            train_dataset = Dataset.from_dict({
                "text": X_train.tolist(),
                "label": y_train.tolist(),
            })
            eval_dataset = Dataset.from_dict({
                "text": X_val.tolist(),
                "label": y_val.tolist(),
            })
            train_dataset = self._tokenize_dataset(train_dataset)
            eval_dataset = self._tokenize_dataset(eval_dataset)
        else:
            train_dataset = Dataset.from_dict({
                "text": X.tolist(),
                "label": y_processed.tolist(),
            })
            train_dataset = self._tokenize_dataset(train_dataset)
            eval_dataset = None

        # Train
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = self._create_trainer(
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                output_dir=tmp_dir,
            )
            trainer.train()

        return self

    def predict(self, X) -> np.ndarray:
        """Predict target values for SMILES strings.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            SMILES strings

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values
        """
        from datasets import Dataset
        from transformers import DataCollatorWithPadding

        if self._model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = np.asarray(X).ravel()

        # Create and tokenize dataset
        dataset = Dataset.from_dict({"text": X.tolist()})
        dataset = self._tokenize_dataset(dataset)
        dataset.set_format("torch", columns=["input_ids", "attention_mask"])

        # Run inference
        self._model.eval()
        device = next(self._model.parameters()).device

        data_collator = DataCollatorWithPadding(self._tokenizer)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.per_device_train_batch_size,
            collate_fn=data_collator,
        )

        all_logits = []
        with torch.inference_mode():
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = self._model(**batch)
                all_logits.append(outputs.logits.cpu().numpy())

        logits = np.concatenate(all_logits, axis=0)
        return self._postprocess_predictions(logits)

    def save(self, path: PathLike):
        """Save model, tokenizer, and state."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self._model.save_pretrained(path / "model")
        self._tokenizer.save_pretrained(path / "tokenizer")

        # Save additional state (subclasses add to this)
        state = self._get_save_state()
        with open(path / "state.json", "w") as f:
            json.dump(state, f)

    def load(self, path: PathLike):
        """Load model, tokenizer, and state."""
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        path = Path(path)

        self._model = AutoModelForSequenceClassification.from_pretrained(
            path / "model", trust_remote_code=self.trust_remote_code
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            path / "tokenizer", trust_remote_code=self.trust_remote_code
        )

        with open(path / "state.json") as f:
            state = json.load(f)
        self._load_save_state(state)

    @abstractmethod
    def _get_save_state(self) -> dict:
        """Get state dict for saving."""
        pass

    @abstractmethod
    def _load_save_state(self, state: dict):
        """Load state from dict."""
        pass


class HuggingFaceRegressor(HuggingFaceEstimatorBase, RegressorMixin):
    """sklearn-compatible HuggingFace regressor.

    Wraps HuggingFace transformers for regression tasks with target
    normalization.
    """

    _problem_type: ClassVar[str] = "regression"

    def __init__(
        self,
        model_id="ibm/MoLFormer-XL-both-10pct",
        learning_rate=5e-5,
        num_train_epochs=200,
        per_device_train_batch_size=32,
        weight_decay=0.01,
        warmup_ratio=0.1,
        dropout=0.1,
        lr_scheduler_type="linear",
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        label_smoothing_factor=0.0,
        early_stopping_patience=5,
        accelerator="auto",
        trust_remote_code=True,
    ):
        super().__init__(
            model_id=model_id,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            dropout=dropout,
            lr_scheduler_type=lr_scheduler_type,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_epsilon=adam_epsilon,
            label_smoothing_factor=label_smoothing_factor,
            early_stopping_patience=early_stopping_patience,
            accelerator=accelerator,
            trust_remote_code=trust_remote_code,
        )
        self._y_mean = 0.0
        self._y_std = 1.0

    def _prepare_labels(self, y: np.ndarray) -> tuple[np.ndarray, int]:
        """Normalize targets for regression."""
        self._y_mean = float(y.mean())
        self._y_std = float(y.std())
        if self._y_std == 0:
            self._y_std = 1.0
        y_norm = (y - self._y_mean) / self._y_std
        return y_norm, 1

    def _postprocess_predictions(self, logits: np.ndarray) -> np.ndarray:
        """Denormalize predictions."""
        preds = logits.squeeze(-1)
        return preds * self._y_std + self._y_mean

    def _get_save_state(self) -> dict:
        return {"y_mean": self._y_mean, "y_std": self._y_std}

    def _load_save_state(self, state: dict):
        self._y_mean = state["y_mean"]
        self._y_std = state["y_std"]


class HuggingFaceClassifier(HuggingFaceEstimatorBase, ClassifierMixin):
    """sklearn-compatible HuggingFace classifier.

    Wraps HuggingFace transformers for classification tasks with label
    encoding.
    """

    _problem_type: ClassVar[str] = "single_label_classification"

    def __init__(
        self,
        model_id="ibm/MoLFormer-XL-both-10pct",
        learning_rate=5e-5,
        num_train_epochs=200,
        per_device_train_batch_size=32,
        weight_decay=0.01,
        warmup_ratio=0.1,
        dropout=0.1,
        lr_scheduler_type="linear",
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        label_smoothing_factor=0.0,
        early_stopping_patience=5,
        accelerator="auto",
        trust_remote_code=True,
    ):
        super().__init__(
            model_id=model_id,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            dropout=dropout,
            lr_scheduler_type=lr_scheduler_type,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_epsilon=adam_epsilon,
            label_smoothing_factor=label_smoothing_factor,
            early_stopping_patience=early_stopping_patience,
            accelerator=accelerator,
            trust_remote_code=trust_remote_code,
        )
        self._classes = None

    def _prepare_labels(self, y: np.ndarray) -> tuple[np.ndarray, int]:
        """Encode labels for classification."""
        from sklearn.preprocessing import LabelEncoder

        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        self._classes = self._label_encoder.classes_
        return y_encoded, len(self._classes)

    def _postprocess_predictions(self, logits: np.ndarray) -> np.ndarray:
        """Convert logits to class predictions."""
        pred_indices = np.argmax(logits, axis=-1)
        return self._label_encoder.inverse_transform(pred_indices)

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            SMILES strings

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities
        """
        from datasets import Dataset
        from scipy.special import softmax
        from transformers import DataCollatorWithPadding

        if self._model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = np.asarray(X).ravel()

        dataset = Dataset.from_dict({"text": X.tolist()})
        dataset = self._tokenize_dataset(dataset)
        dataset.set_format("torch", columns=["input_ids", "attention_mask"])

        self._model.eval()
        device = next(self._model.parameters()).device

        data_collator = DataCollatorWithPadding(self._tokenizer)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.per_device_train_batch_size,
            collate_fn=data_collator,
        )

        all_logits = []
        with torch.inference_mode():
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = self._model(**batch)
                all_logits.append(outputs.logits.cpu().numpy())

        logits = np.concatenate(all_logits, axis=0)
        return softmax(logits, axis=-1)

    @property
    def classes_(self):
        """Return the classes."""
        return self._classes

    def _get_save_state(self) -> dict:
        return {"classes": self._classes.tolist()}

    def _load_save_state(self, state: dict):
        from sklearn.preprocessing import LabelEncoder

        self._classes = np.array(state["classes"])
        self._label_encoder = LabelEncoder()
        self._label_encoder.classes_ = self._classes


class HuggingFaceModel(HuggingFaceModelBase):
    """Base OpenADMET model class for HuggingFace transformers.

    Provides common configuration for regression and classification models.

    Attributes
    ----------
    model_id : str
        HuggingFace model identifier
    learning_rate : float
        Learning rate for fine-tuning
    num_train_epochs : int
        Maximum number of training epochs
    per_device_train_batch_size : int
        Batch size per device
    weight_decay : float
        Weight decay for regularization
    warmup_ratio : float
        LR warmup ratio
    early_stopping_patience : int
        Epochs without improvement before stopping (0 to disable)
    accelerator : str
        Device ("cpu", "cuda", "auto")
    trust_remote_code : bool
        Whether to trust remote code from HF Hub
    """

    # Subclasses must set this
    _estimator_class: ClassVar[type]

    model_id: str = "ibm/MoLFormer-XL-both-10pct"
    learning_rate: float = 5e-5
    num_train_epochs: int = 200
    per_device_train_batch_size: int = 32
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    dropout: float = 0.1
    lr_scheduler_type: Literal[
        "linear", "cosine", "cosine_with_restarts", "polynomial", "constant",
        "constant_with_warmup"
    ] = "linear"
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    label_smoothing_factor: float = 0.0
    early_stopping_patience: int = 5
    accelerator: Literal["cpu", "cuda", "auto"] = "auto"
    trust_remote_code: bool = True

    @field_validator("accelerator")
    @classmethod
    def validate_accelerator(cls, value):
        """Validate the accelerator parameter."""
        if value not in ["cpu", "cuda", "auto"]:
            raise ValueError("accelerator must be 'cpu', 'cuda', or 'auto'")
        return value

    def build(self):
        """Build the sklearn-compatible HuggingFace estimator."""
        if not self.estimator:
            self.estimator = self._estimator_class(
                model_id=self.model_id,
                learning_rate=self.learning_rate,
                num_train_epochs=self.num_train_epochs,
                per_device_train_batch_size=self.per_device_train_batch_size,
                weight_decay=self.weight_decay,
                warmup_ratio=self.warmup_ratio,
                dropout=self.dropout,
                lr_scheduler_type=self.lr_scheduler_type,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                max_grad_norm=self.max_grad_norm,
                adam_beta1=self.adam_beta1,
                adam_beta2=self.adam_beta2,
                adam_epsilon=self.adam_epsilon,
                label_smoothing_factor=self.label_smoothing_factor,
                early_stopping_patience=self.early_stopping_patience,
                accelerator=self.accelerator,
                trust_remote_code=self.trust_remote_code,
            )
        else:
            logger.warning("Model already exists, skipping build")

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the model.

        Parameters
        ----------
        X : np.ndarray
            SMILES strings
        y : np.ndarray
            Target values
        """
        self.build()
        self.estimator.fit(X, y)

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Predict on SMILES strings.

        Parameters
        ----------
        X : np.ndarray
            SMILES strings

        Returns
        -------
        np.ndarray
            Predictions with shape (n_samples, 1)
        """
        if not self.estimator:
            raise ValueError("Model not trained")
        preds = self.estimator.predict(X)
        return np.expand_dims(preds, axis=1)


@models.register("HuggingFaceRegressorModel")
class HuggingFaceRegressorModel(HuggingFaceModel):
    """HuggingFace regression model for molecular property prediction.

    Wraps HuggingFace transformers (e.g., MoLFormer) in an OpenADMET-compatible
    interface that works with sklearn driver and Optuna HPO.

    Example
    -------
    >>> model = HuggingFaceRegressorModel(
    ...     model_id="ibm/MoLFormer-XL-both-10pct",
    ...     learning_rate=5e-5,
    ... )
    >>> model.build()
    >>> model.train(smiles_array, target_array)
    >>> predictions = model.predict(test_smiles)
    """

    type: ClassVar[str] = "HuggingFaceRegressorModel"
    _estimator_class: ClassVar[type] = HuggingFaceRegressor

    @staticmethod
    def default_param_distributions() -> dict[str, dict]:
        """Get default hyperparameter search space for Optuna HPO.

        Returns
        -------
        dict
            Parameter distributions in Optuna format with keys:
            - learning_rate: Log-scale float between 1e-6 and 1e-3
            - weight_decay: Float between 0.0 and 0.3
            - warmup_ratio: Float between 0.0 and 0.2
            - dropout: Float between 0.0 and 0.5
            - lr_scheduler_type: Categorical scheduler types
            - max_grad_norm: Float between 0.5 and 2.0
        """
        return {
            "learning_rate": {
                "type": "float",
                "low": 1e-6,
                "high": 1e-3,
                "log": True,
            },
            "weight_decay": {
                "type": "float",
                "low": 0.0,
                "high": 0.3,
            },
            "warmup_ratio": {
                "type": "float",
                "low": 0.0,
                "high": 0.2,
            },
            "dropout": {
                "type": "float",
                "low": 0.0,
                "high": 0.5,
            },
            "lr_scheduler_type": {
                "type": "categorical",
                "choices": ["linear", "cosine", "constant_with_warmup"],
            },
            "max_grad_norm": {
                "type": "float",
                "low": 0.5,
                "high": 2.0,
            },
        }


@models.register("HuggingFaceClassifierModel")
class HuggingFaceClassifierModel(HuggingFaceModel):
    """HuggingFace classification model for molecular property prediction.

    Wraps HuggingFace transformers (e.g., MoLFormer) in an OpenADMET-compatible
    interface that works with sklearn driver and Optuna HPO.

    Example
    -------
    >>> model = HuggingFaceClassifierModel(
    ...     model_id="ibm/MoLFormer-XL-both-10pct",
    ...     learning_rate=5e-5,
    ... )
    >>> model.build()
    >>> model.train(smiles_array, class_labels)
    >>> predictions = model.predict(test_smiles)
    """

    type: ClassVar[str] = "HuggingFaceClassifierModel"
    _estimator_class: ClassVar[type] = HuggingFaceClassifier

    @staticmethod
    def default_param_distributions() -> dict[str, dict]:
        """Get default hyperparameter search space for Optuna HPO.

        Returns
        -------
        dict
            Parameter distributions in Optuna format with keys:
            - learning_rate: Log-scale float between 1e-6 and 1e-3
            - weight_decay: Float between 0.0 and 0.3
            - warmup_ratio: Float between 0.0 and 0.2
            - dropout: Float between 0.0 and 0.5
            - lr_scheduler_type: Categorical scheduler types
            - max_grad_norm: Float between 0.5 and 2.0
            - label_smoothing_factor: Float between 0.0 and 0.2
        """
        return {
            "learning_rate": {
                "type": "float",
                "low": 1e-6,
                "high": 1e-3,
                "log": True,
            },
            "weight_decay": {
                "type": "float",
                "low": 0.0,
                "high": 0.3,
            },
            "warmup_ratio": {
                "type": "float",
                "low": 0.0,
                "high": 0.2,
            },
            "dropout": {
                "type": "float",
                "low": 0.0,
                "high": 0.5,
            },
            "lr_scheduler_type": {
                "type": "categorical",
                "choices": ["linear", "cosine", "constant_with_warmup"],
            },
            "max_grad_norm": {
                "type": "float",
                "low": 0.5,
                "high": 2.0,
            },
            "label_smoothing_factor": {
                "type": "float",
                "low": 0.0,
                "high": 0.2,
            },
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : np.ndarray
            SMILES strings

        Returns
        -------
        np.ndarray
            Class probabilities with shape (n_samples, n_classes)
        """
        if not self.estimator:
            raise ValueError("Model not trained")
        return self.estimator.predict_proba(X)
