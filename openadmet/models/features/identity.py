"""Identity/passthrough featurizer for models that handle their own featurization."""

from collections.abc import Iterable
from typing import ClassVar

import numpy as np

from openadmet.models.features.feature_base import FeaturizerBase, featurizers

__all__ = ["IdentityFeaturizer"]


@featurizers.register("IdentityFeaturizer")
class IdentityFeaturizer(FeaturizerBase):
    """
    Identity featurizer that passes through SMILES strings unchanged.

    This featurizer is intended for models that handle their own featurization
    internally, such as HuggingFace transformers which use their own tokenizers.
    It simply returns the input SMILES strings as a numpy array.

    Attributes
    ----------
    type : ClassVar[str]
        The type of the featurizer.

    """

    type: ClassVar[str] = "IdentityFeaturizer"

    def featurize(self, smiles: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
        """
        Pass through SMILES strings as features.

        Parameters
        ----------
        smiles : Iterable[str]
            List or iterable of SMILES strings.

        Returns
        -------
        tuple
            Tuple of (features, indices). Features is a 1D numpy array of SMILES
            strings and indices is a 1D numpy array of all indices (all samples
            are considered valid).

        """
        smiles_array = np.array(list(smiles))
        indices = np.arange(len(smiles_array))
        return smiles_array, indices
