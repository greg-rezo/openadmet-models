#!/usr/bin/env python
"""Focused test for MPNN hyperparameter YAML serialization."""

import tempfile
import yaml
from pathlib import Path
from chemprop import nn
from openadmet.models.architecture.chemprop import MPNN


def test_mpnn_hparams_yaml_serialization():
    """Test that MPNN hyperparameters can be saved to YAML."""

    # Create message passing and aggregation
    mp = nn.BondMessagePassing(d_h=100, depth=2, dropout=0.0)
    aggr = nn.NormAggregation()

    # Create FFN
    ffn = nn.RegressionFFN(
        n_tasks=1,
        input_dim=100,
        hidden_dim=100,
        n_layers=1,
        dropout=0.0,
    )

    # Create metrics
    metric_list = [nn.metrics.MSE(), nn.metrics.MAE()]

    # Create MPNN model (our wrapper)
    mpnn = MPNN(
        message_passing=mp,
        agg=aggr,
        predictor=ffn,
        batch_norm=False,
        metrics=metric_list,
        warmup_epochs=2,
        init_lr=1e-4,
        max_lr=1e-3,
        final_lr=1e-4,
    )

    print("MPNN created successfully")
    print(f"hparams keys: {list(mpnn.hparams.keys())}")

    # Test individual hyperparameter serialization
    print("\nTesting individual hparam serialization:")
    for key, value in mpnn.hparams.items():
        try:
            yaml.dump({key: value})
            print(f"  ✓ {key}: {type(value)}")
        except Exception as e:
            print(f"  ✗ {key}: {type(value)} - {e}")
            raise

    # Test full hparams serialization (what PyTorch Lightning does)
    print("\nTesting full hparams YAML serialization:")
    try:
        yaml_str = yaml.dump(dict(mpnn.hparams))
        print(f"✓ Full hparams serialization succeeded ({len(yaml_str)} chars)")
    except Exception as e:
        print(f"✗ Full hparams serialization failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Test save_hparams_to_yaml (exactly what PyTorch Lightning does during checkpointing)
    print("\nTesting save_hparams_to_yaml (Lightning's method):")
    try:
        from lightning.pytorch.core.saving import save_hparams_to_yaml

        with tempfile.TemporaryDirectory() as tmpdir:
            hparams_file = Path(tmpdir) / "hparams.yaml"
            save_hparams_to_yaml(hparams_file, mpnn.hparams)
            print(f"✓ save_hparams_to_yaml succeeded")

            # Verify the file was created and is valid YAML
            with open(hparams_file) as f:
                loaded = yaml.safe_load(f)
            print(f"✓ YAML file is valid and loadable")
            print(f"  Saved hparams: {list(loaded.keys())}")

    except Exception as e:
        print(f"✗ save_hparams_to_yaml failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    test_mpnn_hparams_yaml_serialization()
    print("\n✅ All MPNN hyperparameter serialization tests passed!")
