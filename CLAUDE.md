# OpenADMET Models - Claude Code Guide

This file provides guidance to Claude Code when working with the OpenADMET Models repository.

## Project Overview

OpenADMET Models is a Python library for building and training ADMET (Absorption, Distribution, Metabolism, Excretion, and Toxicity) prediction models using various machine learning architectures including ChemProp, Ridge, and other models.

## Development Commands

### Environment Setup

The project uses Docker for testing to ensure consistent environments:

```bash
# Pull the latest Docker image (if available)
docker pull us-central1-docker.pkg.dev/gke-test-421317/flyte/internal-openadmet:linux-<hash>

# Or use a local image
docker images | grep openadmet
```

### Running Tests Locally

#### Using Docker (Recommended)

Run tests using Docker with mounted source code to test local changes:

```bash
# Run all tests
docker run --rm \
  -v /tmp/openadmet-models:/tmp/openadmet-models \
  --workdir /tmp/openadmet-models \
  --user root \
  <image-name> \
  bash -c "source /root/.venv/bin/activate && PYTHONPATH=/tmp/openadmet-models pytest -v"

# Run specific test file
docker run --rm \
  -v /tmp/openadmet-models:/tmp/openadmet-models \
  --workdir /tmp/openadmet-models \
  --user root \
  <image-name> \
  bash -c "source /root/.venv/bin/activate && PYTHONPATH=/tmp/openadmet-models pytest openadmet/models/tests/integration/test_chemprop_cv.py -v"

# Run specific test function
docker run --rm \
  -v /tmp/openadmet-models:/tmp/openadmet-models \
  --workdir /tmp/openadmet-models \
  --user root \
  <image-name> \
  bash -c "source /root/.venv/bin/activate && PYTHONPATH=/tmp/openadmet-models pytest openadmet/models/tests/integration/test_chemprop_cv.py::test_chemprop_cpu_cv_recipe -v -s"
```

Replace `<image-name>` with the actual Docker image name from `docker images`.

#### Running Focused Tests

For quick validation of specific functionality:

```bash
# Test MPNN hyperparameter serialization
docker run --rm \
  -v /tmp/openadmet-models:/tmp/openadmet-models \
  --workdir /tmp/openadmet-models \
  --user root \
  <image-name> \
  bash -c "source /root/.venv/bin/activate && PYTHONPATH=/tmp/openadmet-models python test_mpnn_hparams.py"

# Test ChemProp YAML serialization
docker run --rm \
  -v /tmp/openadmet-models:/tmp/openadmet-models \
  --workdir /tmp/openadmet-models \
  --user root \
  <image-name> \
  bash -c "source /root/.venv/bin/activate && PYTHONPATH=/tmp/openadmet-models python test_chemprop_yaml.py"
```

### Testing Best Practices

- Always mount the source code directory (`-v /tmp/openadmet-models:/tmp/openadmet-models`) to test local changes
- Set `PYTHONPATH=/tmp/openadmet-models` to ensure proper module imports
- Use `-v` flag for verbose output to see which tests are running
- Use `-s` flag to see print statements and logging output
- Run focused unit tests during development, full integration tests before committing

## Code Standards

### Imports
- Use absolute imports: `from openadmet.models.architecture.chemprop import ChemPropModel`
- Never use wildcard imports
- Never import modules inside functions or classes

### YAML Serialization
When working with PyTorch Lightning models that need to be serialized to YAML:

1. **Pydantic Models**: Use `@model_serializer` to filter non-YAML-serializable fields
2. **Lightning Modules**: Remove non-serializable hyperparameters in `__init__` using `self.hparams.pop()`
3. **Training**: Set `logger=False` when logging is not needed to avoid hyperparameter serialization

Example:
```python
class MPNN(models.MPNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Remove non-YAML-serializable objects
        for key in ["metrics", "message_passing", "agg", "predictor"]:
            self.hparams.pop(key, None)

class ChemPropModel(LightningModelBase):
    @model_serializer
    def serialize_model(self):
        data = {}
        for key, value in self.__dict__.items():
            if key == "_estimator":
                continue
            try:
                yaml.safe_dump({key: value})
                data[key] = value
            except Exception:
                logger.debug(f"Skipping non-YAML-serializable field: {key}")
        return data
```

### Testing
- Create focused tests for specific functionality (e.g., `test_mpnn_hparams.py`)
- Test both unit functionality and integration workflows
- Use small datasets and few epochs for fast test execution
- Verify serialization works end-to-end

## Common Issues

### YAML Serialization Errors
If you see errors like "dictionary update sequence element #0 has length 1; 2 is required":
- Check if Lightning models have non-serializable objects in `hparams`
- Ensure `logger=False` is set when creating trainers for cross-validation folds
- Verify Pydantic models have `@model_serializer` to filter non-serializable fields

### Import Errors in Docker
If tests can't find modules:
- Ensure `PYTHONPATH=/tmp/openadmet-models` is set
- Check that source code is properly mounted (`-v` flag)
- Verify the Docker image has the required dependencies

## Repository Structure

```
openadmet/models/
├── architecture/     # Model implementations (ChemProp, Ridge, etc.)
├── eval/            # Evaluation methods (cross-validation, etc.)
├── featurizer/      # Molecular featurization
├── split/           # Data splitting strategies
├── trainer/         # Training logic
└── tests/
    ├── integration/ # End-to-end tests
    └── unit/        # Unit tests
```
