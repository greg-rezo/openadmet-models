#!/usr/bin/env python
"""Fast test for ChemProp YAML serialization."""

import yaml
from openadmet.models.anvil.specification import AnvilSpecification


def test_chemprop_workflow_yaml_serialization():
    """Test that a workflow with ChemPropModel can be serialized to YAML."""
    # Load the test configuration
    spec = AnvilSpecification.from_recipe(
        "/tmp/openadmet-models/openadmet/models/tests/integration/test_data/chemprop_cv_cpu.yaml"
    )

    print(f"Loaded specification: {spec.metadata.name}")
    print(f"Model type: {spec.procedure.model.type}")

    # Get the actual model instance and build it
    model = spec.procedure.model.to_class()
    model.build(scaler=None)
    print("Model built successfully")

    # Update the spec with the built model (this is what workflow does)
    spec.procedure.model = model

    # Test that model_dump() works on the entire specification
    try:
        spec_data = spec.model_dump()
        print(f"spec.model_dump() succeeded. Top-level keys: {list(spec_data.keys())}")
    except Exception as e:
        print(f"spec.model_dump() failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Test that YAML serialization works (this is what to_recipe() does)
    try:
        yaml_str = yaml.safe_dump(spec_data)
        print("YAML serialization succeeded!")
        print(f"YAML length: {len(yaml_str)} chars")
    except Exception as e:
        print(f"YAML serialization failed: {e}")
        import traceback
        traceback.print_exc()

        # Debug: try to find which field is failing
        print("\nDebugging: Testing each top-level field...")
        for k, v in spec_data.items():
            try:
                yaml.safe_dump({k: v})
                print(f"  ✓ {k}")
            except Exception as field_error:
                print(f"  ✗ {k}: {field_error}")
                # If it's a dict, test its nested fields
                if isinstance(v, dict):
                    print(f"    Testing nested fields in '{k}'...")
                    for k2, v2 in v.items():
                        try:
                            yaml.safe_dump({k2: v2})
                            print(f"      ✓ {k}.{k2}")
                        except Exception as nested_error:
                            print(f"      ✗ {k}.{k2}: {type(v2)} - {nested_error}")
        raise


if __name__ == "__main__":
    test_chemprop_workflow_yaml_serialization()
    print("\n✅ All tests passed!")
