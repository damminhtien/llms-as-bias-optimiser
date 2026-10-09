"""Tests for immutable bias specifications and JSON contracts."""

from collections.abc import Mapping

import pytest

from bias_optimizer.domain.bias import BiasSpec, OperatorSpec


def _sample_bias() -> BiasSpec:
    return BiasSpec(
        name="topology_with_spatial_context",
        hypothesis="Vertical ink location helps separate similar digit shapes.",
        operators=(
            OperatorSpec("topology"),
            OperatorSpec(
                "spatial",
                {"bands": ["top", "middle", "bottom"], "weights": {"ink": 1.0}},
            ),
        ),
        prediction="Should improve validation accuracy for 4 versus 9.",
        falsification="Reject if accuracy at 500 examples does not improve.",
    )


def test_bias_spec_round_trips_through_canonical_json() -> None:
    bias = _sample_bias()

    encoded = bias.to_json()
    restored = BiasSpec.from_json(encoded)

    assert restored == bias
    assert BiasSpec.from_json(restored.to_json()) == bias
    assert '"operators"' in encoded


def test_operator_spec_round_trips_through_json() -> None:
    operator = OperatorSpec("topology", {"threshold": 0.5})

    assert OperatorSpec.from_json(operator.to_json()) == operator


def test_specs_are_deeply_immutable_and_copy_input_values() -> None:
    source = {"nested": {"values": [1, 2]}}
    operator = OperatorSpec("topology", source)
    source["nested"]["values"].append(3)

    assert isinstance(operator.params, Mapping)
    assert operator.params["nested"]["values"] == (1, 2)
    with pytest.raises(TypeError):
        operator.params["new"] = "value"  # type: ignore[index]
    with pytest.raises(TypeError):
        operator.params["nested"]["new"] = "value"  # type: ignore[index]


def test_parameter_json_output_is_canonical_across_key_order() -> None:
    first = OperatorSpec("spatial", {"z": 1, "a": {"y": 2, "b": 3}})
    second = OperatorSpec("spatial", {"a": {"b": 3, "y": 2}, "z": 1})

    assert first.to_json() == second.to_json()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), object()])
def test_operator_params_reject_non_json_values(value: object) -> None:
    with pytest.raises(TypeError, match="finite JSON"):
        OperatorSpec("topology", {"value": value})


def test_operator_names_follow_safe_lowercase_identifier_format() -> None:
    with pytest.raises(ValueError, match="lowercase identifier"):
        OperatorSpec("not-an-operator")
    with pytest.raises(TypeError, match="operator name must be a string"):
        OperatorSpec(123)  # type: ignore[arg-type]


def test_bias_spec_requires_all_text_fields_and_at_least_one_operator() -> None:
    with pytest.raises(ValueError, match="hypothesis cannot be empty"):
        BiasSpec("valid", " ", (OperatorSpec("topology"),), "prediction", "test")
    with pytest.raises(ValueError, match="at least one operator"):
        BiasSpec("valid", "hypothesis", (), "prediction", "test")


def test_bias_json_schema_rejects_missing_and_extra_fields() -> None:
    payload = _sample_bias().to_dict()
    del payload["prediction"]
    with pytest.raises(ValueError, match=r"missing=\['prediction'\]"):
        BiasSpec.from_dict(payload)

    payload = _sample_bias().to_dict()
    payload["python"] = "should not be executed"
    with pytest.raises(ValueError, match=r"extra=\['python'\]"):
        BiasSpec.from_dict(payload)
