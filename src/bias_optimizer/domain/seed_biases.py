"""The five fixed, human-designed candidates used to initialize search."""

from __future__ import annotations

from bias_optimizer.domain.bias import BiasSpec, OperatorSpec


def initial_human_biases() -> tuple[BiasSpec, ...]:
    """Return the five preregistered representations from the experiment plan."""
    return (
        BiasSpec(
            name="raw_pixels_control",
            hypothesis="The fixed learner can use the complete normalized pixel grid.",
            operators=(OperatorSpec("raw_pixels"),),
            prediction="Raw pixels provide a strong direct control.",
            falsification="Reject if the fixed learner cannot learn from pixel values.",
        ),
        BiasSpec(
            name="topology_only",
            hypothesis="Components, holes, endpoints, and junctions carry digit identity.",
            operators=(OperatorSpec("topology"),),
            prediction="Topology should distinguish digits with different loop counts.",
            falsification="Reject if topology performs near chance on validation data.",
        ),
        BiasSpec(
            name="topology_plus_spatial",
            hypothesis="Vertical ink location disambiguates similar topologies.",
            operators=(OperatorSpec("topology"), OperatorSpec("spatial")),
            prediction="Spatial occupancy should improve over topology alone.",
            falsification="Reject if validation accuracy does not improve over topology.",
        ),
        BiasSpec(
            name="topology_plus_curvature",
            hypothesis="Local turning patterns add shape detail to topology.",
            operators=(OperatorSpec("topology"), OperatorSpec("curvature")),
            prediction="Curvature should help separate digits with similar loop counts.",
            falsification="Reject if validation accuracy does not improve over topology.",
        ),
        BiasSpec(
            name="topology_direction_curvature",
            hypothesis=(
                "Stroke orientation and local turning add complementary shape cues."
            ),
            operators=(
                OperatorSpec("topology"),
                OperatorSpec("stroke_direction"),
                OperatorSpec("curvature"),
            ),
            prediction="Direction and curvature should improve low-data accuracy.",
            falsification="Reject if the combined representation fails to improve.",
        ),
    )
