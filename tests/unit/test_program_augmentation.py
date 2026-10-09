"""Automatic raw-pixel anchoring for the separate V3 augmentation track."""

import numpy as np
import pytest

from bias_optimizer.domain.evaluation import Evaluation
from bias_optimizer.domain.program import ProgramBiasSpec
from bias_optimizer.dsl.ast import Expr
from bias_optimizer.dsl.compiler import ProgramCompiler
from bias_optimizer.dsl.mutations import add_raw_pixel_anchor
from bias_optimizer.dsl.validator import ProgramConstraints, SearchTrack
from bias_optimizer.ml.evaluator import ProgramEvaluator


def _node(op: str, *args: Expr, **params: object) -> Expr:
    return Expr(op, tuple(args), params)


def _angle_histogram() -> Expr:
    image = _node("image")
    skeleton = _node("skeletonize", image)
    graph = _node("graph", skeleton)
    paths = _node("paths", graph)
    return _node("histogram", _node("angles", paths), bins=6)


def _bias(program: Expr) -> ProgramBiasSpec:
    return ProgramBiasSpec(
        name="augmentation_candidate",
        hypothesis="A structural vector may add information to a raw pixel anchor.",
        mechanism="The classifier receives both raw intensities and a compact geometric summary.",
        program=program,
        prediction="The anchored representation may improve limited-data accuracy.",
        falsification="Reject it if the anchored candidate does not beat its raw-only control.",
    )


def test_raw_pixel_anchor_mutation_reaches_raw_plus_angle_histogram() -> None:
    structural = _angle_histogram()
    anchored = add_raw_pixel_anchor(structural)
    compiler = ProgramCompiler(
        ProgramConstraints(
            track=SearchTrack.AUGMENTATION,
            max_feature_dim=1_024,
            max_depth=7,
            max_nodes=130,
        )
    )
    image = np.zeros((28, 28), dtype=np.float32)
    image[8, 4:20] = 1.0
    image[8:21, 19] = 1.0

    features = compiler.compile(anchored).transform(image)

    assert anchored.op == "concat"
    assert anchored.args[0].op == "flatten_pixels"
    assert features.shape == (790,)
    np.testing.assert_array_equal(features[:784], image.reshape(-1))
    assert np.isfinite(features).all()


def test_raw_pixel_anchor_rejects_programs_that_already_include_pixels() -> None:
    raw = _node("flatten_pixels", _node("image"))
    with pytest.raises(ValueError, match="already contains raw pixels"):
        add_raw_pixel_anchor(raw)


def test_program_evaluator_applies_anchor_automatically() -> None:
    class CaptureEvaluator:
        def evaluate_pipeline(self, pipeline, *, cache_key: str, cache_namespace: str):
            self.pipeline = pipeline
            assert cache_key
            assert cache_namespace == "mnist_v3_augmentation_program"
            return Evaluation(
                accuracy_500=0.8,
                accuracy_5000=0.9,
                feature_dim=pipeline.feature_dim,
                feature_runtime_ms=1,
                training_runtime_ms=1,
                inference_runtime_ms=1,
                confusion_matrix=np.zeros((10, 10), dtype=np.int64),
            )

    capture = CaptureEvaluator()
    evaluator = ProgramEvaluator(
        evaluator=capture,  # type: ignore[arg-type]
        compiler=ProgramCompiler(
            ProgramConstraints(
                track=SearchTrack.AUGMENTATION,
                max_feature_dim=128,
                forbid_raw_pixels=True,
                require_vector_root=True,
            )
        ),
        raw_pixel_anchor=True,
    )

    result = evaluator.evaluate(_bias(_angle_histogram()))

    assert result.feature_dim == 790
    assert capture.pipeline.program.args[0].op == "flatten_pixels"


def test_program_evaluator_passes_stage_one_train_size() -> None:
    class CaptureEvaluator:
        def evaluate_pipeline(
            self,
            pipeline,
            *,
            cache_key: str,
            cache_namespace: str,
            train_sizes: tuple[int, ...],
        ):
            assert cache_key
            assert cache_namespace == "mnist_v3_discovery_program"
            assert train_sizes == (500,)
            return Evaluation(
                accuracy_500=0.8,
                accuracy_5000=None,
                feature_dim=pipeline.feature_dim,
                feature_runtime_ms=1,
                training_runtime_ms=1,
                inference_runtime_ms=1,
                confusion_matrix=np.zeros((10, 10), dtype=np.int64),
            )

    evaluator = ProgramEvaluator(
        evaluator=CaptureEvaluator(),  # type: ignore[arg-type]
        compiler=ProgramCompiler(
            ProgramConstraints(track=SearchTrack.DISCOVERY, max_feature_dim=128)
        ),
        train_sizes=(500,),
    )

    result = evaluator.evaluate(_bias(_angle_histogram()))

    assert result.accuracy_500 == 0.8
    assert result.accuracy_5000 is None
