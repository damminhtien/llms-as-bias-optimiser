"""Leakage-safe, protocol-driven evaluation of frozen representations."""

from bias_optimizer.evaluation.protocol import EvaluationDataset
from bias_optimizer.evaluation.runner import EvaluationRunner
from bias_optimizer.evaluation.schema import EvaluationKey, EvaluationResult

__all__ = ["EvaluationDataset", "EvaluationKey", "EvaluationResult", "EvaluationRunner"]
