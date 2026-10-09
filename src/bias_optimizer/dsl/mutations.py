"""Deterministic mutations applied by the bounded V3 search tracks."""

from __future__ import annotations

from bias_optimizer.dsl.ast import Expr


def add_raw_pixel_anchor(program: Expr) -> Expr:
    """Return ``concat(flatten_pixels(image), program)`` for augmentation search."""
    if not isinstance(program, Expr):
        raise TypeError("raw-pixel anchor mutation requires an Expr")

    def contains_pixels(node: Expr) -> bool:
        return node.op == "flatten_pixels" or any(
            contains_pixels(child) for child in node.args
        )

    if contains_pixels(program):
        raise ValueError("structural program already contains raw pixels")
    image = Expr("image")
    pixels = Expr("flatten_pixels", args=(image,))
    return Expr("concat", args=(pixels, program))
