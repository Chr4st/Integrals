"""Tests for Python<->Rust prefix format translation.

Covers: python/neurips/data/prefix.py lines 105-149.
Verifies roundtrip correctness, specific mathematical expressions,
edge cases, and double-prefix behavior.
"""
from __future__ import annotations

import pytest
from hypothesis import given, settings, strategies as st

from neurips.data.prefix import (
    python_prefix_to_rust,
    rust_prefix_to_python,
)
from neurips.data.vocab import ARITY

# ---------------------------------------------------------------------------
# Strategies for property-based testing
# ---------------------------------------------------------------------------

_VARS = ["x", "y", "z", "w", "t"]
_PARAMS = ["a", "b", "c", "d", "n", "m", "k"]
_CONSTS = ["pi", "euler_gamma", "catalan"]
_NUMBERS = [str(i) for i in range(10)]

# Leaf tokens that are valid in Python prefix format
_LEAVES = _VARS + _PARAMS + _CONSTS + _NUMBERS

# Unary functions from the vocab
_UNARY_OPS = [k for k, v in ARITY.items() if v == 1]

# Binary operators from the vocab
_BINARY_OPS = [k for k, v in ARITY.items() if v == 2]


@st.composite
def prefix_expression(draw: st.DrawFn, max_depth: int = 3) -> str:
    """Generate a valid prefix expression from the known vocabulary.

    Builds a tree of depth <= max_depth using known unary/binary ops
    and leaf tokens, then serializes to prefix string.
    """
    if max_depth <= 0:
        return draw(st.sampled_from(_LEAVES))

    choice = draw(st.integers(min_value=0, max_value=2))

    if choice == 0:
        # Leaf
        return draw(st.sampled_from(_LEAVES))
    elif choice == 1:
        # Unary: op child
        op = draw(st.sampled_from(_UNARY_OPS))
        child = draw(prefix_expression(max_depth=max_depth - 1))
        return f"{op} {child}"
    else:
        # Binary: op left right
        op = draw(st.sampled_from(_BINARY_OPS))
        left = draw(prefix_expression(max_depth=max_depth - 1))
        right = draw(prefix_expression(max_depth=max_depth - 1))
        return f"{op} {left} {right}"


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------


class TestRoundtripProperty:
    """Roundtrip: rust_prefix_to_python(python_prefix_to_rust(expr)) == expr."""

    @given(expr=prefix_expression(max_depth=3))
    @settings(max_examples=200, deadline=2000)
    def test_roundtrip_for_arbitrary_valid_prefix(self, expr: str) -> None:
        """For any valid prefix expression built from known tokens,
        translating Python->Rust->Python must recover the original string.
        """
        rust_form = python_prefix_to_rust(expr)
        recovered = rust_prefix_to_python(rust_form)
        assert recovered == expr, (
            f"Roundtrip failed:\n"
            f"  original:  {expr!r}\n"
            f"  rust form: {rust_form!r}\n"
            f"  recovered: {recovered!r}"
        )


# ---------------------------------------------------------------------------
# Specific mathematical expressions that must roundtrip
# ---------------------------------------------------------------------------


_ROUNDTRIP_CASES = [
    pytest.param("add x 3", "add var:x 3", id="simple-addition"),
    pytest.param(
        "mul sin x cos y",
        "mul sin var:x cos var:y",
        id="sin-cos-product",
    ),
    pytest.param(
        "div pow x 3 3",
        "div pow var:x 3 3",
        id="power-rule-x3/3",
    ),
    pytest.param("Ei x", "ei var:x", id="exponential-integral"),
    pytest.param(
        "Hyp2F1 a b c x",
        "hyp2f1 param:a param:b param:c var:x",
        id="hypergeometric-4ary-params",
    ),
    pytest.param(
        "mul pi euler_gamma",
        "mul const:pi const:euler_gamma",
        id="constants-product",
    ),
    pytest.param(
        "BesselJ x y",
        "besselJ var:x var:y",
        id="besselJ-two-vars",
    ),
    pytest.param(
        "FresnelS t",
        "fresnelS var:t",
        id="fresnelS-var-t",
    ),
]


class TestSpecificExpressions:
    """Verify specific math expressions translate correctly both ways."""

    @pytest.mark.parametrize("python_form,expected_rust", _ROUNDTRIP_CASES)
    def test_python_to_rust(
        self, python_form: str, expected_rust: str
    ) -> None:
        """Python prefix -> Rust prefix produces the expected tagged format."""
        assert python_prefix_to_rust(python_form) == expected_rust

    @pytest.mark.parametrize("python_form,expected_rust", _ROUNDTRIP_CASES)
    def test_rust_to_python(
        self, python_form: str, expected_rust: str
    ) -> None:
        """Rust prefix -> Python prefix recovers the original bare format."""
        assert rust_prefix_to_python(expected_rust) == python_form

    @pytest.mark.parametrize("python_form,expected_rust", _ROUNDTRIP_CASES)
    def test_full_roundtrip(
        self, python_form: str, expected_rust: str
    ) -> None:
        """Full roundtrip: Python -> Rust -> Python == original."""
        assert rust_prefix_to_python(python_prefix_to_rust(python_form)) == python_form


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases that must not crash and must produce expected output."""

    def test_empty_string_roundtrips(self) -> None:
        """Empty string -> empty string in both directions."""
        assert python_prefix_to_rust("") == ""
        assert rust_prefix_to_python("") == ""

    def test_single_variable(self) -> None:
        """A bare variable roundtrips: x -> var:x -> x."""
        assert python_prefix_to_rust("x") == "var:x"
        assert rust_prefix_to_python("var:x") == "x"

    def test_single_param(self) -> None:
        """A bare parameter roundtrips: a -> param:a -> a."""
        assert python_prefix_to_rust("a") == "param:a"
        assert rust_prefix_to_python("param:a") == "a"

    def test_single_constant(self) -> None:
        """A bare constant roundtrips: pi -> const:pi -> pi."""
        assert python_prefix_to_rust("pi") == "const:pi"
        assert rust_prefix_to_python("const:pi") == "pi"

    def test_single_number_passthrough(self) -> None:
        """Numbers are not prefixed — they pass through unchanged."""
        assert python_prefix_to_rust("42") == "42"
        assert rust_prefix_to_python("42") == "42"

    def test_unknown_token_passthrough(self) -> None:
        """An unknown token passes through unchanged in both directions."""
        assert python_prefix_to_rust("foobar") == "foobar"
        assert rust_prefix_to_python("foobar") == "foobar"

    def test_whitespace_handling(self) -> None:
        """Leading/trailing whitespace is stripped; internal spaces normalized."""
        assert python_prefix_to_rust("  add x 3  ") == "add var:x 3"
        assert rust_prefix_to_python("  add var:x 3  ") == "add x 3"

    def test_all_token_types_in_one_expression(self) -> None:
        """Expression with vars, params, constants, functions, operators, numbers."""
        py = "add mul a x div pi 2"
        rust = python_prefix_to_rust(py)
        # Verify each token type is correctly tagged
        assert "param:a" in rust  # param
        assert "var:x" in rust  # var
        assert "const:pi" in rust  # constant
        assert "add" in rust  # operator (no prefix)
        assert "mul" in rust  # operator (no prefix)
        assert "div" in rust  # operator (no prefix)
        assert " 2" in rust  # number (no prefix)
        # Roundtrip
        assert rust_prefix_to_python(rust) == py

    def test_rat_prefix_passthrough_in_rust_to_python(self) -> None:
        """rat: prefix in Rust format passes through to Python unchanged.
        The code explicitly keeps rat: tokens as-is.
        """
        assert rust_prefix_to_python("rat:1/2") == "rat:1/2"


# ---------------------------------------------------------------------------
# Negative tests: double-prefix behavior
# ---------------------------------------------------------------------------


class TestDoublePrefix:
    """Document and verify what happens when already-tagged tokens
    are fed through the wrong direction.
    """

    def test_python_to_rust_on_already_tagged_var(self) -> None:
        """Feeding 'var:x' (Rust format) into python_prefix_to_rust.

        'var:x' is not in _VARS, _PARAMS, _CONSTS, or _FUNC_PY_TO_RUST,
        so it passes through unchanged. It does NOT double-prefix.
        """
        result = python_prefix_to_rust("var:x")
        # 'var:x' as a single token isn't a known var/param/const/func,
        # so it passes through as-is
        assert result == "var:x"

    def test_python_to_rust_on_already_tagged_param(self) -> None:
        """Feeding 'param:a' into python_prefix_to_rust passes through."""
        result = python_prefix_to_rust("param:a")
        assert result == "param:a"

    def test_rust_to_python_on_bare_python_tokens(self) -> None:
        """Feeding bare Python tokens ('x') into rust_to_python.

        Bare 'x' has no 'var:' prefix, so it won't be stripped.
        But 'x' might match _FUNC_RUST_TO_PY — it doesn't, so it passes through.
        """
        result = rust_prefix_to_python("x")
        # 'x' has no prefix to strip and isn't in _FUNC_RUST_TO_PY
        assert result == "x"

    def test_double_roundtrip_is_idempotent(self) -> None:
        """Applying the roundtrip twice must give the same result as once."""
        expr = "add mul sin x cos y div pow z 2 3"
        once = rust_prefix_to_python(python_prefix_to_rust(expr))
        twice = rust_prefix_to_python(python_prefix_to_rust(once))
        assert once == twice == expr


# ---------------------------------------------------------------------------
# All special function name mappings
# ---------------------------------------------------------------------------


class TestFunctionNameMappings:
    """Every entry in _FUNC_PY_TO_RUST must roundtrip correctly."""

    _FUNC_PAIRS = [
        ("Ei", "ei"), ("Si", "si"), ("Ci", "ci"), ("Li", "li"),
        ("Gamma", "gamma"), ("BesselJ", "besselJ"), ("BesselY", "besselY"),
        ("FresnelS", "fresnelS"), ("FresnelC", "fresnelC"),
        ("EllipticK", "ellipticK"), ("EllipticE", "ellipticE"),
        ("Hyp2F1", "hyp2f1"),
    ]

    @pytest.mark.parametrize(
        "py_name,rust_name", _FUNC_PAIRS, ids=[p[0] for p in _FUNC_PAIRS]
    )
    def test_function_name_python_to_rust(
        self, py_name: str, rust_name: str
    ) -> None:
        """Each Python function name maps to the expected Rust name."""
        assert python_prefix_to_rust(py_name) == rust_name

    @pytest.mark.parametrize(
        "py_name,rust_name", _FUNC_PAIRS, ids=[p[0] for p in _FUNC_PAIRS]
    )
    def test_function_name_rust_to_python(
        self, py_name: str, rust_name: str
    ) -> None:
        """Each Rust function name maps back to the expected Python name."""
        assert rust_prefix_to_python(rust_name) == py_name
