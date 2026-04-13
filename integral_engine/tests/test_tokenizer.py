"""Tests for prefix-notation tokenizer."""
import sympy as sp
from integral_engine.tokenizer import to_prefix, from_prefix, encode, decode
from integral_engine.vocabulary import BOS_INDEX, EOS_INDEX, MAX_INPUT_LEN

x = sp.Symbol("x")


class TestToPrefix:
    def test_positive_integer(self):
        tokens = to_prefix(sp.Integer(2))
        assert tokens == ["INT+", "2"]

    def test_negative_integer(self):
        tokens = to_prefix(sp.Integer(-3))
        assert tokens == ["INT-", "3"]

    def test_multi_digit_integer_base100(self):
        """Phase 14: 42 is a single base-100 token."""
        tokens = to_prefix(sp.Integer(42))
        assert tokens == ["INT+", "42"]
        tokens = to_prefix(sp.Integer(-567))
        assert tokens == ["INT-", "5", "67"]

    def test_large_integer_base100(self):
        """Phase 14: 123456 → ["INT+", "12", "34", "56"]."""
        tokens = to_prefix(sp.Integer(123456))
        assert tokens == ["INT+", "12", "34", "56"]

    def test_simple_symbol(self):
        assert to_prefix(x) == ["x"]

    def test_addition(self):
        tokens = to_prefix(x**2 + 2 * x)
        assert tokens[0] == "Add"

    def test_sin_x(self):
        assert to_prefix(sp.sin(x)) == ["sin", "x"]


class TestFromPrefix:
    def test_round_trip_simple(self):
        exprs = [
            x, x**2, sp.sin(x), sp.cos(x) + x, x**2 + 2 * x + 1,
            sp.exp(x) * sp.sin(x), sp.log(x**2 + 1), sp.Integer(42),
            sp.Rational(3, 4),
        ]
        for expr in exprs:
            tokens = to_prefix(expr)
            recovered = from_prefix(tokens)
            assert sp.simplify(recovered - expr) == 0, (
                f"Round-trip failed for {expr}: got {recovered}"
            )

    def test_round_trip_negative_integer(self):
        tokens = to_prefix(sp.Integer(-5))
        assert tokens == ["INT-", "5"]
        assert from_prefix(tokens) == sp.Integer(-5)

    def test_round_trip_multi_digit_base100(self):
        """Phase 14: Base-100 round trip."""
        tokens = to_prefix(sp.Integer(42))
        assert tokens == ["INT+", "42"]
        assert from_prefix(tokens) == sp.Integer(42)
        tokens = to_prefix(sp.Integer(-567))
        assert tokens == ["INT-", "5", "67"]
        assert from_prefix(tokens) == sp.Integer(-567)

    def test_round_trip_large_integer(self):
        """Phase 14: Large integer round trip."""
        for val in [0, 1, 99, 100, 999, 1234, 99999, 123456]:
            expr = sp.Integer(val)
            tokens = to_prefix(expr)
            recovered = from_prefix(tokens)
            assert recovered == expr, f"Round-trip failed for {val}"

    def test_round_trip_constants_c1_c2(self):
        C1, C2 = sp.Symbol("C1"), sp.Symbol("C2")
        expr = C1 * sp.sin(x) + C2 * sp.cos(x)
        tokens = to_prefix(expr)
        recovered = from_prefix(tokens)
        assert sp.simplify(recovered - expr) == 0


class TestEncodeDecode:
    def test_adds_bos_eos(self):
        indices = encode(sp.sin(x))
        assert indices[0] == BOS_INDEX
        assert indices[-1] == EOS_INDEX

    def test_truncation(self):
        expr = x
        for i in range(200):
            expr = expr + x ** (i + 2)
        indices = encode(expr, max_len=MAX_INPUT_LEN)
        assert len(indices) <= MAX_INPUT_LEN
        assert indices[-1] == EOS_INDEX

    def test_decode_round_trip(self):
        expr = x**2 + sp.sin(x)
        indices = encode(expr)
        recovered = decode(indices)
        assert sp.simplify(recovered - expr) == 0
