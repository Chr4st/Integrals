"""Tests for the tokenizer (Phase 7a)."""
import pytest
from neurips.data.tokenizer import IntegralTokenizer


@pytest.fixture
def tokenizer():
    return IntegralTokenizer()


class TestVocab:
    def test_vocab_size(self, tokenizer):
        assert tokenizer.vocab_size == 256

    def test_special_tokens_present(self, tokenizer):
        from neurips.data.vocab import VOCAB
        for tok in ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]:
            assert tok in VOCAB


class TestNumberEncoding:
    def test_encode_positive(self, tokenizer):
        ids = tokenizer.encode_number(347)
        assert ids[0] == 100  # "+" sign
        decoded = tokenizer.decode_number(ids)
        assert decoded == 347

    def test_encode_negative(self, tokenizer):
        ids = tokenizer.encode_number(-42)
        assert ids[0] == 101  # "-" sign
        decoded = tokenizer.decode_number(ids)
        assert decoded == -42

    def test_encode_zero(self, tokenizer):
        ids = tokenizer.encode_number(0)
        decoded = tokenizer.decode_number(ids)
        assert decoded == 0

    def test_roundtrip_large(self, tokenizer):
        for n in [0, 1, -1, 99, 100, -9999, 5432]:
            assert tokenizer.decode_number(tokenizer.encode_number(n)) == n


class TestEncodeDecode:
    def test_simple_roundtrip(self, tokenizer):
        prefix = "add x x"
        encoded = tokenizer.encode(prefix, task="INDEF", var="x")
        assert isinstance(encoded, list)
        assert all(isinstance(t, int) for t in encoded)
        assert encoded[-1] == 2  # EOS

    def test_task_token_indef(self, tokenizer):
        encoded = tokenizer.encode("x", task="INDEF", var="x")
        assert encoded[0] == 5  # [INDEF]

    def test_task_token_def(self, tokenizer):
        encoded = tokenizer.encode("x", task="DEF", var="x")
        assert encoded[0] == 6  # [DEF]

    def test_var_token(self, tokenizer):
        encoded = tokenizer.encode("x", task="INDEF", var="x")
        assert 7 in encoded  # [VAR]
        var_idx = encoded.index(7)
        assert encoded[var_idx + 1] == 70  # x

    def test_multivariate(self, tokenizer):
        encoded = tokenizer.encode("mul x y", task="INDEF", var="x")
        assert 70 in encoded  # x
        assert 71 in encoded  # y
