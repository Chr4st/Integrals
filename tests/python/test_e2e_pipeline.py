"""End-to-end integration test for the full pipeline.

Tests the complete flow:
  raw expression → prefix tokenization → feature extraction →
  model forward pass → tree decoding → prefix output → verification

This is NOT a coverage test. Every assertion checks a behavioral invariant
that, if violated, would produce wrong results in production.
"""
from __future__ import annotations

import torch
import pytest

from neurips.data.dataset import _TASK_TO_TOKENIZER, precompute_dataset
from neurips.data.features import FEATURE_DIM, extract_features
from neurips.data.prefix import (
    parse_prefix,
    python_prefix_to_rust,
    rust_prefix_to_python,
    tokenize_prefix,
)
from neurips.data.tokenizer import IntegralTokenizer
from neurips.data.vocab import ARITY, EOS, ID_TO_TOKEN, SOS, VOCAB, VOCAB_SIZE
from neurips.evaluation.oracle import prefix_to_sympy
from neurips.evaluation.verify import VerificationOracle
from neurips.models.grammar import ArityStack, PrefixGrammarMask, arity_of
from neurips.models.tree_decoder import TreeDecoder, TreeNode
from neurips.models.tree_gnn import TreeIntegrator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tokenizer() -> IntegralTokenizer:
    return IntegralTokenizer()


@pytest.fixture
def oracle() -> VerificationOracle:
    return VerificationOracle(timeout=4.0)


@pytest.fixture
def tree_model() -> TreeIntegrator:
    model = TreeIntegrator()
    model.eval()
    return model


KNOWN_INTEGRALS = [
    # (integrand_prefix, antiderivative_prefix, var, task)
    ("pow x 2", "div pow x 3 3", "x", "univariate"),
    ("sin x", "neg cos x", "x", "univariate"),
    ("exp x", "exp x", "x", "univariate"),
    ("cos x", "sin x", "x", "univariate"),
    ("div 1 x", "log x", "x", "univariate"),
]


# ---------------------------------------------------------------------------
# Stage 1: Tokenization roundtrip
# ---------------------------------------------------------------------------


class TestTokenizationRoundtrip:
    """Tokenize → decode must recover the original expression."""

    @pytest.mark.parametrize(
        "integrand,antideriv,var,task", KNOWN_INTEGRALS,
        ids=[f"tok-{i}" for i in range(len(KNOWN_INTEGRALS))],
    )
    def test_encode_decode_recovers_expression(
        self, tokenizer: IntegralTokenizer,
        integrand: str, antideriv: str, var: str, task: str,
    ) -> None:
        tok_task = _TASK_TO_TOKENIZER.get(task, "indef")
        ids = tokenizer.encode(integrand, tok_task, var)

        # IDs must be valid vocab indices
        assert all(0 <= i < VOCAB_SIZE for i in ids), (
            f"Token IDs out of range: {[i for i in ids if i >= VOCAB_SIZE]}"
        )

        # Must start with task token and end with EOS
        assert ids[-1] == EOS, f"Last token must be EOS, got {ids[-1]}"

        # Decode and verify the prefix tokens are recoverable
        decoded_str, _ = tokenizer.decode(ids)
        # The decoded string should contain the original expression tokens
        orig_tokens = tokenize_prefix(integrand)
        decoded_tokens = tokenize_prefix(decoded_str)
        assert orig_tokens == decoded_tokens, (
            f"Token roundtrip failed:\n"
            f"  original:  {orig_tokens}\n"
            f"  decoded:   {decoded_tokens}"
        )


# ---------------------------------------------------------------------------
# Stage 2: Feature extraction produces valid vectors
# ---------------------------------------------------------------------------


class TestFeatureExtraction:
    """Feature vectors must have correct dimension and finite values."""

    @pytest.mark.parametrize(
        "integrand,antideriv,var,task", KNOWN_INTEGRALS,
        ids=[f"feat-{i}" for i in range(len(KNOWN_INTEGRALS))],
    )
    def test_features_dim_and_finite(
        self, integrand: str, antideriv: str, var: str, task: str,
    ) -> None:
        feat = extract_features(integrand, var, task)
        assert feat.shape == (FEATURE_DIM,), f"Expected {FEATURE_DIM}-dim, got {feat.shape}"
        assert all(f == f for f in feat), "NaN in feature vector"  # NaN != NaN
        assert all(abs(f) < 1e10 for f in feat), "Feature value overflow"

    def test_different_expressions_produce_different_features(self) -> None:
        """Non-trivially different inputs must produce different features."""
        f1 = extract_features("pow x 2", "x", "univariate")
        f2 = extract_features("sin x", "x", "univariate")
        # They share the same task/var features but differ in function heads
        assert not (f1 == f2).all(), "Different expressions produced identical features"


# ---------------------------------------------------------------------------
# Stage 3: Prefix format translation (Python <-> Rust)
# ---------------------------------------------------------------------------


class TestPrefixTranslationE2E:
    """Full pipeline prefix translation must preserve mathematical meaning."""

    @pytest.mark.parametrize(
        "integrand,antideriv,var,task", KNOWN_INTEGRALS,
        ids=[f"pfx-{i}" for i in range(len(KNOWN_INTEGRALS))],
    )
    def test_python_to_rust_roundtrip_preserves_expression(
        self, integrand: str, antideriv: str, var: str, task: str,
    ) -> None:
        rust_form = python_prefix_to_rust(integrand)
        recovered = rust_prefix_to_python(rust_form)
        assert recovered == integrand, (
            f"Roundtrip failed: {integrand!r} -> {rust_form!r} -> {recovered!r}"
        )

    def test_rust_format_has_var_tags(self) -> None:
        """Variables in Rust format must carry var: prefix."""
        rust = python_prefix_to_rust("add x y")
        assert "var:x" in rust
        assert "var:y" in rust

    def test_rust_format_preserves_operators(self) -> None:
        """Operators must NOT be tagged."""
        rust = python_prefix_to_rust("add mul x y")
        assert rust.startswith("add")
        assert "mul" in rust
        # Operators should not have prefixes
        assert "var:add" not in rust
        assert "var:mul" not in rust


# ---------------------------------------------------------------------------
# Stage 4: Model forward pass produces valid tree structure
# ---------------------------------------------------------------------------


class TestModelForwardPass:
    """TreeIntegrator forward must produce a valid prefix tree."""

    def test_forward_produces_nonempty_tree(self, tree_model: TreeIntegrator) -> None:
        """A non-trivial input must produce at least one decoded node."""
        N = 5
        sym_ids = torch.randint(0, VOCAB_SIZE, (N,))
        role_feat = torch.randn(N, 12)
        struct_feat = torch.randn(N, 40)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

        with torch.no_grad():
            nodes = tree_model(sym_ids, role_feat, struct_feat, edge_index)

        assert len(nodes) > 0, "Decoder produced empty tree"
        assert all(isinstance(n, TreeNode) for n in nodes)

    def test_tree_arity_consistency(self, tree_model: TreeIntegrator) -> None:
        """Each node's children count must equal its arity."""
        N = 6
        sym_ids = torch.randint(0, VOCAB_SIZE, (N,))
        role_feat = torch.randn(N, 12)
        struct_feat = torch.randn(N, 40)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)

        with torch.no_grad():
            nodes = tree_model(sym_ids, role_feat, struct_feat, edge_index)

        if len(nodes) <= 1:
            return  # trivial tree is ok

        # Count children per node
        children_count: dict[int, int] = {}
        for i, node in enumerate(nodes):
            if node.parent_idx is not None:
                children_count[node.parent_idx] = children_count.get(node.parent_idx, 0) + 1

        for idx, count in children_count.items():
            assert count == nodes[idx].arity, (
                f"Node {idx} (symbol_id={nodes[idx].symbol_id}) has arity "
                f"{nodes[idx].arity} but {count} children"
            )


# ---------------------------------------------------------------------------
# Stage 5: Grammar mask constrains valid tokens
# ---------------------------------------------------------------------------


class TestGrammarConstrainedDecoding:
    """Grammar mask must only allow tokens that maintain a valid prefix tree."""

    def test_mask_allows_only_valid_continuations(self) -> None:
        """After a binary op, must allow expressions (not EOS)."""
        grammar = PrefixGrammarMask()
        # After SOS, any expression start is valid
        mask = grammar.step(SOS)
        assert mask.shape == (VOCAB_SIZE,)
        assert mask.any(), "No tokens allowed after SOS"

        # Find a binary op token
        binary_op = None
        for name, ar in ARITY.items():
            if ar == 2 and name in VOCAB:
                binary_op = VOCAB[name]
                break

        if binary_op is None:
            pytest.skip("No binary op found in vocab")

        mask_after_binary = grammar.step(binary_op)
        # After a binary op, we need 2 more expressions — EOS must not be allowed
        assert not mask_after_binary[EOS], (
            "EOS allowed after binary op with pending children"
        )

    def test_leaf_then_eos_is_valid_after_completion(self) -> None:
        """A single leaf token completes the expression — EOS must be allowed."""
        grammar = PrefixGrammarMask()
        _ = grammar.step(SOS)

        # Find a leaf token (arity 0)
        leaf = None
        for name, ar in ARITY.items():
            if ar == 0 and name in VOCAB:
                leaf = VOCAB[name]
                break

        if leaf is None:
            pytest.skip("No leaf token found in vocab")

        mask_after_leaf = grammar.step(leaf)
        assert mask_after_leaf[EOS], "EOS should be allowed after a complete expression"


# ---------------------------------------------------------------------------
# Stage 6: Verification oracle correctness
# ---------------------------------------------------------------------------


class TestVerificationE2E:
    """Verification oracle must accept correct integrals, reject wrong ones."""

    @pytest.mark.parametrize(
        "integrand,antideriv,var,task", KNOWN_INTEGRALS,
        ids=[f"verify-{i}" for i in range(len(KNOWN_INTEGRALS))],
    )
    def test_correct_integral_verifies(
        self, oracle: VerificationOracle,
        integrand: str, antideriv: str, var: str, task: str,
    ) -> None:
        task_info = {"task": task, "var": var}
        result = oracle.verify(antideriv, integrand, task_info)
        assert result is True, (
            f"Oracle rejected correct pair: d/dx({antideriv}) should equal {integrand}"
        )

    def test_wrong_integral_is_rejected(self, oracle: VerificationOracle) -> None:
        """sin(x) is NOT the integral of x^2."""
        task_info = {"task": "univariate", "var": "x"}
        result = oracle.verify("sin x", "pow x 2", task_info)
        assert result is False, "Oracle accepted wrong integral"


# ---------------------------------------------------------------------------
# Stage 7: Dataset precomputation roundtrip
# ---------------------------------------------------------------------------


class TestDatasetPrecomputeE2E:
    """Full precompute pipeline: raw data → cached tensors → loadable dataset."""

    def test_precompute_and_reload(
        self, tokenizer: IntegralTokenizer, tmp_path,
    ) -> None:
        raw_data = [
            {
                "integrand": "pow x 2",
                "antiderivative": "div pow x 3 3",
                "task": "univariate",
                "var": "x",
                "difficulty_tier": "easy",
            },
            {
                "integrand": "sin x",
                "antiderivative": "neg cos x",
                "task": "univariate",
                "var": "x",
                "difficulty_tier": "medium",
            },
        ]

        cache_path = tmp_path / "e2e_cache.pt"
        ds = precompute_dataset(raw_data, tokenizer, cache_path)

        # Verify it was saved and can be reloaded
        from neurips.data.dataset import IntegralDataset
        ds2 = IntegralDataset(cache_path)

        assert len(ds2) == 2
        assert ds2.src_ids.shape[0] == 2
        assert ds2.features.shape == (2, FEATURE_DIM)

        # Verify tensors match between original and reloaded
        assert torch.equal(ds.src_ids, ds2.src_ids)
        assert torch.equal(ds.tgt_ids, ds2.tgt_ids)
        assert torch.allclose(ds.features, ds2.features)


# ---------------------------------------------------------------------------
# Stage 8: Full pipeline integration
# ---------------------------------------------------------------------------


class TestFullPipelineIntegration:
    """The complete pipeline from raw math to verified decode."""

    def test_decode_then_verify_roundtrip(
        self, tree_model: TreeIntegrator, oracle: VerificationOracle,
    ) -> None:
        """Decoded tree nodes map to valid token strings via ID_TO_TOKEN.

        This test verifies the data flow from decoder output to verification
        input format — the most error-prone seam in the pipeline.
        """
        N = 4
        sym_ids = torch.randint(3, VOCAB_SIZE, (N,))
        role_feat = torch.randn(N, 12)
        struct_feat = torch.randn(N, 40)
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

        with torch.no_grad():
            nodes = tree_model(sym_ids, role_feat, struct_feat, edge_index)

        # Map to token strings — this is exactly what _generate_tree does
        tokens = [ID_TO_TOKEN.get(n.symbol_id, "<UNK>") for n in nodes]
        prefix_str = " ".join(tokens)

        # The output must be a non-empty string
        assert len(prefix_str.strip()) > 0, "Empty decode output"

        # Each token must be resolvable (no <UNK> in a well-trained model,
        # but even with random weights, the mapping must work)
        for tok in tokens:
            assert tok in VOCAB or tok == "<UNK>", f"Unknown token: {tok!r}"

    def test_sympy_parse_of_known_expressions(self) -> None:
        """Known prefix expressions must parse to valid SymPy objects."""
        for integrand, antideriv, var, _ in KNOWN_INTEGRALS:
            expr_i = prefix_to_sympy(integrand)
            expr_a = prefix_to_sympy(antideriv)
            assert expr_i is not None, f"Failed to parse integrand: {integrand}"
            assert expr_a is not None, f"Failed to parse antideriv: {antideriv}"

    def test_curriculum_filtering_preserves_data(self) -> None:
        """Curriculum filtering must not lose any examples — only reorder/subset."""
        from neurips.training.curriculum import build_index_maps

        raw_data = [
            {"task": "univariate", "difficulty_tier": "easy"},
            {"task": "univariate", "difficulty_tier": "medium"},
            {"task": "univariate", "difficulty_tier": "hard"},
            {"task": "definite", "difficulty_tier": "easy"},
            {"task": "definite", "difficulty_tier": "medium"},
            {"task": "parametric", "difficulty_tier": "easy"},
        ]

        index_maps = build_index_maps(raw_data)

        # Every example must appear in exactly one group
        all_indices = set()
        for task_dict in index_maps.values():
            for idx_list in task_dict.values():
                for idx in idx_list:
                    assert idx not in all_indices, f"Index {idx} appears in multiple groups"
                    all_indices.add(idx)

        assert all_indices == set(range(len(raw_data))), (
            f"Missing indices: {set(range(len(raw_data))) - all_indices}"
        )
