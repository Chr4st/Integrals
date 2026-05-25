"""Tests for ValueHead and VerificationCache."""

import torch

from neurips.inference.verify_cache import VerificationCache
from neurips.models.action_policy import ValueHead


class TestValueHead:
    def test_output_shape(self):
        vh = ValueHead(embed_dim=256, hidden_dim=128)
        x = torch.randn(4, 256)
        out = vh(x)
        assert out.shape == (4, 1)

    def test_output_in_range(self):
        vh = ValueHead(embed_dim=64, hidden_dim=32)
        x = torch.randn(16, 64)
        out = vh(x)
        assert (out >= -1.0).all()
        assert (out <= 1.0).all()

    def test_gradient_flow(self):
        vh = ValueHead(embed_dim=32, hidden_dim=16)
        x = torch.randn(2, 32, requires_grad=True)
        out = vh(x)
        out.sum().backward()
        assert x.grad is not None


class TestVerificationCache:
    def test_cache_hit(self):
        cache = VerificationCache(max_size=10)
        call_count = 0

        def compute(state: str, action: int):
            nonlocal call_count
            call_count += 1
            return state + "_done", True, True

        r1 = cache.get_or_compute("x", 0, compute)
        r2 = cache.get_or_compute("x", 0, compute)
        assert r1 == r2
        assert call_count == 1
        assert cache.hits == 1
        assert cache.misses == 1

    def test_eviction(self):
        cache = VerificationCache(max_size=3)
        compute = lambda s, a: (f"{s}_{a}", False, False)

        for i in range(5):
            cache.get_or_compute(f"s{i}", 0, compute)

        assert len(cache._cache) == 3

    def test_hit_rate(self):
        cache = VerificationCache(max_size=100)
        compute = lambda s, a: (s, False, False)

        cache.get_or_compute("a", 0, compute)
        cache.get_or_compute("a", 0, compute)
        cache.get_or_compute("b", 0, compute)
        assert abs(cache.hit_rate - 1 / 3) < 1e-5

    def test_clear(self):
        cache = VerificationCache(max_size=100)
        cache.get_or_compute("a", 0, lambda s, a: (s, False, False))
        cache.clear()
        assert cache.hits == 0
        assert cache.misses == 0
        assert len(cache._cache) == 0
