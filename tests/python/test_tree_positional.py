"""Tests for tree positional encoding modules."""

import torch

from neurips.models.tree_positional import (
    DepthIndexPE,
    LaplacianPE,
    RWSE,
    build_tree_pe,
)


class TestDepthIndexPE:
    def test_output_shape(self):
        pe = DepthIndexPE(d_out=64, max_depth=20)
        depths = torch.tensor([0, 1, 2, 1])
        child_idx = torch.tensor([0, 1, 2, 1])
        out = pe(depths, child_idx)
        assert out.shape == (4, 64)

    def test_different_depths_differ(self):
        pe = DepthIndexPE(d_out=32)
        d1 = torch.tensor([0, 1, 2, 3])
        d2 = torch.tensor([3, 2, 1, 0])
        ci = torch.zeros(4, dtype=torch.long)
        out1 = pe(d1, ci)
        out2 = pe(d2, ci)
        assert not torch.allclose(out1, out2)


class TestRWSE:
    def test_output_shape(self):
        rwse = RWSE(k=4, d_out=64)
        n_nodes = 8
        edges = []
        for i in range(1, n_nodes):
            edges.append([i - 1, i])
            edges.append([i, i - 1])
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        out = rwse(edge_index, n_nodes)
        assert out.shape == (n_nodes, 64)

    def test_zero_edges_gives_finite(self):
        rwse = RWSE(k=4, d_out=32)
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        out = rwse(edge_index, 4)
        assert torch.isfinite(out).all()


class TestLaplacianPE:
    def test_output_shape(self):
        lpe = LaplacianPE(k=4, d_out=64)
        n_nodes = 8
        edges = []
        for i in range(1, n_nodes):
            edges.append([i - 1, i])
            edges.append([i, i - 1])
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        out = lpe(edge_index, n_nodes)
        assert out.shape == (n_nodes, 64)


class TestBuildTreePE:
    def test_none_returns_none(self):
        assert build_tree_pe("none") is None

    def test_depth_index_builds(self):
        pe = build_tree_pe("depth_index", d_out=64)
        assert pe is not None
        assert isinstance(pe, DepthIndexPE)

    def test_rwse_builds(self):
        pe = build_tree_pe("rwse", d_out=64, k=4)
        assert pe is not None
        assert isinstance(pe, RWSE)

    def test_laplacian_builds(self):
        pe = build_tree_pe("laplacian", d_out=64, k=4)
        assert pe is not None
        assert isinstance(pe, LaplacianPE)
