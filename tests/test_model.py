import numpy as np
import torch
import torch.nn.functional as F

from seqrec_rl.data import PAD
from seqrec_rl.model import LinearCrossEntropy, WindowSampler


def test_linear_ce_matches_reference():
    torch.manual_seed(0)
    h = torch.randn(300, 16, device="cuda", requires_grad=True)
    w = torch.randn(1000, 16, device="cuda", requires_grad=True)
    y = torch.randint(0, 1000, (300,), device="cuda")

    ref = F.cross_entropy(h @ w.t(), y, reduction="sum")
    gh, gw = torch.autograd.grad(ref, (h, w))
    out = LinearCrossEntropy.apply(h, w, y, 128)
    oh, ow = torch.autograd.grad(out, (h, w))

    assert torch.allclose(out, ref, rtol=1e-4)
    assert torch.allclose(oh, gh, atol=1e-4)
    assert torch.allclose(ow, gw, atol=1e-4)


def test_linear_ce_under_autocast():
    torch.manual_seed(0)
    h = torch.randn(300, 16, device="cuda", requires_grad=True)
    w = torch.randn(1000, 16, device="cuda", requires_grad=True)
    y = torch.randint(0, 1000, (300,), device="cuda")

    ref = F.cross_entropy(h @ w.t(), y, reduction="sum")
    gh, gw = torch.autograd.grad(ref, (h, w))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = LinearCrossEntropy.apply(h, w, y, 128)
    oh, ow = torch.autograd.grad(out, (h, w))

    def rel(a, b):
        return ((a - b).norm() / b.norm()).item()

    assert ow.dtype == torch.float32
    assert rel(out, ref) < 1e-2
    assert rel(oh, gh) < 2e-2
    assert rel(ow, gw) < 2e-2


def test_window_sampler_covers_each_target_once():
    offsets = np.array([0, 1, 7, 30])
    tokens = np.arange(100, 130)
    sampler = WindowSampler(offsets, tokens, seq_len=4, seed=1)
    labels = []
    for batch in sampler.batches(3):
        for row in batch:
            lab = row[1:]
            labels.extend(lab[lab != PAD].tolist())
    # every token except the first of each user is a target exactly once
    expected = [t for u in range(3) for t in tokens[offsets[u] + 1:offsets[u + 1]]]
    assert sorted(labels) == sorted(expected)
