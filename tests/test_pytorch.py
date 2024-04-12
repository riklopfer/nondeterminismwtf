from typing import Optional

import numpy as np
import pytest
import torch


class BoringModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


def encoder_factory(device: Optional[str]):
    model = BoringModel()
    if device:
        model = model.to(device)
    model.eval()

    def forward(x):
        if device:
            x = x.to(device)

        with torch.no_grad():
            output = model(x)

        return output.cpu().detach().numpy()

    return forward


@pytest.mark.parametrize(
    "encoder_fn",
    [
        encoder_factory(device=None),
        encoder_factory(device="mps"),
    ],
)
def test_idempotency(encoder_fn):
    test_tensor = torch.rand([1, 10])

    res1 = encoder_fn(test_tensor)
    for _ in range(5):
        res2 = encoder_fn(test_tensor)
        assert np.all(res1 == res2)
