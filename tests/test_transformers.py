from typing import Optional

import numpy as np
import pytest
import torch
from transformers import AutoModel, AutoTokenizer


def encoder_factory(model_name_or_path: str, device: Optional[str]):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path)
    model.eval()
    if device:
        model.to(device)

    def encode(text: str):
        tokenized = tokenizer(text, return_tensors="pt")
        if device:
            tokenized.to(device)
        with torch.no_grad():
            output = model(**tokenized)
        return output["last_hidden_state"].cpu().detach().numpy()

    return encode


@pytest.mark.parametrize(
    "encoder_fn",
    [
        encoder_factory("gpt2", device=None),
        encoder_factory("gpt2", device="mps"),
        encoder_factory("gpt2", device="cpu"),
        encoder_factory("roberta-base", device="mps"),
    ],
)
def test_idempotency(encoder_fn):
    the_text = "this is a test"
    res1 = encoder_fn(the_text)
    for _ in range(5):
        res2 = encoder_fn(the_text)
        assert np.all(res1 == res2)
