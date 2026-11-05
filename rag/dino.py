import pandas  # noqa: F401  torch보다 먼저 (build_chroma.py 참고)
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

NAME = "facebook/dinov2-base"
device = "cuda" if torch.cuda.is_available() else "cpu"
_proc = AutoImageProcessor.from_pretrained(NAME)
_model = AutoModel.from_pretrained(NAME).to(device).eval()
if device == "cuda":
    _model = _model.half()


def _unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def embed(images, batch=32):
    # images: 정사각 패딩된 PIL 리스트 → (N, 1536) 단위벡터. [CLS | 패치 평균]을 각각 정규화해 이어 붙임
    out = []
    for i in range(0, len(images), batch):
        x = _proc(images=images[i:i + batch], return_tensors="pt")["pixel_values"].to(device, _model.dtype)
        with torch.no_grad():
            h = _model(pixel_values=x).last_hidden_state.float().cpu().numpy()
        out.append(np.hstack([_unit(h[:, 0]), _unit(h[:, 1:].mean(1))]) / np.sqrt(2))
    return np.concatenate(out)
