"""프레임 이미지를 임베딩으로 바꾸는 인코더 모음.
새 인코더는 `(PIL 이미지 목록) -> (N, D) 단위벡터` 함수를 만들어 맨 아래 ENCODERS에 이름과 함께 등록하면 된다.
(행마다 길이가 1이어야 한다: 평가가 내적을 코사인 유사도로 쓴다.)
"""
from functools import lru_cache

import numpy as np
from PIL import Image

DINOV2_NAME = "facebook/dinov2-base"
PAD_COLOR = (255, 255, 255)


def pad_square(image, color=PAD_COLOR):
    image = image.convert("RGB")
    side = max(image.size)
    canvas = Image.new("RGB", (side, side), color)
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return canvas


def unit_rows(matrix):
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


@lru_cache(maxsize=None)
def load_dinov2():
    import torch
    from transformers import AutoImageProcessor, AutoModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(DINOV2_NAME).to(device).eval()
    if device == "cuda":
        model = model.half()
    return torch, device, AutoImageProcessor.from_pretrained(DINOV2_NAME), model


def embed_dinov2(images):
    """[CLS 벡터 | 패치 평균 벡터]를 각각 단위벡터로 만든 뒤 이어 붙인다 → (N, 1536). 이미지는 정사각 패딩된 상태여야 한다."""
    torch, device, processor, model = load_dinov2()
    pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device, model.dtype)
    with torch.no_grad():
        hidden = model(pixel_values=pixels).last_hidden_state.float().cpu().numpy()
    return np.hstack([unit_rows(hidden[:, 0]), unit_rows(hidden[:, 1:].mean(axis=1))]) / np.sqrt(2)


ENCODERS = {"dinov2": embed_dinov2}
