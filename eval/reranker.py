"""질문과 영상 설명의 관련도를 bge-reranker-v2-m3(추가 학습 없는 사전학습 모델)로 계산한다. 모델은 처음 호출할 때 불러온다."""
import numpy as np

MODEL_NAME = "BAAI/bge-reranker-v2-m3"
_model = None


def rerank_scores(query, documents):
    global _model
    if _model is None:
        import torch
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(MODEL_NAME, device="cuda" if torch.cuda.is_available() else "cpu",
                              model_kwargs={"torch_dtype": torch.float16})
    return [float(score) for score in _model.predict([(query, document) for document in documents])]


def minmax(values):
    values = np.asarray(values, dtype=float)
    spread = values.max() - values.min()
    return (values - values.min()) / spread if spread > 0 else np.zeros_like(values)
