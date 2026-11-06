"""평가 공통 모듈: data/의 분할·프레임 임베딩·영상 설명을 읽고, 결과 파일 이름을 정하고, 건물별 최대 유사도를 계산한다.
(스크립트마다 pandas를 torch보다 먼저 import하는 이유: Windows에서 순서가 바뀌면 프로세스가 비정상 종료된다.)
"""
import os

import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
SEEDS = (0, 1, 2, 3)
DEFAULT_EMBEDDINGS = "dinov2"
DEFAULT_GENERATOR = "claude-haiku-5-5"


def add_embeddings_argument(parser):
    parser.add_argument("--embeddings", default=DEFAULT_EMBEDDINGS,
                        help="프레임 임베딩 이름: data/frame_embeddings_<이름>.npz (기본 dinov2)")


def embeddings_path(name):
    return os.path.join(DATA_DIR, f"frame_embeddings_{name}.npz")


def result_path(stem, *tags, extension):
    """results/<stem>_<tag>_<tag>.<extension>. 임베딩·생성기를 tag로 붙여 모델마다 결과가 따로 저장되게 한다."""
    return os.path.join(RESULTS_DIR, "_".join((stem, *tags)) + "." + extension)


def load_splits():
    return pd.read_csv(os.path.join(DATA_DIR, "splits.csv"))


def load_frames(embeddings_name=DEFAULT_EMBEDDINGS):
    archive = np.load(embeddings_path(embeddings_name))
    frames = pd.DataFrame({"video_id": archive["video_id"], "frame_idx": archive["frame_idx"]})
    embeddings = archive["embeddings"]
    assert embeddings.ndim == 2 and len(embeddings) == len(frames), "embeddings는 (프레임 수, 차원) 모양이어야 한다"
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-3), "임베딩이 단위벡터가 아니다(행마다 길이 1로 정규화)"
    assert set(frames.video_id) == set(load_splits().video_id), "splits.csv의 영상과 임베딩의 영상이 다르다"
    return frames, embeddings


def load_descriptions():
    return pd.read_csv(os.path.join(DATA_DIR, "video_descriptions.csv"), index_col="video_id")


def building_max_scores(similarity, db_frame_buildings, buildings):
    """similarity: (질의 프레임, DB 프레임). 건물마다 가장 비슷한 DB 프레임의 유사도를 골라 (질의 프레임, 건물) 표로 돌려준다."""
    return np.stack([similarity[:, db_frame_buildings == building].max(axis=1) for building in buildings], axis=1)
