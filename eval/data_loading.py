"""평가 공통 모듈: data/의 분할·프레임 임베딩·영상 설명을 읽고, 건물별 최대 유사도를 계산한다.
(스크립트마다 pandas를 torch보다 먼저 import하는 이유: Windows에서 순서가 바뀌면 프로세스가 비정상 종료된다.)
"""
import os

import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
SEEDS = (0, 1, 2, 3)


def load_splits():
    return pd.read_csv(os.path.join(DATA_DIR, "splits.csv"))


def load_frames():
    archive = np.load(os.path.join(DATA_DIR, "frame_embeddings.npz"))
    frames = pd.DataFrame({"video_id": archive["video_id"], "frame_idx": archive["frame_idx"]})
    return frames, archive["embeddings"]


def load_descriptions():
    return pd.read_csv(os.path.join(DATA_DIR, "video_descriptions.csv"), index_col="video_id")


def building_max_scores(similarity, db_frame_buildings, buildings):
    """similarity: (질의 프레임, DB 프레임). 건물마다 가장 비슷한 DB 프레임의 유사도를 골라 (질의 프레임, 건물) 표로 돌려준다."""
    return np.stack([similarity[:, db_frame_buildings == building].max(axis=1) for building in buildings], axis=1)
