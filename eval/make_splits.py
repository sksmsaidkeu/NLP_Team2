"""낮 영상 129개를 영상 단위·건물 비율 유지(stratified)로 train/val/test로 나눈다 (seed 0~3).
입력: data/video_descriptions.csv   결과: data/splits.csv (video_id, building, seed0..seed3 = train|val|test)
실행: python eval/make_splits.py
"""
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from data_loading import DATA_DIR, SEEDS, load_descriptions

TEST_RATIO = 0.2
VAL_RATIO = 0.2


def split_once(videos, seed):
    train_val, test = train_test_split(videos, test_size=TEST_RATIO, stratify=videos.building, random_state=seed)
    train, val = train_test_split(train_val, test_size=VAL_RATIO, stratify=train_val.building, random_state=seed)
    ids = [set(train.video_id), set(val.video_id), set(test.video_id)]
    assert not ids[0] & ids[1] and not ids[0] & ids[2] and not ids[1] & ids[2] and len(set.union(*ids)) == len(videos)
    role_of_video = {**dict.fromkeys(ids[0], "train"), **dict.fromkeys(ids[1], "val"), **dict.fromkeys(ids[2], "test")}
    return videos.video_id.map(role_of_video)


def main():
    videos = load_descriptions().reset_index()[["video_id", "building"]]
    videos = videos.sort_values("video_id").reset_index(drop=True)
    assert len(videos) == 129 and videos.video_id.is_unique

    for seed in SEEDS:
        videos[f"seed{seed}"] = split_once(videos, seed)
        print(f"seed{seed}", pd.crosstab(videos.building, videos[f"seed{seed}"]).T.to_string(), "\n")
    videos.to_csv(os.path.join(DATA_DIR, "splits.csv"), index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
