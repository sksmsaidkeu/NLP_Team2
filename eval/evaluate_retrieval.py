"""Retrieval 평가(명세 ①): DB는 train+val 영상의 프레임, 질의는 test 영상의 프레임.
건물마다 가장 비슷한 프레임의 유사도로 8개 건물 순위를 매겨 Recall@1/2/4·MRR을 구하고, 프레임 → 영상 → seed 순으로 평균한다.
실행: python eval/evaluate_retrieval.py   (결과: results/retrieval_results.csv)
"""
import os

import pandas as pd
import numpy as np

from data_loading import RESULTS_DIR, SEEDS, load_frames, load_splits, building_max_scores

RECALL_KS = (1, 2, 4)


def evaluate_seed(seed, splits, frames, embeddings):
    role_of_video = splits.set_index("video_id")[f"seed{seed}"]
    building_of_video = splits.set_index("video_id")["building"]
    buildings = sorted(building_of_video.unique())

    frame_role = frames.video_id.map(role_of_video).to_numpy()
    frame_building = frames.video_id.map(building_of_video).to_numpy()
    is_query = frame_role == "test"
    is_database = ~is_query
    assert not set(frames.video_id[is_database]) & set(frames.video_id[is_query])

    similarity = embeddings[is_query] @ embeddings[is_database].T
    scores = building_max_scores(similarity, frame_building[is_database], buildings)
    true_index = np.array([buildings.index(building) for building in frame_building[is_query]])
    true_score = scores[np.arange(len(scores)), true_index]
    rank_of_true_building = (scores > true_score[:, None]).sum(axis=1) + 1

    per_frame = pd.DataFrame({"video": frames.video_id[is_query].to_numpy(), "rank": rank_of_true_building})
    for k in RECALL_KS:
        per_frame[f"R@{k}"] = per_frame["rank"] <= k
    per_frame["MRR"] = 1 / per_frame["rank"]
    per_video = per_frame.drop(columns="rank").groupby("video").mean()
    print(f"seed{seed}: test 영상 {len(per_video)}개, 질의 프레임 {len(per_frame)}개")
    return per_video.mean().rename(f"seed{seed}")


def main():
    splits = load_splits()
    frames, embeddings = load_frames()
    assert len(sorted(splits.building.unique())) == 8

    table = pd.DataFrame([evaluate_seed(seed, splits, frames, embeddings) for seed in SEEDS])
    mean, std = table.mean(), table.std(ddof=0)
    table.loc["mean"], table.loc["std"] = mean, std
    table.loc["random"] = [1 / 8, 2 / 8, 4 / 8, np.mean([1 / rank for rank in range(1, 9)])]
    print(table.round(3).to_string())
    table.round(4).to_csv(os.path.join(RESULTS_DIR, "retrieval_results.csv"), encoding="utf-8-sig")


if __name__ == "__main__":
    main()
