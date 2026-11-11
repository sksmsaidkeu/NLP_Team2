import argparse
import math

import pandas as pd
import numpy as np

from data_loading import SEEDS, add_embeddings_argument, load_descriptions, load_frames, load_splits, result_path
from reranker import minmax, rerank_scores

QUESTION_TYPES = ("inside", "outside")
CANDIDATE_BUILDINGS = 3
PHOTO_WEIGHT = 0.5
SOFTMAX_TEMPERATURE = 0.02
RECALL_KS = (1, 2, 4)

parser = argparse.ArgumentParser(description="재정렬기를 거친 건물 검색 평가")
add_embeddings_argument(parser)
args = parser.parse_args()

splits = load_splits()
frames, embeddings = load_frames(args.embeddings)
descriptions = load_descriptions()

building_of_video = dict(zip(splits.video_id, splits.building))
buildings = sorted(set(building_of_video.values()))
assert len(buildings) == 8
frame_video = frames.video_id.to_numpy()
frame_building = frames.video_id.map(building_of_video).to_numpy()

rerank_cache = {}
seed_rows = []

for seed in SEEDS:
    role_of_video = dict(zip(splits.video_id, splits[f"seed{seed}"]))
    frame_role = frames.video_id.map(role_of_video).to_numpy()
    is_database = frame_role != "test"
    test_videos = sorted(video for video, role in role_of_video.items() if role == "test")
    database_videos_of_building = {building: sorted(video for video, role in role_of_video.items()
                                                    if role != "test" and building_of_video[video] == building)
                                   for building in buildings}
    assert not set(frame_video[is_database]) & set(test_videos)

    frame_rows = []
    for video in test_videos:
        true_building = building_of_video[video]
        true_index = buildings.index(true_building)
        questions = [descriptions.at[video, f"query_{kind}"] for kind in QUESTION_TYPES]

        for query_index in np.where(frame_video == video)[0]:
            # 사진만으로 건물 순위 (evaluate_qa.building_probabilities와 같은 계산)
            similarity = embeddings[is_database] @ embeddings[query_index]
            best_similarity = {building: similarity[frame_building[is_database] == building].max() for building in buildings}
            photo_ranking = sorted(best_similarity, key=best_similarity.get, reverse=True)
            weights = {building: math.exp((best_similarity[building] - best_similarity[photo_ranking[0]]) / SOFTMAX_TEMPERATURE)
                       for building in buildings}
            probability = {building: weight / sum(weights.values()) for building, weight in weights.items()}
            photo_rank = photo_ranking.index(true_building) + 1

            top_buildings = photo_ranking[:CANDIDATE_BUILDINGS]
            candidates = [candidate for building in top_buildings for candidate in database_videos_of_building[building]]

            rerank_ranks = []
            for question in questions:
                # 질문 + 후보 묶음이 같으면 재정렬기를 다시 돌리지 않는다 (evaluate_qa.pick_description과 같은 방식)
                cache_key = (question, tuple(candidates))
                if cache_key not in rerank_cache:
                    texts = [descriptions.at[candidate, "place_description"] for candidate in candidates]
                    rerank_cache[cache_key] = minmax(rerank_scores(question, texts))
                candidate_scores = [PHOTO_WEIGHT * probability[building_of_video[candidate]] + (1 - PHOTO_WEIGHT) * rerank
                                    for candidate, rerank in zip(candidates, rerank_cache[cache_key])]

                best_candidate_score = {}
                for candidate, score in zip(candidates, candidate_scores):
                    building = building_of_video[candidate]
                    best_candidate_score[building] = max(score, best_candidate_score.get(building, -1))
                reranked_top = sorted(best_candidate_score, key=best_candidate_score.get, reverse=True)
                rerank_ranking = reranked_top + photo_ranking[CANDIDATE_BUILDINGS:]
                rerank_ranks.append(rerank_ranking.index(true_building) + 1)

            row = {"video": video}
            for k in RECALL_KS:
                row[f"photo R@{k}"] = photo_rank <= k
                row[f"rerank R@{k}"] = np.mean([rank <= k for rank in rerank_ranks])
            row["photo MRR"] = 1 / photo_rank
            row["rerank MRR"] = np.mean([1 / rank for rank in rerank_ranks])
            frame_rows.append(row)

    # 프레임 → 영상 → seed 순으로 평균
    per_video = pd.DataFrame(frame_rows).groupby("video").mean()
    print(f"seed{seed}: test 영상 {len(per_video)}개, 질의 프레임 {len(frame_rows)}개", flush=True)
    seed_rows.append(per_video.mean().rename(f"seed{seed}"))

table = pd.DataFrame(seed_rows)
table = table[[f"{name} {metric}" for name in ("photo", "rerank") for metric in ("R@1", "R@2", "R@4", "MRR")]]
mean, std = table.mean(), table.std(ddof=0)
table.loc["mean"], table.loc["std"] = mean, std
table.loc["random"] = [1 / 8, 2 / 8, 4 / 8, np.mean([1 / rank for rank in range(1, 9)])] * 2
print(table.round(3).to_string())
table.round(4).to_csv(result_path("retrieval_reranker", args.embeddings, extension="csv"), encoding="utf-8-sig")
