"""QA 평가(명세 ②): test 영상마다 inside·outside 예/아니오 질문 2개. 근거는 영상 설명(place_description)만 쓴다.
조건 oracle: 질문이 달린 영상 자신의 설명.
조건 echo: 질의 사진 → 상위 3개 건물의 다른 영상 설명을 후보로 → 건물 확률과 재정렬기 점수를 0.5:0.5로 섞어 1개 선택(자기 영상 제외).
영상 단위 평균 정확도(seed별) → 평균±표준편차, 혼동행렬(seed 합산).
실행: python eval/evaluate_qa.py [seed ...]   (기본 0 1 2 3, 환경변수 ANTHROPIC_API_KEY 필요)
호출 기록은 results/qa_calls.jsonl에 쌓이고, 다시 실행하면 이미 있는 호출은 건너뛴다.
"""
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np

from data_loading import RESULTS_DIR, SEEDS, load_descriptions, load_frames, load_splits
from reranker import minmax, rerank_scores

CALLS_PATH = os.path.join(RESULTS_DIR, "qa_calls.jsonl")
SUMMARY_PATH = os.path.join(RESULTS_DIR, "qa_summary.csv")
MODEL = "claude-haiku-5-5"
QUESTION_TYPES = ("inside", "outside")
PHOTOS_PER_VIDEO = 3
CANDIDATE_BUILDINGS = 3
PHOTO_WEIGHT = 0.5
SOFTMAX_TEMPERATURE = 0.02
ANSWERS = ("예", "아니오", "기타")
CONDITIONS = ("echo", "oracle")

# 프롬프트는 RAG 평가 가이드 슬라이드 35~37 그대로 (temperature 0은 이 모델이 지원하지 않아 적용하지 못함)
SYSTEM_PROMPT = "당신은 항상 사용자가 묻는 질문에만 답합니다. 그 외의 것에는 답하지 마세요."
ANSWER_INSTRUCTION = "설명 없이 '예' 또는 '아니오'로만 답하세요."
FEW_SHOT_EXAMPLES = (
    "Context: 알프스산맥의 남쪽은 코모 호수와 맞닿아 있다.\n"
    "Question: 이 산은 코모 호수 안에 있거나 그 옆에 있나요?\n" + ANSWER_INSTRUCTION + "\nShort answer is: 예\n\n"
    "Context: 에펠탑은 프랑스 파리의 센강 근처에 있다.\n"
    "Question: 이 건물은 이탈리아 로마에 있나요?\n" + ANSWER_INSTRUCTION + "\nShort answer is: 아니오\n\n")


def parse_answer(text):
    text = text.strip()
    if text.startswith("아니오"):
        return "아니오"
    return "예" if text.startswith("예") else "기타"


def build_prompt(context, question):
    return f"{FEW_SHOT_EXAMPLES}Context: {context}\nQuestion: {question}\n{ANSWER_INSTRUCTION}\nShort answer is:"


def questions_of(descriptions, video_id):
    row = descriptions.loc[video_id]
    return [(kind, row[f"query_{kind}"], row[f"ground_truth_{kind}"].strip()) for kind in QUESTION_TYPES]


def building_probabilities(query_embedding, other_embeddings, other_buildings, buildings):
    similarity = other_embeddings @ query_embedding
    best = {building: similarity[other_buildings == building].max() for building in buildings}
    ranked = sorted(best, key=best.get, reverse=True)
    weights = {building: math.exp((best[building] - best[ranked[0]]) / SOFTMAX_TEMPERATURE) for building in buildings}
    return ranked, {building: weight / sum(weights.values()) for building, weight in weights.items()}


def pick_description(question, candidates, probability, building_of_video, descriptions, rerank_cache):
    cache_key = (question, tuple(candidates))
    if cache_key not in rerank_cache:
        texts = [descriptions.at[candidate, "place_description"] for candidate in candidates]
        rerank_cache[cache_key] = minmax(rerank_scores(question, texts))
    scores = [PHOTO_WEIGHT * probability[building_of_video[candidate]] + (1 - PHOTO_WEIGHT) * rerank
              for candidate, rerank in zip(candidates, rerank_cache[cache_key])]
    return candidates[int(np.argmax(scores))]


def build_jobs(test_videos, splits, frames, embeddings, descriptions, finished_keys=()):
    """호출할 작업 목록. echo 작업 중 finished_keys에 있는 것은 (재정렬기를 돌리지 않으려고) 만들지 않는다."""
    building_of_video = dict(zip(splits.video_id, splits.building))
    buildings = sorted(set(building_of_video.values()))
    videos_of_building = {building: sorted(splits.video_id[splits.building == building]) for building in buildings}
    frame_video = frames.video_id.to_numpy()
    frame_building = frames.video_id.map(building_of_video).to_numpy()
    rerank_cache = {}
    jobs = []

    for video in test_videos:
        questions = questions_of(descriptions, video)
        for kind, question, truth in questions:
            jobs.append({"key": f"oracle|{video}|-|{kind}", "cond": "oracle", "video": video, "frame": "-", "qtype": kind,
                         "gt": truth, "prompt": build_prompt(descriptions.at[video, "place_description"], question)})

        own_frames = np.where(frame_video == video)[0]
        photo_positions = np.linspace(0, len(own_frames) - 1, PHOTOS_PER_VIDEO).round().astype(int)
        for query_index in own_frames[photo_positions]:
            frame_idx = int(frames.frame_idx.iloc[query_index])
            is_other = frame_video != video
            ranked, probability = building_probabilities(embeddings[query_index], embeddings[is_other],
                                                         frame_building[is_other], buildings)
            candidates = [candidate for building in ranked[:CANDIDATE_BUILDINGS]
                          for candidate in videos_of_building[building] if candidate != video]
            for kind, question, truth in questions:
                key = f"echo|{video}|{frame_idx}|{kind}"
                if key in finished_keys:
                    continue
                picked = pick_description(question, candidates, probability, building_of_video, descriptions, rerank_cache)
                assert picked != video
                jobs.append({"key": key, "cond": "echo", "video": video, "frame": frame_idx, "qtype": kind, "gt": truth,
                             "pick": picked, "pick_building_ok": building_of_video[picked] == building_of_video[video],
                             "prompt": build_prompt(descriptions.at[picked, "place_description"], question)})
    return jobs


def call_model(client, job):
    response = client.messages.create(model=MODEL, max_tokens=2048, system=SYSTEM_PROMPT,
                                      messages=[{"role": "user", "content": job["prompt"]}])
    text = "".join(block.text for block in response.content if block.type == "text")
    record = {key: value for key, value in job.items() if key != "prompt"}
    return {**record, "pred": parse_answer(text), "raw": text.strip()[:40],
            "tok_in": response.usage.input_tokens, "tok_out": response.usage.output_tokens}


def load_finished_calls():
    if not os.path.exists(CALLS_PATH):
        return {}
    with open(CALLS_PATH, encoding="utf-8") as file:
        return {record["key"]: record for record in map(json.loads, file)}


def summarize(finished, splits, seeds):
    results = pd.DataFrame(finished.values())
    results["correct"] = results.pred == results["gt"]
    confusion = {cond: pd.DataFrame(0, index=["예", "아니오"], columns=list(ANSWERS)) for cond in CONDITIONS}
    rows = []
    for seed in seeds:
        test_videos = set(splits.video_id[splits[f"seed{seed}"] == "test"])
        in_seed = results[results.video.isin(test_videos)]
        row = {"seed": seed}
        for cond in CONDITIONS:
            of_cond = in_seed[in_seed.cond == cond]
            row[f"Acc({cond})"] = of_cond.groupby("video").correct.mean().mean()
            for truth in ("예", "아니오"):
                for pred in ANSWERS:
                    confusion[cond].loc[truth, pred] += ((of_cond["gt"] == truth) & (of_cond.pred == pred)).sum()
        row["픽 건물 정답률"] = in_seed[in_seed.cond == "echo"].groupby("video").pick_building_ok.mean().mean()
        rows.append(row)
    table = pd.DataFrame(rows).set_index("seed")
    mean, std = table.mean(), table.std(ddof=0)
    table.loc["mean"], table.loc["std"] = mean, std
    return table, confusion


def check_parse_answer():
    assert parse_answer("예") == "예"
    assert parse_answer(" 아니오.") == "아니오"
    assert parse_answer("모르겠다") == "기타"
    assert parse_answer("") == "기타"


def main():
    check_parse_answer()
    seeds = [int(arg) for arg in sys.argv[1:]] or list(SEEDS)
    splits, (frames, embeddings), descriptions = load_splits(), load_frames(), load_descriptions()
    for video in splits.video_id:
        assert all(truth in ("예", "아니오") for _, _, truth in questions_of(descriptions, video))

    test_videos = sorted({video for seed in seeds for video in splits.video_id[splits[f"seed{seed}"] == "test"]})
    finished = load_finished_calls()
    jobs = build_jobs(test_videos, splits, frames, embeddings, descriptions, finished_keys=finished)
    todo = [job for job in jobs if job["key"] not in finished]
    print(f"호출 {len(todo)}건 (기록에 이미 있는 호출 {len(finished)}건)", flush=True)

    if todo:
        import anthropic
        client = anthropic.Anthropic()
        with ThreadPoolExecutor(8) as pool, open(CALLS_PATH, "a", encoding="utf-8") as file:
            for record in pool.map(lambda job: call_model(client, job), todo):
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
                file.flush()
                finished[record["key"]] = record
    tokens_in = sum(record["tok_in"] for record in finished.values())
    tokens_out = sum(record["tok_out"] for record in finished.values())
    print(f"누적 토큰 입력 {tokens_in:,} / 출력 {tokens_out:,}")

    table, confusion = summarize(finished, splits, seeds)
    print(table.round(3).to_string())
    for cond, matrix in confusion.items():
        print(f"\nConfusion Matrix ({cond}, 행=정답·열=예측, seed 합산)\n{matrix.to_string()}")
    table.round(4).to_csv(SUMMARY_PATH, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
