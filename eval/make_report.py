"""평가 결과를 슬라이드 42 양식의 표(results/results_table.csv)와 seed×조건 혼동행렬 그림(results/confusion_matrices.png)으로 정리한다. API 호출 없음.
입력: results/retrieval_results.csv, results/qa_summary.csv, results/qa_calls.jsonl, data/splits.csv
실행: python eval/make_report.py
"""
import json
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loading import RESULTS_DIR, SEEDS, load_splits

LABELS = ["예", "아니오"]
CONDITIONS = (("echo", "EchoSight"), ("oracle", "Oracle"))
plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "NanumGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def results_path(name):
    return os.path.join(RESULTS_DIR, name)


def make_results_table():
    retrieval = pd.read_csv(results_path("retrieval_results.csv"), index_col=0)
    qa = pd.read_csv(results_path("qa_summary.csv"), index_col=0)
    qa.index = [f"seed{i}" if str(i).isdigit() else str(i) for i in qa.index]
    table = retrieval.join(qa[["Acc(echo)", "Acc(oracle)"]].rename(
        columns={"Acc(echo)": "Acc(EchoSight)", "Acc(oracle)": "Acc(Oracle)"}))
    table.index = [f"Seed {int(name[4:]) + 1}" if name.startswith("seed") else name for name in table.index]
    table = table.rename(index={"mean": "평균", "std": "표준편차", "random": "무작위 기준선"})
    table.loc["무작위 기준선", ["Acc(EchoSight)", "Acc(Oracle)"]] = float("nan")
    table.round(3).to_csv(results_path("results_table.csv"), encoding="utf-8-sig")
    print(table.round(3).to_string())


def draw_confusion(axis, calls, title):
    matrix = pd.crosstab(calls["gt"], calls.pred).reindex(index=LABELS, columns=LABELS, fill_value=0)
    others = int((~calls.pred.isin(LABELS)).sum())
    axis.imshow(matrix.values, cmap="Blues", vmin=0, vmax=matrix.values.max())
    for row in range(2):
        for col in range(2):
            count = matrix.values[row, col]
            axis.text(col, row, int(count), ha="center", va="center", fontsize=14,
                      color="white" if count > matrix.values.max() / 2 else "black")
    axis.set_xticks([0, 1], LABELS)
    axis.set_yticks([0, 1], LABELS)
    axis.set_xlabel("예측")
    axis.set_ylabel("정답")
    axis.set_title(title + (f" (기타 {others})" if others else ""), fontsize=11)


def make_confusion_figure():
    splits = load_splits()
    with open(results_path("qa_calls.jsonl"), encoding="utf-8") as file:
        calls = pd.DataFrame(map(json.loads, file))
    figure, axes = plt.subplots(2, len(SEEDS), figsize=(13, 6.4))
    for col, seed in enumerate(SEEDS):
        test_videos = set(splits.video_id[splits[f"seed{seed}"] == "test"])
        for row, (cond, name) in enumerate(CONDITIONS):
            in_seed = calls[(calls.cond == cond) & calls.video.isin(test_videos)]
            draw_confusion(axes[row, col], in_seed, f"{name} · Seed {seed + 1}")
    figure.suptitle("Confusion Matrix (EchoSight는 사진×질문, Oracle은 질문 단위 건수)", fontsize=12)
    figure.tight_layout()
    figure.savefig(results_path("confusion_matrices.png"), dpi=130)
    print("saved confusion_matrices.png")


if __name__ == "__main__":
    make_results_table()
    make_confusion_figure()
