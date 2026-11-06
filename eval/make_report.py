"""평가 결과를 슬라이드 42 양식의 표(results/results_table_<임베딩>_<생성기>.csv)와 seed×조건 혼동행렬 그림(confusion_matrices_….png)으로 정리한다. API 호출 없음.
입력: results/retrieval_<임베딩>.csv, qa_summary_<임베딩>_<생성기>.csv, qa_calls_<임베딩>_<생성기>.jsonl, data/splits.csv
실행: python eval/make_report.py [--embeddings 이름] [--generator 이름]
"""
import argparse
import json

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loading import DEFAULT_GENERATOR, SEEDS, add_embeddings_argument, load_splits, result_path

LABELS = ["예", "아니오"]
CONDITIONS = (("echo", "EchoSight"), ("oracle", "Oracle"))
plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "NanumGothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def make_results_table(embeddings, generator):
    retrieval = pd.read_csv(result_path("retrieval", embeddings, extension="csv"), index_col=0)
    qa_summary = pd.read_csv(result_path("qa_summary", embeddings, generator, extension="csv"), index_col=0)
    qa_summary.index = [f"seed{name}" if str(name).isdigit() else str(name) for name in qa_summary.index]
    table = retrieval.join(qa_summary[["Acc(echo)", "Acc(oracle)"]].rename(
        columns={"Acc(echo)": "Acc(EchoSight)", "Acc(oracle)": "Acc(Oracle)"}))
    table.index = [f"Seed {int(name[4:]) + 1}" if name.startswith("seed") else name for name in table.index]
    table = table.rename(index={"mean": "평균", "std": "표준편차", "random": "무작위 기준선"})
    table.loc["무작위 기준선", ["Acc(EchoSight)", "Acc(Oracle)"]] = float("nan")
    table.round(3).to_csv(result_path("results_table", embeddings, generator, extension="csv"), encoding="utf-8-sig")
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


def make_confusion_figure(embeddings, generator):
    splits = load_splits()
    with open(result_path("qa_calls", embeddings, generator, extension="jsonl"), encoding="utf-8") as file:
        calls = pd.DataFrame(map(json.loads, file))
    figure, axes = plt.subplots(2, len(SEEDS), figsize=(13, 6.4))
    for col, seed in enumerate(SEEDS):
        test_videos = set(splits.video_id[splits[f"seed{seed}"] == "test"])
        for row, (cond, name) in enumerate(CONDITIONS):
            in_seed = calls[(calls.cond == cond) & calls.video.isin(test_videos)]
            draw_confusion(axes[row, col], in_seed, f"{name} · Seed {seed + 1}")
    figure.suptitle("Confusion Matrix (EchoSight는 사진×질문, Oracle은 질문 단위 건수)", fontsize=12)
    figure.tight_layout()
    path = result_path("confusion_matrices", embeddings, generator, extension="png")
    figure.savefig(path, dpi=130)
    print("saved", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="평가 결과 표와 혼동행렬 그림")
    add_embeddings_argument(parser)
    parser.add_argument("--generator", default=DEFAULT_GENERATOR)
    args = parser.parse_args()
    make_results_table(args.embeddings, args.generator)
    make_confusion_figure(args.embeddings, args.generator)
