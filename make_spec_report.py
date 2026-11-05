"""명세 결과 정리(API 호출 없음): ① 슬라이드 42 양식의 결과 표 results/eval_spec_results_table.csv
② seed별·조건별 2x2 Confusion Matrix 그림 results/eval_spec_confusion.png (예/아니오 외의 답 "기타"는 칸에 못 넣어 제목에 건수로 표시, 오답으로 셈)
입력(results/): eval_spec_retrieval_results.csv, eval_spec_qa_summary.csv, eval_spec_qa_results.jsonl, splits.csv
"""
import pandas as pd  # torch보다 먼저 (build_chroma.py 참고)
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
P = lambda f: os.path.join(HERE, f)
R = lambda f: os.path.join(HERE, "results", f)  # 결과 파일 폴더
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

r = pd.read_csv(R("eval_spec_retrieval_results.csv"), index_col=0)
q = pd.read_csv(R("eval_spec_qa_summary.csv"), index_col=0)
q.index = [f"seed{i}" if str(i).isdigit() else str(i) for i in q.index]  # QA 요약은 0~3, 검색 요약은 seed0~3
t = r.join(q[["Acc(echo)", "Acc(oracle)"]].rename(columns={"Acc(echo)": "Acc(EchoSight)", "Acc(oracle)": "Acc(Oracle)"}))
t.index = [f"Seed {int(i[4:]) + 1}" if i.startswith("seed") else i for i in t.index]  # 슬라이드는 Seed 1~4 (코드는 seed0~3)
t = t.rename(index={"mean": "평균", "std": "표준편차", "random": "무작위 기준선"})
t.loc["무작위 기준선", ["Acc(EchoSight)", "Acc(Oracle)"]] = float("nan")
t.round(3).to_csv(R("eval_spec_results_table.csv"), encoding="utf-8-sig")
print(t.round(3).to_string())

sp = pd.read_csv(P("splits.csv"))
d = pd.DataFrame(map(json.loads, open(R("eval_spec_qa_results.jsonl"), encoding="utf-8")))
fig, ax = plt.subplots(2, 4, figsize=(13, 6.4))
for j, s in enumerate(range(4)):
    test = set(sp.video_id[sp[f"seed{s}"] == "test"])
    for i, (c, name) in enumerate((("echo", "EchoSight"), ("oracle", "Oracle"))):
        x = d[(d.cond == c) & d.video.isin(test)]
        m = pd.crosstab(x["gt"], x.pred).reindex(index=["예", "아니오"], columns=["예", "아니오"], fill_value=0)
        etc = int((~x.pred.isin(["예", "아니오"])).sum())
        a = ax[i, j]
        a.imshow(m.values, cmap="Blues", vmin=0, vmax=m.values.max())
        for y in range(2):
            for xx in range(2):
                v = m.values[y, xx]
                a.text(xx, y, int(v), ha="center", va="center", color="white" if v > m.values.max() / 2 else "black", fontsize=14)
        a.set_xticks([0, 1], ["예", "아니오"]); a.set_yticks([0, 1], ["예", "아니오"])
        a.set_xlabel("예측"); a.set_ylabel("정답")
        a.set_title(f"{name} · Seed {s + 1}" + (f" (기타 {etc})" if etc else ""), fontsize=11)
fig.suptitle("Confusion Matrix (EchoSight는 사진×질문, Oracle은 질문 단위 건수)", fontsize=12)
fig.tight_layout()
fig.savefig(R("eval_spec_confusion.png"), dpi=130)
print("saved eval_spec_confusion.png")
