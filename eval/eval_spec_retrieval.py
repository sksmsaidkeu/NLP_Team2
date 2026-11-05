"""명세 Retrieval 평가: DB=train+val 영상 프레임, 질의=test 영상 프레임(DINOv2).
프레임별 건물 max-pooling 유사도 → 8건물 순위 → Recall@1/2/4·MRR, 영상 단위 평균 → seed 평균±표준편차.
실행: python eval/eval_spec_retrieval.py  (Docker chroma 필요)
"""
import pandas as pd  # torch보다 먼저 (build_chroma.py 참고)
import os
import numpy as np
import chromadb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 프로젝트 루트
DATA = os.path.join(ROOT, "data")  # frames, building_docs, splits.csv 등
KS = (1, 2, 4)

sp = pd.read_csv(os.path.join(DATA, "splits.csv"))
g = chromadb.HttpClient(host="localhost", port=8000).get_collection("frames_dino").get(include=["embeddings", "metadatas"])
fr = pd.DataFrame(g["metadatas"])
X = np.asarray(g["embeddings"], dtype=np.float32)  # 단위벡터라 내적 = 코사인
fr = fr.merge(sp[["video_id"]].assign(ok=True), on="video_id", how="left")
keep = fr.ok.fillna(False).to_numpy(bool)  # 낮 129개 영상의 프레임만 (사진·밤 영상 제외)
fr, X = fr[keep].reset_index(drop=True), X[keep]
bld = sorted(sp.character.unique())
assert len(bld) == 8
vb = dict(zip(sp.video_id, sp.character))
fb = fr.video_id.map(vb).to_numpy()

rows = []
for s in range(4):
    role = fr.video_id.map(dict(zip(sp.video_id, sp[f"seed{s}"]))).to_numpy()
    db, q = role != "test", role == "test"
    assert not set(fr.video_id[db]) & set(fr.video_id[q])  # 영상 ID 겹침 금지
    sim = X[q] @ X[db].T
    # 건물별 max-pool → (질의 프레임, 8)
    sc = np.stack([sim[:, fb[db] == b].max(1) for b in bld], 1)
    true = np.array([bld.index(b) for b in fb[q]])
    rank = (sc > sc[np.arange(len(sc)), true][:, None]).sum(1) + 1  # 정답 건물 순위(1-based)
    d = pd.DataFrame({"video": fr.video_id[q].to_numpy(), "rank": rank})
    for k in KS:
        d[f"R@{k}"] = d["rank"] <= k
    d["MRR"] = 1 / d["rank"]
    v = d.drop(columns="rank").groupby("video").mean().mean()  # 프레임 평균 → 영상 평균
    rows.append(v.rename(f"seed{s}"))
    print(f"seed{s}: test 영상 {d.video.nunique()}개, 질의 프레임 {len(d)}개")

res = pd.DataFrame(rows)
res.loc["mean"], res.loc["std"] = res.mean(), res.std(ddof=0)
res.loc["random"] = [1 / 8, 2 / 8, 4 / 8, np.mean([1 / r for r in range(1, 9)])]  # 0.125/0.25/0.5/0.34
print(res.round(3).to_string())
res.round(4).to_csv(os.path.join(ROOT, "results", "eval_spec_retrieval_results.csv"), encoding="utf-8-sig")
