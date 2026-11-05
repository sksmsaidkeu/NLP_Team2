"""검색 비교 평가: 정답 청크(data/eval_evidence.xlsx의 확정_섹션)가 1·3위 안에 드는 비율.

대상은 청크 정답이 있는 행. 질의 사진은 그 영상의 프레임 3장(25/50/75%)이고, 식별·닮은 장소 설명에서 그 영상 자신은 뺀다(LOO).
생성은 하지 않고 검색만 본다. 방법마다 generate.py의 rank_chunks/tune_cands를 그대로 부른다.
사용: python eval/eval_retrieval.py [방법1,방법2,...]   → 표 출력 + results/eval_retrieval_results.csv (방법을 고르면 csv는 안 씀)
"""
import pandas as pd  # torch보다 먼저 (build_chroma.py 참고)
import os
import sys

import chromadb
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 프로젝트 루트
DATA = os.path.join(ROOT, "data")  # frames, building_docs, splits.csv 등
sys.path.insert(0, os.path.join(ROOT, "rag"))  # dino · search · generate 모듈
import generate
import search

client = chromadb.HttpClient(host="localhost", port=8000)

# 청크 전체 (id, 문서, 임베딩, 건물, 섹션)
g = client.get_collection("building_docs", embedding_function=None).get(include=["documents", "embeddings", "metadatas"])
C_ID, C_DOC, C_EMB = g["ids"], g["documents"], np.array(g["embeddings"])
C_BLD = [m["character"] for m in g["metadatas"]]
C_SEC = [m["section"] for m in g["metadatas"]]
ID_OF = dict(zip(C_DOC, C_ID))

# 프레임 DINO 임베딩
d = client.get_collection("frames_dino", embedding_function=None).get(include=["embeddings", "metadatas"])
F_ID, F_EMB = d["ids"], np.array(d["embeddings"])
F_BLD = np.array([m["character"] for m in d["metadatas"]])
F_VID = np.array([m["video_id"] for m in d["metadatas"]])
F_IDX = {f: i for i, f in enumerate(F_ID)}
frames = pd.read_csv(os.path.join(DATA, "frames.csv"))

METHODS = {  # 이름: (mode, alpha, rerank, loc_b, loc_a, oracle)
    "text-only(전 건물)": ("weighted", 0.0, 0, 0, 0, 0),
    "filter": ("filter", 0, 0, 0, 0, 0),
    "weighted a=.1": ("weighted", 0.1, 0, 0, 0, 0),
    "echo": ("echo", 0, 0, 0, 0, 0),
    "echo+rerank": ("echo", 0, 1, 0, 0, 0),
    "echo+rerank+B": ("echo", 0, 1, 1, 0, 0),
    "echo+rerank+A": ("echo", 0, 1, 0, 1, 0),
    "echo+A": ("echo", 0, 0, 0, 1, 0),
    "echo+rerank+B+A": ("echo", 0, 1, 1, 1, 0),
    "[식별 정답] filter": ("filter", 0, 0, 0, 0, 1),
    "[식별 정답] +rerank": ("echo", 0, 1, 0, 0, 1),
    "[식별 정답] +rerank+B": ("echo", 0, 1, 1, 0, 1),
    "[식별 정답] +rerank+A": ("echo", 0, 1, 0, 1, 1),
    "[식별 정답] +A": ("echo", 0, 0, 0, 1, 1),
    "[식별 정답] +rerank+B+A": ("echo", 0, 1, 1, 1, 1),
}


def identify(frame_id, vid):
    # 그 영상 자신을 뺀 식별: (건물별 확률 votes, 닮은 프레임 hits)
    q = F_EMB[F_IDX[frame_id]]
    s = np.where(F_VID == vid, -9, F_EMB @ q)
    order = np.argsort(-s)
    sims = {}
    for i in order[: len(order) - int((F_VID == vid).sum())]:
        sims.setdefault(F_BLD[i], []).append(s[i])
    hits = [(F_BLD[i], float(s[i]), F_VID[i], F_ID[i]) for i in order[:10]]
    return search.vote(sims), hits


def retrieve(qvec, question, votes, hits, true_b, mode, alpha, rerank, loc_b, loc_a, oracle, k=3):
    share = {true_b: 1.0} if oracle else dict(votes)
    cands = [(float(C_EMB[i] @ qvec), C_BLD[i], C_SEC[i], C_DOC[i], C_EMB[i]) for i in range(len(C_ID))]
    if mode == "echo" and (rerank or loc_a):
        top_b = max(share, key=share.get)
        loc = generate.near_places(hits, search.img_col, top_b, generate.LOC_K)
        cands = generate.tune_cands(cands, share, question, loc, rerank, loc_b, loc_a, search.txt_model)
    top = generate.rank_chunks(cands, share, mode, alpha, k)
    return [ID_OF[c[3]] for c in top]


if __name__ == "__main__":
    ev = pd.read_excel(os.path.join(DATA, "eval_evidence.xlsx"))
    ev = ev[~ev.확정_섹션.isin(["없음", "없음(부정형)", "식별만"])].reset_index(drop=True)
    gold = [{x.split(" | ")[0].strip() for x in s.split(";")} for s in ev.확정_섹션]
    qvecs = search.txt_model.encode(ev.질문.tolist(), normalize_embeddings=True)

    only = sys.argv[1].split(",") if len(sys.argv) > 1 else None
    rows = []
    for n, r in ev.iterrows():
        fr = frames[frames.video_id == r.video_id].sort_values("frame_idx").frame_path.tolist()
        picks = [fr[len(fr) // 4], fr[len(fr) // 2], fr[3 * len(fr) // 4]]
        for fid in picks:
            votes, hits = identify(fid, r.video_id)
            id_ok = max(votes, key=lambda kv: kv[1])[0] == r.건물
            for name, (mode, alpha, rr, lb, la, orc) in METHODS.items():
                if only and name not in only:
                    continue
                ids = retrieve(qvecs[n], r.질문, votes, hits, r.건물, mode, alpha, rr, lb, la, orc)
                rows.append(dict(row=n, 건물=r.건물, 유형=r.유형, 확신도=r.확신도, 방법=name, id_ok=id_ok,
                                 hit1=ids[0] in gold[n], hit3=any(i in gold[n] for i in ids)))
    res = pd.DataFrame(rows)
    if not only:
        res.to_csv(os.path.join(ROOT, "results", "eval_retrieval_results.csv"), index=False, encoding="utf-8-sig")

    per_row = res.groupby(["방법", "row"], sort=False).agg(hit1=("hit1", "mean"), hit3=("hit3", "mean"), 유형=("유형", "first"), 확신도=("확신도", "first")).reset_index()
    print(f"행 {len(ev)}개 (확신 {(ev.확신도 == '확신').sum()}, 애매 {(ev.확신도 == '애매').sum()}), 행당 사진 3장. 식별 정확도 {res.id_ok.mean():.1%}")
    fmt = lambda s: f"{s.mean():.1%}"
    out = pd.DataFrame({
        "hit@1": per_row.groupby("방법", sort=False).hit1.apply(fmt),
        "hit@3": per_row.groupby("방법", sort=False).hit3.apply(fmt),
        "inside@1": per_row[per_row.유형 == "inside"].groupby("방법", sort=False).hit1.apply(fmt),
        "outside@1": per_row[per_row.유형 == "outside"].groupby("방법", sort=False).hit1.apply(fmt),
        "확신@1": per_row[per_row.확신도 == "확신"].groupby("방법", sort=False).hit1.apply(fmt),
        "애매@1": per_row[per_row.확신도 == "애매"].groupby("방법", sort=False).hit1.apply(fmt),
    })
    print(out.to_string())
