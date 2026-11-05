"""평가용 정답 청크 초안: metadata.xlsx의 (영상, inside/outside) 질문마다 근거 청크 후보를 뽑아 data/eval_evidence_draft.xlsx로 저장.

방법: ① 정답 문자열(쉼표로 나눈 항목)이 그 건물 청크에 얼마나 들어 있는지 ② 재정렬기 점수(질문+정답 ↔ 청크).
둘을 합쳐 1·2순위 후보를 적고, 확신이 낮으면 검토 필요=Y. 최종 확정은 사람이 '확정_섹션' 열에 적는다.
정답이 청크에는 없고 그 영상의 장소 설명에만 있으면 근거출처=장소설명 (문서로는 못 푸는 질문).
"""
import pandas as pd  # torch보다 먼저 (build_chroma.py 참고)
import os
import re

import sys

import chromadb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 프로젝트 루트
DATA = os.path.join(ROOT, "data")  # frames, building_docs, splits.csv 등
sys.path.insert(0, os.path.join(ROOT, "rag"))  # dino · search · generate 모듈
import generate

SKIP_STR = {"예", "아니오", "아니요", "없다", "있다", "존재한다", "존재하지않는다"}  # 문자열 일치가 의미 없는 짧은 답


def norm(x):
    return re.sub(r"[\s\.,·()\[\]'\"]", "", str(x))


def found_frac(gt, text):
    # 정답 항목(쉼표·슬래시 구분) 중 text에 들어 있는 비율. 의미 없는 짧은 답은 None
    items = [norm(i) for i in re.split(r"[,/]", str(gt)) if norm(i)]
    if not items or all(i in SKIP_STR for i in items):
        return None
    t = norm(text)
    return sum(i in t for i in items) / len(items)


def pick(question, gt, chunks):
    # chunks: [(id, 섹션, 문서)] → 점수순 [(id, 섹션, 점수, 일치율)]
    rr = generate.rerank_scores(f"{question} {gt}", [c[2] for c in chunks])
    out = []
    for (cid, sec, doc), r in zip(chunks, rr):
        f = found_frac(gt, doc)
        out.append((cid, sec, r + (f or 0), r, f))  # 일치 항목이 있으면 가산 (최대 +1)
    return sorted(out, key=lambda t: -t[2])


if __name__ == "__main__":
    assert found_frac("우리은행", "1층에 우리 은행이 있다") == 1 and found_frac("예", "예예") is None
    assert found_frac("학사지원팀, 증명서ATM기기", "학사지원팀이 있다") == 0.5
    col = chromadb.HttpClient(host="localhost", port=8000).get_collection("building_docs", embedding_function=None)
    d = col.get(include=["documents", "metadatas"])
    by_b = {}
    for cid, doc, m in zip(d["ids"], d["documents"], d["metadatas"]):
        by_b.setdefault(m["character"], []).append((cid, m["section"], doc))

    meta = pd.read_excel(os.path.join(ROOT, "nlp_project_dataset", "metadata.xlsx"))
    meta = meta[meta.type == "video"]
    rows = []
    for r in meta.itertuples():
        for kind in ("inside", "outside"):
            q, gt = getattr(r, f"query_{kind}"), getattr(r, f"ground_truth_{kind}")
            if pd.isna(q) or pd.isna(gt) or r.character not in by_b:
                continue
            ranked = pick(q, gt, by_b[r.character])
            (c1, s1, sc1, rr1, f1), (c2, s2, sc2, rr2, f2) = ranked[0], ranked[1]
            in_desc = found_frac(gt, r.place_description)
            has_chunk = any((t[4] or 0) > 0 for t in ranked)
            if norm(gt).rstrip(".") in SKIP_STR or norm(gt) in SKIP_STR:
                source = "예아니오"  # 문자열로 근거를 못 찾음. 후보는 재정렬기 순위일 뿐이라 사람이 확인
            elif not has_chunk and in_desc:
                source = "장소설명"
            elif has_chunk:
                source = "청크"
            else:
                source = "문서에없음"  # 정답이 어떤 청크·장소 설명에도 없음 → 문서 보강 대상일 수 있음
            # 확신: 1순위가 정답 문자열을 전부 포함하고 2순위와 점수 차가 크거나, 문자열 일치 없이 재정렬기만 있어도 차가 큼
            sure = source == "청크" and (f1 == 1.0 or (f1 is None and rr1 >= 0.5)) and sc1 - sc2 >= 0.3
            rows.append(dict(video_id=r.video_id, 건물=r.character, 유형=kind, 질문=q, 정답=gt, 근거출처=source,
                             후보1=f"{c1} | {s1}", 후보1_점수=round(sc1, 2), 후보1_일치율=f1,
                             후보2=f"{c2} | {s2}", 후보2_점수=round(sc2, 2), 장소설명에_정답=in_desc,
                             검토필요="N" if sure else "Y", 확정_섹션=""))
    out = pd.DataFrame(rows)
    path = os.path.join(DATA, "eval_evidence_draft.xlsx")
    out.to_excel(path, index=False)
    print(len(out), "rows ->", path)
    print(out.groupby(["유형", "근거출처"]).size().unstack(fill_value=0))
    print(out.groupby("유형").검토필요.value_counts().unstack(fill_value=0))
