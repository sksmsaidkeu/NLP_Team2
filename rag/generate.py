import pandas  # noqa: F401  torch보다 먼저 (build_chroma.py 참고)
import re
import sys
import anthropic
import numpy as np
from PIL import Image

SYSTEM = ("너는 가톨릭대학교 성심교정 건물 안내자다. 질문은 '예/아니오'로 답할 수 있는 것만 받는다. 아래 <문서>와 <장소>에 있는 내용으로만 답한다. "
          "출력은 정확히 두 줄이다. 첫 줄: '예', '아니오', '문서에 없음' 중 하나만(다른 말 금지). 둘째 줄: '이유: ' 뒤에 근거를 문서 내용으로 1~2문장. "
          "문서와 장소 어디에도 근거가 없으면 추측하지 말고 첫 줄을 '문서에 없음'으로 한다. 공식 설명이 다른 출처와 다르면 공식 설명을 따른다. "
          "예/아니오로 답할 수 없는 질문이면 첫 줄을 '질문 형식 오류'로 하고 둘째 줄에 '이유: 예/아니오로 답할 수 있게 질문해 주세요'라고 쓴다. "
          "묻지 않은 정보나 <장소>에 대한 설명은 덧붙이지 않는다. ")
# 건물 식별 확신도(1위 비중)에 따라 <장소>를 다르게 안내. 0.9: LOO에서 비중 0.85~0.95 구간 정확도 84%, 0.95 이상 98%
CONF_HIGH = 0.9
PLACE_SURE = "<장소>는 질문 사진의 촬영 장소로 추정되는 곳의 설명이다. 사진 속 위치나 주변을 묻는 질문에는 이를 사진의 위치로 보고 단서 없이 답한다."
PLACE_UNSURE = "<장소>는 질문 사진과 닮은 다른 사진의 촬영 장소 설명이다. 같은 장소라고 단정하지 말고, 사진 속 위치나 주변을 묻는 질문에만 참고한다."

VERDICTS = ("예", "아니오", "문서에 없음", "질문 형식 오류")


def verdict(text):
    # 답변 첫 줄 → 판정. 형식이 어긋나면 None (채점은 평가 스크립트에서 이 값으로)
    first = text.strip().splitlines()[0].strip() if text.strip() else ""
    return first if first in VERDICTS else None


llm = anthropic.Anthropic()


def answer(question, chunks, model="claude-haiku-5-5", places=(), confident=False):
    # chunks: 문서 텍스트 리스트. 공식 설명 청크가 앞에 오도록 호출 쪽에서 정렬
    # places: near_places 결과. 사진과 닮은 프레임의 장소 설명 (Img2Loc처럼 이웃의 메타데이터를 프롬프트에 넣음)
    docs = "\n\n".join(f"<문서>\n{c}\n</문서>" for c in chunks)
    docs += "".join(f'\n\n<장소 건물="{b}" 유사도="{sim:.2f}">\n{desc}\n</장소>' for b, sim, desc in places)
    r = llm.messages.create(model=model, max_tokens=2048, system=SYSTEM + (PLACE_SURE if confident else PLACE_UNSURE),  # 한국어는 토큰을 많이 먹고 thinking도 이 한도를 씀 (500에서 잘렸음)
                            messages=[{"role": "user", "content": f"{docs}\n\n질문: {question}"}])
    text = "".join(b.text for b in r.content if b.type == "text")  # 5.x는 thinking 블록이 먼저 올 수 있음
    return text + "\n\n(⚠ 답변이 길이 제한으로 잘렸습니다)" if r.stop_reason == "max_tokens" else text


def near_places(hits, col, building=None, n=3):
    # hits: identify_building이 준 [(건물, 유사도, video_id, 프레임 id)] 유사도 순. 같은 영상 프레임은 설명이 같아 영상당 1개만
    # building을 주면 그 건물만 (filter 모드와 맞춤). 반환: [(건물, 유사도, 장소 설명)]
    seen, picks = set(), []
    for b, sim, vid, fid in hits:
        if (building and b != building) or vid in seen:
            continue
        seen.add(vid)
        picks.append((b, sim, fid))
        if len(picks) == n:
            break
    if not picks:
        return []
    r = col.get(ids=[p[2] for p in picks], include=["documents"])  # get은 요청 순서를 보장하지 않아 id로 맞춤
    desc = dict(zip(r["ids"], r["documents"]))
    return [(b, sim, desc[fid]) for b, sim, fid in picks]


# --- echo 모드 선택 기능 (켜고 끄며 하나씩 효과를 잼). 기본은 모두 꺼짐 ---
LOC_K, LOC_W = 3, 0.25  # 위치 적합도에 쓸 닮은 영상 수, 질문 점수 대비 가중치 (평가셋으로 다시 잡을 값)
_rr = None


def rerank_scores(query, docs):
    # 재정렬기: (질문, 청크) 쌍마다 관련도 0~1. 처음 쓸 때 로드
    global _rr
    if _rr is None:
        import torch
        from sentence_transformers import CrossEncoder
        _rr = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda" if torch.cuda.is_available() else "cpu",
                           model_kwargs={"torch_dtype": torch.float16})
    return [float(x) for x in _rr.predict([(query, d) for d in docs])]


def loc_fit(chunk_embs, place_sent_embs, weights):
    # 청크별 위치 적합도: 닮은 영상 설명 문장들과의 최대 유사도를 영상 유사도로 가중평균. 임베딩은 단위벡터
    per_place = [(np.asarray(chunk_embs) @ np.asarray(e).T).max(1) for e in place_sent_embs]
    return sum(w * p for w, p in zip(weights, per_place)) / sum(weights)


def minmax(x):
    x = np.asarray(x, dtype=float)
    return (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else np.zeros_like(x)


ECHO_TOP_B, ECHO_ALPHA = 3, 0.5  # EchoSight는 사진 점수 0.5 + 재정렬 점수 0.5. 우리 평가셋으로 다시 잡을 값


def tune_cands(cands, share, question, loc, rerank, loc_b, loc_a, enc):
    # cands: [(질문 유사도, 건물, 섹션, 문서, 임베딩)]. 후보 건물(상위 ECHO_TOP_B)로 줄인 뒤 첫 항목(점수)을 바꿔 돌려줌
    # rerank: 점수를 재정렬기로 / loc_b: 재정렬기 질의에 1위 닮은 영상 설명을 붙임(rerank 필요) / loc_a: 위치 적합도를 LOC_W만큼 더함
    keep = sorted(share, key=share.get, reverse=True)[:ECHO_TOP_B]
    cands = [c for c in cands if c[1] in keep]
    score = minmax([c[0] for c in cands])
    if rerank:
        q = question + (" [사진 장소] " + loc[0][2] if loc_b and loc else "")
        score = minmax(rerank_scores(q, [c[3] for c in cands]))
    if loc_a and loc:
        sents = [[t for t in re.split(r"(?<=\.)\s+", desc.strip()) if t] for _, _, desc in loc]
        embs = [enc.encode(t, normalize_embeddings=True) for t in sents]
        score = score + LOC_W * minmax(loc_fit([c[4] for c in cands], embs, [sim for _, sim, _ in loc]))
    return [(float(x),) + c[1:] for x, c in zip(score, cands)]


def rank_chunks(cands, share, mode, alpha, k):
    # cands: [(유사도, 건물, 섹션, 문서)], share: {건물: 식별 점수 비중 0~1}
    # filter: 1위 건물 청크만 / weighted: 유사도 + alpha*비중으로 전 건물 청크를 다시 매김
    # echo: EchoSight식. 비중 상위 ECHO_TOP_B개 건물의 청크만 후보로 두고, 비중과 질문 유사도(후보 안에서 0~1로 늘림)를 ECHO_ALPHA로 섞음
    if mode == "filter":
        top = max(share, key=share.get)
        cands = [c for c in cands if c[1] == top]
        return sorted(cands, key=lambda c: -c[0])[:k]
    if mode == "echo":
        keep = sorted(share, key=share.get, reverse=True)[:ECHO_TOP_B]
        cands = [c for c in cands if c[1] in keep]
        lo, hi = min(c[0] for c in cands), max(c[0] for c in cands)
        norm = lambda x: (x - lo) / (hi - lo) if hi > lo else 0.0
        return sorted(cands, key=lambda c: -(ECHO_ALPHA * share[c[1]] + (1 - ECHO_ALPHA) * norm(c[0])))[:k]
    return sorted(cands, key=lambda c: -(c[0] + alpha * share.get(c[1], 0)))[:k]


def ask(img, question, model="claude-haiku-5-5", k=3, mode="echo", alpha=0.5, building=None, rerank=True, loc_b=False, loc_a=False):
    # 사진 → 건물 식별 → 청크 검색(mode) → 생성. building을 주면 식별을 건너뜀(식별 오류와 검색/생성 오류를 분리해 볼 때)
    # 반환: 답변 문자열(answer)만. 식별 점수·근거 청크·닮은 장소는 돌려주지 않음
    import chromadb
    import search  # 모델 로드가 무거워서 여기서 import
    doc_col = chromadb.HttpClient(host="localhost", port=8000).get_collection("building_docs", embedding_function=None)
    if building is None:
        votes, hits = search.identify_building(img)
    else:
        votes, hits = [(building, 1.0)], []
    total = sum(v for _, v in votes)
    share = {c: v / total for c, v in votes}
    q = search.txt_model.encode([question], normalize_embeddings=True).tolist()
    r = doc_col.query(query_embeddings=q, n_results=doc_col.count(), include=["documents", "metadatas", "distances", "embeddings"])
    cands = [(1 - d, m["character"], m["section"], doc, e)
             for doc, m, d, e in zip(r["documents"][0], r["metadatas"][0], r["distances"][0], r["embeddings"][0])]
    if mode == "echo" and (rerank or loc_a):
        loc = near_places(hits, search.img_col, max(share, key=share.get), LOC_K)  # 1위 건물의 닮은 영상 설명 (식별 생략 시 빈 목록)
        cands = tune_cands(cands, share, question, loc, rerank, loc_b, loc_a, search.txt_model)
    top = rank_chunks(cands, share, mode, alpha, 1 if mode == "echo" else k)  # echo는 1섹션만
    near = {"filter": max(share, key=share.get), "echo": top[0][1]}.get(mode)  # echo: 고른 청크의 건물만, weighted: 전체
    places = near_places(hits, search.img_col, near)
    return answer(question, [c[3] for c in top], model, places, confident=max(share.values()) >= CONF_HIGH)


if __name__ == "__main__":
    assert verdict("예\n이유: 있다") == "예" and verdict(" 문서에 없음 \n이유: x") == "문서에 없음" and verdict("네, 있습니다") is None and verdict("") is None
    c = [(0.5, "A", "s1", "a"), (0.6, "B", "s2", "b"), (0.4, "A", "s3", "c")]
    assert [x[2] for x in rank_chunks(c, {"A": 0.9, "B": 0.1}, "filter", 0, 5)] == ["s1", "s3"]  # B는 걸러짐
    assert rank_chunks(c, {"A": 0.9, "B": 0.1}, "weighted", 0.5, 1)[0][2] == "s1"  # 비중이 유사도 차를 뒤집음
    assert rank_chunks(c, {"A": 0.5, "B": 0.5}, "weighted", 0.5, 1)[0][2] == "s2"
    e = [(0.50, "A", "s1", "a"), (0.52, "B", "s2", "b"), (0.30, "C", "s3", "c"), (0.90, "D", "s4", "d")]
    assert rank_chunks(e, {"A": .5, "B": .3, "C": .15, "D": .05}, "echo", 0, 1)[0][2] == "s1"  # D는 상위 3개 건물이 아니라 후보 밖
    assert rank_chunks(e, {"A": .4, "B": .4, "C": .2, "D": 0}, "echo", 0, 1)[0][2] == "s2"  # 비중 같으면 질문 유사도가 정함
    # 위치 적합도: 설명 문장 중 하나라도 비슷하면 높음, 영상 가중평균
    ce = [[1, 0], [0, 1]]
    assert list(loc_fit(ce, [[[1, 0]], [[0, 1]]], [3, 1])) == [0.75, 0.25]
    assert list(minmax([2, 2])) == [0, 0] and list(minmax([1, 3])) == [0, 1]
    class FakeCol:  # get이 요청 순서를 뒤집어 돌려줘도 id로 맞춰지는지
        def get(self, ids, include):
            return {"ids": ids[::-1], "documents": [f"d{i}" for i in ids[::-1]]}
    h = [("A", .9, "v1", "f1"), ("A", .8, "v1", "f2"), ("B", .7, "v2", "f3"), ("A", .6, "v3", "f4")]
    assert near_places(h, FakeCol()) == [("A", .9, "df1"), ("B", .7, "df3"), ("A", .6, "df4")]  # v1 중복 제거
    assert near_places(h, FakeCol(), building="A", n=2) == [("A", .9, "df1"), ("A", .6, "df4")]  # B 제외
    assert near_places([], FakeCol()) == []
    # 사용: python rag/generate.py 사진경로 "질문" [echo|filter|weighted]  (기본 echo+재정렬기)
    img, question, *mode = sys.argv[1:]
    print(ask(Image.open(img), question, mode=(mode or ["echo"])[0]))
