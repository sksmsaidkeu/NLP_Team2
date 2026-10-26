import pandas  # noqa: F401  torch보다 먼저 (build_chroma.py 참고)
import sys
import anthropic
from PIL import Image

SYSTEM = ("너는 가톨릭대학교 성심교정 건물 안내자다. 아래 <문서>에 있는 내용으로만 한국어로 답한다. "
          "문서에 근거가 없으면 추측하지 말고 '문서에 없는 내용'이라고 말한다. 공식 설명이 다른 출처와 다르면 공식 설명을 따른다.")

llm = anthropic.Anthropic()


def answer(question, chunks, model="claude-haiku-5-5"):
    # chunks: 문서 텍스트 리스트. 공식 설명 청크가 앞에 오도록 호출 쪽에서 정렬
    docs = "\n\n".join(f"<문서>\n{c}\n</문서>" for c in chunks)
    r = llm.messages.create(model=model, max_tokens=2048, system=SYSTEM,  # 한국어는 토큰을 많이 먹고 thinking도 이 한도를 씀 (500에서 잘렸음)
                            messages=[{"role": "user", "content": f"{docs}\n\n질문: {question}"}])
    text = "".join(b.text for b in r.content if b.type == "text")  # 5.x는 thinking 블록이 먼저 올 수 있음
    return text + "\n\n(⚠ 답변이 길이 제한으로 잘렸습니다)" if r.stop_reason == "max_tokens" else text


def rank_chunks(cands, share, mode, alpha, k):
    # cands: [(유사도, 건물, 섹션, 문서)], share: {건물: 식별 점수 비중 0~1}
    # filter: 1위 건물 청크만 / weighted: 유사도 + alpha*비중으로 전 건물 청크를 다시 매김
    if mode == "filter":
        top = max(share, key=share.get)
        cands = [c for c in cands if c[1] == top]
        return sorted(cands, key=lambda c: -c[0])[:k]
    return sorted(cands, key=lambda c: -(c[0] + alpha * share.get(c[1], 0)))[:k]


def ask(img, question, model="claude-haiku-5-5", k=3, mode="filter", alpha=0.5, building=None):
    # 사진 → 건물 식별 → 청크 검색(mode) → 생성. building을 주면 식별을 건너뜀(식별 오류와 검색/생성 오류를 분리해 볼 때)
    import chromadb
    import search  # 모델 로드가 무거워서 여기서 import
    doc_col = chromadb.HttpClient(host="localhost", port=8000).get_collection("building_docs", embedding_function=None)
    if building is None:
        votes = search.identify_building(img)[0]
    else:
        votes = [(building, 1.0)]
    total = sum(v for _, v in votes)
    share = {c: v / total for c, v in votes}
    q = search.txt_model.encode([question], normalize_embeddings=True).tolist()
    r = doc_col.query(query_embeddings=q, n_results=doc_col.count(), include=["documents", "metadatas", "distances"])
    cands = [(1 - d, m["character"], m["section"], doc)
             for doc, m, d in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])]
    top = rank_chunks(cands, share, mode, alpha, k)
    return {"votes": votes, "chunks": top, "answer": answer(question, [c[3] for c in top], model)}


if __name__ == "__main__":
    c = [(0.5, "A", "s1", "a"), (0.6, "B", "s2", "b"), (0.4, "A", "s3", "c")]
    assert [x[2] for x in rank_chunks(c, {"A": 0.9, "B": 0.1}, "filter", 0, 5)] == ["s1", "s3"]  # B는 걸러짐
    assert rank_chunks(c, {"A": 0.9, "B": 0.1}, "weighted", 0.5, 1)[0][2] == "s1"  # 비중이 유사도 차를 뒤집음
    assert rank_chunks(c, {"A": 0.5, "B": 0.5}, "weighted", 0.5, 1)[0][2] == "s2"
    # 사용: python generate.py 사진경로 "질문" [filter|weighted]
    img, question, *mode = sys.argv[1:]
    out = ask(Image.open(img), question, mode=(mode or ["filter"])[0])
    print(out["votes"], [(c[1], c[2], round(c[0], 3)) for c in out["chunks"]], "", out["answer"], sep="\n")
