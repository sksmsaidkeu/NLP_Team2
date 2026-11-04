import pandas  # noqa: F401  torch보다 먼저 (build_chroma.py 참고)
import math
import os
import statistics
import sys
import chromadb
import dino
from PIL import Image
from sentence_transformers import SentenceTransformer

# 질의 속 건물 이름 → character 메타데이터 (표기는 place_description에 실제로 쓰인 것)
BUILDINGS = {"니콜스": "니콜스", "N관": "니콜스", "마리아": "마리아", "밤비노": "밤비노", "BA관": "밤비노",
             "비루투스": "비루투스", "비르투스": "비루투스", "V관": "비루투스",
             "김수환": "김수환", "다솔": "다솔", "중앙도서관": "중앙도서관", "베리타스": "중앙도서관",
             "학생회관": "학생회관", "소피아바라": "학생회관", "소피이바라": "학생회관"}

client = chromadb.HttpClient(host="localhost", port=8000)
img_col = client.get_collection("frames_image", embedding_function=None)
txt_col = client.get_collection("videos_text", embedding_function=None)
img_model = SentenceTransformer(img_col.metadata["text_model"])  # 질의 → 이미지 공간 (다국어 CLIP)
txt_model = SentenceTransformer(txt_col.metadata["text_model"])  # 질의 → 설명 공간 (bge-m3)
dino_col = client.get_collection("frames_dino", embedding_function=None)  # 사진 → 건물 식별용 (CLIP보다 정확: 52%→80%)
def pad_square(img, color=(255, 255, 255)):
    # build_chroma.py와 같은 흰색 letterbox (거기 import하면 DB를 다시 만들어서 복사해 둠)
    img = img.convert("RGB")
    side = max(img.size)
    canvas = Image.new("RGB", (side, side), color)
    canvas.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
    return canvas


TOP_M, TEMP = 3, 0.02  # 건물별 상위 3프레임 평균 유사도를 softmax(T=0.02)로 확률화. LOO 8곳: 합÷√n 80.5% → 82.7%, 확률 0.9↑가 62%에서 96% 정확


def vote(sims):
    # sims: {건물: 유사도 리스트(내림차순)} → [(건물, 확률)] 큰 순. generate.py의 확신 기준(0.9)이 이 확률을 씀
    score = {c: sum(v[:TOP_M]) / len(v[:TOP_M]) for c, v in sims.items()}
    top = max(score.values())
    e = {c: math.exp((x - top) / TEMP) for c, x in score.items()}
    return sorted(((c, x / sum(e.values())) for c, x in e.items()), key=lambda kv: -kv[1])


def identify_building(img, k=10):
    # 전 프레임을 받아 건물별로 묶음 (프레임 1555개라 가벼움). hits는 닮은 장소 표시용 상위 k프레임
    q = dino.embed([pad_square(img)]).tolist()
    r = dino_col.query(query_embeddings=q, n_results=dino_col.count(), include=["metadatas", "distances"])
    hits = [(m["character"], 1 - d, m["video_id"], rid)
            for rid, m, d in zip(r["ids"][0], r["metadatas"][0], r["distances"][0])]
    sims = {}
    for c, sim, _, _ in hits:
        sims.setdefault(c, []).append(sim)
    return vote(sims), hits[:k]


def building_filter(query):
    # ponytail: 문자열 포함 여부만 봄. "니콜스관 옆 건물"처럼 언급≠정답인 질의도 그 건물로 걸러짐
    found = sorted({c for kw, c in BUILDINGS.items() if kw in query.upper()})
    return {"character": {"$in": found}} if found else None


def best_per_video(col, q, where):
    # ponytail: 컬렉션 전체를 점수 매김(수백~천 건). 수만 건이면 n_results 상한을 둘 것
    r = col.query(query_embeddings=q, n_results=col.count(), where=where,
                  include=["metadatas", "distances", "documents"])
    best = {}  # video_id → (유사도, 레코드 id, 문서)
    for rid, meta, dist, doc in zip(r["ids"][0], r["metadatas"][0], r["distances"][0], r["documents"][0]):
        vid, sim = meta["video_id"], 1 - dist
        if vid not in best or sim > best[vid][0]:
            best[vid] = (sim, rid, doc)
    return best


def zsum(*bests):
    # 이미지·설명 점수는 크기가 달라서 질의마다 표준화해 더함. RRF는 설명 1위가 크게 앞서도 그 차이를 버려서 탈락
    score = {}
    for best in bests:
        sims = [b[0] for b in best.values()]
        mean, sd = statistics.mean(sims), statistics.pstdev(sims) or 1  # 후보가 1개거나 점수가 전부 같으면 sd=0
        for vid, b in best.items():
            score[vid] = score.get(vid, 0) + (b[0] - mean) / sd
    return score


def search(query, top_videos=5, where="auto"):
    if where == "auto":
        where = building_filter(query)
    enc = lambda m: m.encode([query], normalize_embeddings=True).tolist()
    img, txt = best_per_video(img_col, enc(img_model), where), best_per_video(txt_col, enc(txt_model), where)
    score = zsum(img, txt)
    top = sorted(score, key=lambda v: -score[v])[:top_videos]
    return [(vid, score[vid], img[vid], txt[vid]) for vid in top]


if __name__ == "__main__":
    assert building_filter("밤의 V관") == {"character": {"$in": ["비루투스"]}}
    assert building_filter("니콜스관과 마리아관 사이") == {"character": {"$in": ["니콜스", "마리아"]}}
    assert building_filter("소피아바라관 2층 식당") == {"character": {"$in": ["학생회관"]}}
    assert building_filter("베리타스관 앞 분수") == {"character": {"$in": ["중앙도서관"]}}
    assert building_filter("다솔관과 중앙도서관") == {"character": {"$in": ["다솔", "중앙도서관"]}}
    assert building_filter("밤에 불 켜진 건물") is None  # '밤'은 밤비노로 오인 안 함
    # 설명에서 크게 앞선 x가 이미지에서 조금 뒤져도 이겨야 함 (RRF였으면 동점)
    img_t, txt_t = {"x": (0.26,), "y": (0.27,), "z": (0.25,)}, {"x": (0.80,), "y": (0.60,), "z": (0.59,)}
    assert max(zsum(img_t, txt_t).items(), key=lambda kv: kv[1])[0] == "x"
    assert zsum({"x": (0.5,)}) == {"x": 0.0}  # sd=0에서 0으로 나누지 않음
    v = vote({"마리아": [0.9], "니콜스": [0.8, 0.8, 0.8, 0.1]})  # 상위 3 평균: 1프레임 0.9가 이김
    assert v[0][0] == "마리아" and abs(sum(p for _, p in v) - 1) < 1e-9
    assert pad_square(Image.new("RGB", (40, 20))).size == (40, 40)
    # 인자 없이 실행하면 대화형: 모델을 한 번만 로드하고 빈 줄 입력 시 종료
    queries = sys.argv[1:] or iter(lambda: input("\n질의 (빈 줄=종료)> ").strip(), "")
    for query in queries:
        print(f"\n## {query}  (filter: {building_filter(query)})")
        for vid, s, (isim, frame, _), (tsim, _, sent) in search(query):
            print(f"  z={s:5.2f}  img={isim:.3f}  txt={tsim:.3f}  {vid}\n"
                  f"      └ {sent[:70]}\n      └ {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frames', frame)}")
