import pandas  # noqa: F401  torch보다 먼저 (build_chroma.py 참고)
import os
import streamlit as st
from PIL import Image
import search  # import 시 모델 3개 로드. Streamlit 재실행 때도 모듈은 캐시돼 한 번만 로드됨

FRAMES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frames")

st.set_page_config(page_title="캠퍼스 장소 검색", layout="wide")
st.title("캠퍼스 장소 검색")
query = st.text_input("질의", placeholder="예: 텔레토비동산이 보이는 곳")
top = st.slider("보여줄 영상 수", 1, 10, 5)

if query:
    st.caption(f"건물 필터: {search.building_filter(query)}")
    results = search.search(query, top_videos=top)
    for col, (rank, (vid, z, (isim, frame, _), (tsim, _, sent))) in zip(st.columns(len(results)), enumerate(results, 1)):
        with col:
            st.image(os.path.join(FRAMES, frame), width="stretch")
            st.markdown(f"**{rank}. {vid}**")
            st.caption(f"z={z:.2f} · img={isim:.3f} · txt={tsim:.3f}")
            st.write(sent)

st.divider()
st.subheader("사진으로 건물 찾기")
up = st.file_uploader("사진", type=["jpg", "jpeg", "png", "webp"])
if up:
    img = Image.open(up)
    ranking, hits = search.identify_building(img)
    left, right = st.columns([1, 2])
    left.image(img, width="stretch")
    total = sum(s for _, s in ranking)
    right.markdown(f"### 이 사진은 **{ranking[0][0]}관**으로 보입니다")
    for c, s in ranking:
        right.write(f"{c}: {s / total:.0%}  (상위 {len(hits)}프레임 유사도 합 {s:.2f})")
    for col, (c, sim, vid, frame) in zip(st.columns(5), hits[:5]):
        col.image(os.path.join(FRAMES, frame), width="stretch")
        col.caption(f"{c} · {vid} · {sim:.3f}")
