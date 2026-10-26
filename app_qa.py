import pandas  # noqa: F401  torch보다 먼저 (build_chroma.py 참고)
import streamlit as st
from PIL import Image
import generate

st.title("캠퍼스 건물 질의응답")
mode = st.sidebar.radio("검색 방식", ["filter", "weighted"], help="filter: 식별된 건물 청크만 / weighted: 전 건물 청크에 식별 점수를 가중합")
alpha = st.sidebar.slider("식별 가중치 α (weighted)", 0.0, 2.0, 0.5, 0.1, disabled=mode == "filter")
model = st.sidebar.selectbox("생성 모델", ["claude-haiku-5-5", "claude-sonnet-5-5"])
k = st.sidebar.slider("근거 청크 수", 1, 6, 3)

up = st.file_uploader("건물 사진", type=["jpg", "jpeg", "png"])
question = st.text_input("질문", placeholder="이 건물은 누구를 기리기 위해 만든 건물이야?")
if up:
    img = Image.open(up)
    st.image(img, width=360)
if up and question and st.button("물어보기"):
    with st.spinner("식별·검색·생성 중..."):
        out = generate.ask(img, question, model=model, k=k, mode=mode, alpha=alpha)
    st.subheader("답변")
    st.write(out["answer"])
    st.subheader("건물 식별")
    st.table({"건물": [c for c, _ in out["votes"]], "투표 점수": [round(v, 3) for _, v in out["votes"]]})
    st.subheader("근거 청크")
    for sim, b, sec, doc in out["chunks"]:
        with st.expander(f"{b}관 · {sec} (유사도 {sim:.3f})"):
            st.write(doc)
