import pandas  # noqa: F401  torch보다 먼저 (build_chroma.py 참고)
import os
import sys
import streamlit as st
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 프로젝트 루트
sys.path.insert(0, os.path.join(ROOT, "rag"))  # dino · search · generate 모듈
import generate

CONF_WARN = 0.8  # 영상 78개 평가에서 식별 오답은 모두 1위 비중이 0.8 미만. 정답 일부도 미만이라 오답 확정이 아닌 주의 표시

st.title("캠퍼스 건물 질의응답")
mode = st.sidebar.radio("검색 방식", ["echo", "filter", "weighted"], help="filter: 식별된 건물 청크만 / weighted: 전 건물 청크에 식별 점수를 가중합 / echo: 상위 3개 건물 후보, 식별 확률과 질문 유사도를 0.5:0.5로 섞어 1청크만")
alpha = st.sidebar.slider("식별 가중치 α (weighted)", 0.0, 0.3, 0.1, 0.05, disabled=mode != "weighted")  # 0.25 이상이면 filter와 같은 결과 (질문 유사도 차가 작아 비중이 순위를 정함)
model = st.sidebar.selectbox("생성 모델", ["claude-haiku-5-5", "claude-sonnet-5-5"])
rerank = st.sidebar.checkbox("재정렬기 (bge-reranker)", value=True, disabled=mode != "echo", help="echo 전용. 질문↔청크 점수를 크로스 인코더로")
loc_b = st.sidebar.checkbox("B: 재정렬기 질의에 닮은 장소 설명", disabled=mode != "echo" or not rerank)
loc_a = st.sidebar.checkbox("A: 위치 적합도 점수", disabled=mode != "echo", help="닮은 영상 3개의 장소 설명과 청크의 유사도를 더함")
k = st.sidebar.slider("근거 청크 수", 1, 6, 3, disabled=mode == "echo")

up = st.file_uploader("건물 사진", type=["jpg", "jpeg", "png"])
question = st.text_input("질문", placeholder="예/아니오로 답할 수 있게: 이 건물에 엘리베이터가 있어?")
if up:
    img = Image.open(up)
    st.image(img, width=360)
if up and question and st.button("물어보기"):
    with st.spinner("식별·검색·생성 중..."):
        out = generate.ask(img, question, model=model, k=k, mode=mode, alpha=alpha, rerank=rerank, loc_b=loc_b, loc_a=loc_a)
    st.subheader("답변")
    st.write(out)
#     st.subheader("건물 식별")
#     st.table({"건물": [c for c, _ in out["votes"]], "투표 점수": [round(v, 3) for _, v in out["votes"]]})
#     st.subheader("닮은 장소")
#     for b, sim, desc in out["places"]:
#         with st.expander(f"{b}관 (유사도 {sim:.3f})"):
#             st.write(desc)
#     st.subheader("근거 청크")
#     for sim, b, sec, doc in out["chunks"]:
#         with st.expander(f"{b}관 · {sec} (유사도 {sim:.3f})"):
#             st.write(doc)
