# 캠퍼스 건물 질의응답 (멀티모달 RAG)

가톨릭대학교 성심교정 건물 사진과 질문을 받아 답을 돌려주는 자연어처리 팀 프로젝트(Team2)입니다.

```
사진 → 건물 식별(DINOv2) → 질문과 맞는 건물 설명 청크 선택(bge-m3 + 재정렬기) → Claude가 답변 생성
```

## 파일별 기능

### 데이터 준비
| 파일 | 하는 일 |
|---|---|
| `nlp_project_dataset_picturechange.py` | `nlp_project_dataset/metadata.xlsx`의 영상에서 프레임(영상당 10~20장)을 뽑아 `frames/`에 저장하고 `frames.csv`를 만듭니다. HEIC 사진은 jpg로 변환합니다. |
| `building_docs/*.md` | 건물별 설명 문서(8개). 나무위키와 학교 캠퍼스맵을 정리한 초안이며, 아직 검증 전입니다. |

### DB 만들기 (Chroma, `localhost:8000`)
| 파일 | 하는 일 |
|---|---|
| `build_chroma.py` | `frames_image`(프레임 CLIP 임베딩)와 `videos_text`(장소 설명 문장 bge-m3 임베딩) 컬렉션을 지우고 새로 만듭니다. |
| `build_dino.py` | `frames_dino`(프레임 DINOv2 임베딩, 건물 식별용) 컬렉션만 새로 만듭니다. |
| `build_docs.py` | `building_docs/*.md`를 `## 섹션` 단위로 쪼개 `building_docs` 컬렉션을 새로 만듭니다. |
| `dino.py` | DINOv2-base로 이미지 임베딩을 계산하는 함수(식별에 쓰임). |

### 검색과 답변
| 파일 | 하는 일 |
|---|---|
| `search.py` | 건물 식별(`identify_building`: 건물별 상위 3프레임 평균 유사도를 softmax로 확률화)과, 텍스트 질의로 영상을 찾는 검색(`search`)을 합니다. 직접 실행하면 자가검사 후 질의를 받습니다. |
| `generate.py` | 전체 파이프라인 `ask(사진, 질문)`. 청크 선택 방식(`echo`/`filter`/`weighted`)과 재정렬기 옵션을 갖고 있으며, 기본은 `echo` + `bge-reranker-v2-m3`입니다. `python generate.py 사진 "질문"`으로도 실행됩니다. |
| `app_qa.py` | 사진을 올리고 질문하면 답변만 보여주는 Streamlit 화면(8002). |
| `app.py` | 텍스트로 영상을 찾고, 사진으로 건물을 식별해 보는 확인용 Streamlit 화면(8001). |

### 평가
| 파일 | 하는 일 |
|---|---|
| `draft_evidence.py` | 질문별 정답 근거 청크 후보를 뽑아 `eval_evidence_draft.xlsx`를 만듭니다. |
| `label_evidence.py` | 문서를 직접 읽고 정한 정답 청크 라벨을 초안에 붙여 `eval_evidence.xlsx`를 만듭니다. |
| `eval_retrieval.py` | 정답 청크가 1·3위 안에 드는 비율로 검색 방식들을 비교하고 `eval_retrieval_results.csv`를 씁니다. 질의 사진의 영상 자신은 DB에서 뺍니다. |

## 실행 순서
1. Docker로 Chroma 서버(1.0.0)를 `localhost:8000`에 띄웁니다.
2. `python nlp_project_dataset_picturechange.py` → `build_chroma.py` → `build_dino.py` → `build_docs.py`
3. 환경변수 `ANTHROPIC_API_KEY`를 설정하고 `python -m streamlit run app_qa.py --server.port 8002`

모든 스크립트의 첫 줄이 `import pandas`인 것은 Windows에서 torch보다 먼저 불러오지 않으면 크래시가 나기 때문입니다.
`requirements.txt`는 아직 없습니다. 사용한 모델은 CLIP(clip-ViT-B-32), bge-m3, DINOv2-base, bge-reranker-v2-m3이며 처음 실행할 때 내려받습니다.
원본 영상(`nlp_project_dataset/`)은 저장소에 포함하지 않았습니다.

## 지금까지 측정한 것
- **건물 식별**: 영상 하나를 빼고 맞히는 방식(LOO)으로 프레임 기준 82.7%(8개 건물). 같은 데이터로 방식을 골랐고 새 장소에서의 정확도는 아닙니다.
- **검색**: 정답 청크가 있는 126개 질문에서 1위 적중률 `filter` 54.5% → `echo`+재정렬기 68.8%. 표본이 작아 오차가 약 ±4%p이고, 정답 청크는 문서를 읽고 직접 붙인 라벨입니다.
- **답변 정확도는 아직 측정하지 않았습니다.**

## 알려진 한계
- 밤비노관·비르투스관은 질문·정답이 거의 비어 있어 평가에서 사실상 빠져 있습니다.
- 문서에 정답이 없는 질문이 일부 있습니다(`eval_evidence.xlsx`에서 `없음`으로 표시).
- `app_qa.py`는 답변만 보여주며 식별된 건물이나 근거 청크는 표시하지 않습니다.
