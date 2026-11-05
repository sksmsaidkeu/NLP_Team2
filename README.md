# 캠퍼스 건물 질의응답 (멀티모달 RAG)

가톨릭대학교 성심교정 건물 사진과 질문을 받아 답을 돌려주는 자연어처리 팀 프로젝트(Team2)입니다.

```
사진 → 건물 식별(DINOv2) → 질문과 맞는 건물 설명 청크 선택(bge-m3 + 재정렬기) → Claude가 답변 생성
```

## 폴더 구성
| 위치 | 내용 |
|---|---|
| `rag/` | 핵심 모듈 3개: `dino.py`(이미지 임베딩), `search.py`(건물 식별·검색), `generate.py`(청크 선택·재정렬·답변) |
| `apps/` | Streamlit 화면 2개: `app.py`(영상 검색 확인용), `app_qa.py`(질의응답) |
| `build/` | 프레임 추출과 Chroma DB 구축 스크립트 |
| `eval/` | 명세 평가, 보충 평가, 정답 청크 라벨링 스크립트 |
| `data/` | 프로젝트가 쓰는 데이터: `building_docs/`(건물 설명 문서), `frames/`·`frames.csv`(프레임 1555장과 목록), `splits.csv`(명세 평가용 영상 분할), `eval_evidence*.xlsx`(정답 청크 라벨) |
| `results/` | 평가 결과 파일(csv·jsonl·그림) |
| `docs/` | 평가 명세서, 가이드 슬라이드, 참고 논문 PDF(PDF는 저장소에서 제외) |
| `nlp_project_dataset/` | 원본 영상·`metadata.xlsx`(저장소에서 제외) |

### `results/` 파일 설명
| 파일 | 보여주는 것 |
|---|---|
| `eval_spec_retrieval_results.csv` | 명세 ① 건물 검색: seed별 Recall@1/2/4·MRR, 평균·표준편차, 무작위 기준선 |
| `eval_spec_qa_summary.csv` | 명세 ② QA: seed별 Acc(EchoSight)·Acc(Oracle)과 EchoSight가 고른 설명의 건물 정답률 |
| `eval_spec_qa_results.jsonl` | 명세 ② QA의 호출 한 건당 기록(영상·질문·정답·예측·고른 설명·토큰). 요약과 혼동행렬의 원자료 |
| `eval_spec_results_table.csv` | 위 두 요약을 가이드 슬라이드 42 양식으로 합친 결과표(seed 1~4, 평균, 표준편차, 기준선) |
| `eval_spec_confusion.png` | 명세 ② seed별·조건별(EchoSight·Oracle) 2×2 혼동행렬 그림 8개 |
| `eval_spec_qa_summary_v1_prompt.csv`, `eval_spec_qa_results_v1_prompt.jsonl` | 프롬프트를 가이드 문구로 고치기 전의 명세 ② QA 요약과 호출 기록(비교용, 공식 결과 아님) |
| `eval_retrieval_results.csv` | 보충 분석: 정답 청크 질문별로 검색 방식(text-only·filter·echo·재정렬기 등)마다 1·3위 적중 여부 |
| `eval_answers_results.jsonl` | 보충 분석: 답변 정확도 평가의 호출 기록(질문 294개 × 조건 5개: Vanilla·filter·echo+rerank·장소 설명 제거·정답 청크 직접 제공) |

## 파일별 기능

### 데이터 준비
| 파일 | 하는 일 |
|---|---|
| `build/nlp_project_dataset_picturechange.py` | `nlp_project_dataset/metadata.xlsx`의 영상에서 프레임(영상당 10~20장)을 뽑아 `data/frames/`에 저장하고 `data/frames.csv`를 만듭니다. HEIC 사진은 jpg로 변환합니다. |
| `data/building_docs/*.md` | 건물별 설명 문서(8개). 나무위키와 학교 캠퍼스맵을 정리한 초안이며, 아직 검증 전입니다. |

### DB 만들기 (Chroma, `localhost:8000`)
| 파일 | 하는 일 |
|---|---|
| `build/build_chroma.py` | `frames_image`(프레임 CLIP 임베딩)와 `videos_text`(장소 설명 문장 bge-m3 임베딩) 컬렉션을 지우고 새로 만듭니다. |
| `build/build_dino.py` | `frames_dino`(프레임 DINOv2 임베딩, 건물 식별용) 컬렉션만 새로 만듭니다. |
| `build/build_docs.py` | `data/building_docs/*.md`를 `## 섹션` 단위로 쪼개 `building_docs` 컬렉션을 새로 만듭니다. |
| `rag/dino.py` | DINOv2-base로 이미지 임베딩을 계산하는 함수(식별에 쓰임). |

### 검색과 답변
| 파일 | 하는 일 |
|---|---|
| `rag/search.py` | 건물 식별(`identify_building`: 건물별 상위 3프레임 평균 유사도를 softmax로 확률화)과, 텍스트 질의로 영상을 찾는 검색(`search`)을 합니다. 직접 실행하면 자가검사 후 질의를 받습니다. |
| `rag/generate.py` | 전체 파이프라인 `ask(사진, 질문)`. 청크 선택 방식(`echo`/`filter`/`weighted`)과 재정렬기 옵션을 갖고 있으며, 기본은 `echo` + `bge-reranker-v2-m3`입니다. `python generate.py 사진 "질문"`으로도 실행됩니다. |
| `apps/app_qa.py` | 사진을 올리고 질문하면 답변만 보여주는 Streamlit 화면(8002). |
| `apps/app.py` | 텍스트로 영상을 찾고, 사진으로 건물을 식별해 보는 확인용 Streamlit 화면(8001). |

### 평가
| 파일 | 하는 일 |
|---|---|
| `eval/draft_evidence.py` | 질문별 정답 근거 청크 후보를 뽑아 `data/eval_evidence_draft.xlsx`를 만듭니다. |
| `eval/label_evidence.py` | 문서를 직접 읽고 정한 정답 청크 라벨을 초안에 붙여 `data/eval_evidence.xlsx`를 만듭니다. |
| `eval/eval_retrieval.py` | 정답 청크가 1·3위 안에 드는 비율로 검색 방식들을 비교하고 `results/eval_retrieval_results.csv`를 씁니다. 질의 사진의 영상 자신은 DB에서 뺍니다. |
| `eval/make_splits.py` | 낮 영상 129개를 건물별 비율을 맞춰 train/val/test(64:16:20)로 나눕니다(seed 0~3). 결과는 `data/splits.csv`. |
| `eval/eval_spec_retrieval.py` | **명세 평가 ①** 건물 검색 Recall@1/2/4·MRR → `results/eval_spec_retrieval_results.csv` |
| `eval/eval_spec_qa.py` | **명세 평가 ②** 예/아니오 QA 정확도(EchoSight·Oracle)와 혼동행렬 → `results/eval_spec_qa_results.jsonl`(호출 기록, 다시 돌리면 이어서 실행), `results/eval_spec_qa_summary.csv` (`results/*_v1_prompt.*`는 프롬프트 수정 전 결과) |
| `eval/make_spec_report.py` | 명세 결과를 슬라이드 42 양식 표(`results/eval_spec_results_table.csv`)와 seed×조건 혼동행렬 그림(`results/eval_spec_confusion.png`)으로 정리합니다. API 호출 없음. |

## 실행 순서
1. Docker로 Chroma 서버(1.0.0)를 `localhost:8000`에 띄웁니다.
2. `python build/nlp_project_dataset_picturechange.py` → `build/build_chroma.py` → `build/build_dino.py` → `build/build_docs.py` (명령은 프로젝트 루트에서 `python 경로`로 실행하는 것을 기준으로 적었습니다)
3. 환경변수 `ANTHROPIC_API_KEY`를 설정하고 `python -m streamlit run apps/app_qa.py --server.port 8002`

모든 스크립트의 첫 줄이 `import pandas`인 것은 Windows에서 torch보다 먼저 불러오지 않으면 크래시가 나기 때문입니다.
Python 패키지 버전은 `requirements.txt`(개발 환경에서 쓴 버전 그대로, 새 환경에서 설치해 본 적은 없음)에 있습니다. 사용한 모델은 CLIP(clip-ViT-B-32), bge-m3, DINOv2-base, bge-reranker-v2-m3이며 처음 실행할 때 내려받습니다.
원본 영상(`nlp_project_dataset/`)은 저장소에 포함하지 않았습니다.

## 명세 평가 결과 (`docs/명세서.txt` 기준, 공식 결과)

### 설정
- **데이터**: 낮 영상 129개(사진·밤 영상 제외), 8개 건물. 영상 단위로 건물별 비율을 맞춰 train:test = 8:2로 나누고, train을 다시 train:val = 8:2로 나눕니다(영상 수 82/21/26). seed 0·1·2·3 네 번 반복해 평균±표준편차(모집단 표준편차)를 냅니다. 영상 ID가 겹치지 않는지 assert로 확인합니다.
- **모델**: 건물 식별은 DINOv2-base 임베딩, 재정렬기는 `bge-reranker-v2-m3`(추가 학습 없음), 답변은 `claude-haiku-5-5`(명세의 GPT-4 대신). 모든 설정은 기존 값을 그대로 썼고 val로 고른 값이 없습니다. val 영상은 DB에만 들어갑니다.
- 점수는 질의 프레임별로 계산한 뒤 영상 단위로 평균합니다.

### ① 건물 검색 (`eval/eval_spec_retrieval.py`)
DB는 train+val 프레임, 질의는 test 프레임. 건물별로 가장 비슷한 프레임의 유사도(max-pooling)로 8개 건물 순위를 매깁니다.

| seed | R@1 | R@2 | R@4 | MRR |
|---|---|---|---|---|
| 0 | 0.746 | 0.863 | 0.992 | 0.846 |
| 1 | 0.789 | 0.896 | 0.978 | 0.872 |
| 2 | 0.782 | 0.919 | 0.989 | 0.874 |
| 3 | 0.819 | 0.893 | 0.962 | 0.883 |
| **평균±표준편차** | **0.784±0.026** | **0.893±0.020** | **0.980±0.012** | **0.869±0.014** |
| 무작위 | 0.125 | 0.250 | 0.500 | 0.340 |

test 영상은 seed마다 26개(질의 프레임 265~284개)입니다. 이 표의 건물 순위는 위 파이프라인의 "상위 3프레임 평균+softmax"가 아니라 명세대로 max-pooling이므로, 앞의 LOO 수치와 직접 비교할 수 없습니다.

### ② 예/아니오 QA (`eval/eval_spec_qa.py`)
test 영상마다 inside·outside 질문 2개(정답은 예/아니오). context는 `place_description`만 쓰고 `building_docs`는 쓰지 않습니다. 프롬프트는 가이드 슬라이드 35~37의 문구를 그대로 썼습니다(SYSTEM "항상 사용자가 묻는 질문에만 답합니다", 예·아니오 few-shot 각 1개, 각 질문 뒤 "설명 없이 '예' 또는 '아니오'로만 답하세요." + `Short answer is:`). 가이드의 temperature 0은 `claude-haiku-5-5`가 `temperature`를 받지 않아(400 오류) 적용하지 못했습니다.

| 조건 | context |
|---|---|
| **EchoSight** | 질의 사진 → 건물 순위(위와 같은 방식, 자기 영상 제외) → 상위 3개 건물의 영상 설명을 후보로, 건물 확률(softmax, T=0.02)과 재정렬기 점수를 0.5:0.5로 섞어 설명 **1개**를 고름. 자기 영상은 후보에 없음(assert). |
| **Oracle** | 그 질문이 달린 영상 자신의 설명. 검색이 완벽하다고 가정한 **상한** 조건이며, 정답 근거가 설명에 들어 있는 것이 전제입니다. |

영상당 질의 사진은 3장(균등 간격)이고, 점수는 사진×질문을 영상별로 평균한 뒤 영상 평균을 냅니다.

| seed | Acc(EchoSight) | Acc(Oracle) | EchoSight가 고른 설명의 건물이 맞은 비율 |
|---|---|---|---|
| 0 | 0.654 | 0.712 | 0.769 |
| 1 | 0.763 | 0.865 | 0.808 |
| 2 | 0.801 | 0.904 | 0.737 |
| 3 | 0.718 | 0.827 | 0.776 |
| **평균±표준편차** | **0.734±0.055** | **0.827±0.072** | 0.772 |

슬라이드 42 양식으로 R@1·R@2·R@4·MRR과 함께 한 표에 모은 것은 `results/eval_spec_results_table.csv`, seed별·조건별 2×2 혼동행렬 그림은 `results/eval_spec_confusion.png`입니다(`eval/make_spec_report.py`가 API 호출 없이 만듦).

혼동행렬(4개 seed 합산, 행=정답·열=예측. EchoSight는 사진×질문 단위, Oracle은 질문 단위이며 같은 영상이 여러 seed의 test에 들어가면 중복해서 셉니다):

| EchoSight | 예 | 아니오 |
|---|---|---|
| 정답 예 | 206 | 100 |
| 정답 아니오 | 66 | 252 |

| Oracle | 예 | 아니오 |
|---|---|---|
| 정답 예 | 70 | 32 |
| 정답 아니오 | 4 | 102 |

첫 단어가 예/아니오가 아닌 답("기타")은 오답으로 세는데, 이번 실행에는 없었습니다(처음 프롬프트 실행에는 Oracle에서 1건).

### 해석과 한계
- **EchoSight는 Oracle보다 평균 약 9%p 낮습니다**(seed마다 6~11%p, 같은 방향). 다만 seed마다 영상 26개·질문 52개라 seed 간 표준편차가 5~7%p로 크고, 두 조건의 차이를 검정하지는 않았습니다.
- **Oracle도 100%가 아닙니다.** Oracle 오답을 영상·질문 단위로 중복 없이 세면 164문항 중 30개(18%)입니다. 이 30개를 설명과 비교해 직접 읽고 분류하면 25개는 설명에 정답 근거가 없었고(시설명·호수·상호 등), 3개는 근거가 있는데 모델이 틀렸거나 추론을 못 했으며, 2개는 질문·정답이 모호했습니다. 이 분류는 사람이 읽은 판단이라 경계 사례는 달리 볼 수 있습니다. 근거가 없을 때 모델이 "아니오"로 답하는 경향이 혼동행렬에서 보입니다(정답 "예" 중 Oracle의 31%, EchoSight의 33%가 "아니오").
- **프롬프트 수정 전후 비교**: 처음 프롬프트(SYSTEM·지시문 위치가 달랐음) 결과는 `results/eval_spec_qa_summary_v1_prompt.csv`에 남겨 두었고 평균은 EchoSight 0.720·Oracle 0.803이었습니다. 두 실행에서 Oracle의 같은 문항 164개 중 오답이 33개 → 30개로 바뀌었고, 두 실행에 모두 오답인 문항은 27개입니다(처음 33개 중 6개가 맞게 바뀌고 3개가 새로 틀림). 설명에 근거가 없는 문항에서 모델이 찍은 답이 바뀐 경우가 있어, 프롬프트 차이와 호출 무작위성을 구분하지는 못합니다. 오답 분류는 이 30개를 기준으로 다시 했습니다.
- **`place_description`을 어떻게 썼는지에 따라 해석이 달라집니다.** Oracle은 설명에 정답 근거가 있다는 전제의 상한입니다. 질문을 보고 설명에 내용을 넣었다면 Oracle(과, 같은 건물의 다른 영상 설명을 쓰는 EchoSight)이 실제 서비스보다 높게 나올 수 있는데, 작성 방식을 확인하지 못했습니다.
- 질문·정답이 모호한 문항(다솔 IMG_7277, 니콜스 IMG_7123, 처음 실행에서 틀렸던 밤비노 5403)은 팀의 확인이 필요합니다.
- 마리아관은 낮 영상이 6개뿐이라 test에 1개만 들어갑니다. 건물별 점수는 계산하지 않았고, 계산하더라도 마리아관은 영상 1개에 좌우됩니다.
- 사진 3장, 설명 1개, 건물 확률/재정렬기 가중치 0.5는 임의로 정한 값이며 val로 비교하지 않았습니다.
- temperature를 0으로 고정하지 못했고 같은 프롬프트로 반복 실행한 적도 없어서, 호출 무작위성에서 오는 오차는 재지 못했습니다. 위 표는 한 번 실행한 값입니다.
- 실행: `python eval/make_splits.py`(이미 `data/splits.csv` 있음) → `python eval/eval_spec_retrieval.py` → `python eval/eval_spec_qa.py [seed ...]` → `python eval/make_spec_report.py`. 앞의 두 평가는 Chroma가 켜져 있어야 하고, QA는 `ANTHROPIC_API_KEY`가 필요합니다(전체 약 입력 57만·출력 7만 토큰). 다시 돌리면 `results/eval_spec_qa_results.jsonl`에 이미 있는 호출은 건너뜁니다.

## 보충 분석 (명세 이전 평가)
아래는 질문이 예/아니오로 바뀌기 전 질문 126개를 기준으로 한 이전 평가로, 현재 질문·라벨과 수치가 다릅니다(최신 라벨 기준 재측정 결과는 `results/eval_retrieval_results.csv`).
- **건물 식별**: 영상 하나를 빼고 맞히는 방식(LOO)으로 프레임 기준 82.7%(8개 건물). 같은 데이터로 방식을 골랐고 새 장소에서의 정확도는 아닙니다.
- **검색**: 정답 청크가 있는 126개 질문에서 1위 적중률 `filter` 54.5% → `echo`+재정렬기 68.8%. 표본이 작아 오차가 약 ±4%p이고, 정답 청크는 문서를 읽고 직접 붙인 라벨입니다.
- 청크 정답이 있는 행의 답변 정확도(`eval/eval_answers.py`, `results/eval_answers_results.jsonl`)도 따로 측정했지만 이 README에는 정리하지 않았습니다.

## 알려진 한계
- 밤비노관·비르투스관은 질문·정답이 거의 비어 있어 평가에서 사실상 빠져 있습니다.
- 문서에 정답이 없는 질문이 일부 있습니다(`data/eval_evidence.xlsx`에서 `없음`으로 표시).
- `apps/app_qa.py`는 답변만 보여주며 식별된 건물이나 근거 청크는 표시하지 않습니다.
