# 캠퍼스 건물 질의응답 — 평가 코드 (Team2)

가톨릭대학교 성심교정 건물 사진과 질문으로 답을 찾는 멀티모달 RAG 프로젝트의 **명세 평가 코드**입니다(`docs/명세서.txt`, 가이드 슬라이드 기준).
실제 동작하는 데모(건물 식별·검색·답변 앱, DB 구축 스크립트)와 이전 보충 분석 코드는 이 저장소에 없고 별도로 보관합니다.

이 저장소의 평가는 Docker·Chroma·DINOv2 모델·원본 영상 없이 `data/`의 파일만으로 재현됩니다. 재정렬기 모델(bge-reranker-v2-m3)만 처음 실행할 때 내려받으며, 호출을 새로 만들 때는 GPU가 없으면 느립니다.

## 실행 방법
```
pip install -r requirements.txt       # 결과 확인에 필요한 패키지 4개 (QA를 새로 호출할 때만 requirements-qa.txt도 설치)
python eval/evaluate_retrieval.py     # 명세 ① 건물 검색 → results/retrieval_results.csv
python eval/evaluate_qa.py            # 명세 ② 예/아니오 QA → results/qa_summary.csv  (환경변수 ANTHROPIC_API_KEY 필요)
python eval/make_report.py            # 슬라이드 42 양식 표와 혼동행렬 그림 (API 호출 없음)
```
- `evaluate_qa.py`는 `results/qa_calls.jsonl`에 이미 있는 호출을 건너뜁니다. 지금 저장소 상태에서 그대로 실행하면 API를 호출하지 않고(API 키·torch·anthropic 불필요) 저장된 기록으로 요약만 다시 계산하며, 몇 초 걸립니다. 처음부터 다시 호출하려면 `pip install -r requirements-qa.txt`를 하고 이 파일을 지우세요(입력 약 57만·출력 약 7만 토큰, `ANTHROPIC_API_KEY` 필요).
- `evaluate_retrieval.py`는 API가 필요 없습니다.
- 분할은 `python eval/make_splits.py`로 다시 만들 수 있고 결과(`data/splits.csv`)는 같습니다.
- 스크립트 첫머리에서 pandas를 torch보다 먼저 불러오는 것은 의도입니다. Windows에서 순서가 바뀌면 프로세스가 비정상 종료되는 문제가 있었습니다.

## 코드 읽는 순서
1. `eval/data_loading.py` — 파일 읽기 함수와 `building_max_scores`(건물마다 가장 비슷한 프레임의 유사도). 두 평가가 공통으로 씁니다.
2. `eval/evaluate_retrieval.py` — `evaluate_seed`가 seed 하나의 점수를 계산하고, `main`이 seed 평균·표준편차를 냅니다.
3. `eval/evaluate_qa.py` — 위에서 아래로 읽으면 됩니다.
   `build_prompt`(프롬프트) → `building_probabilities`(사진 → 건물 확률) → `pick_description`(후보 중 설명 1개 선택) → `build_jobs`(oracle·echo 호출 목록) → `call_model`(API 한 건) → `summarize`(정확도·혼동행렬) → `main`.
4. `eval/make_report.py` — 결과 파일을 표와 그림으로 정리합니다.

## 폴더 구성
| 위치 | 내용 |
|---|---|
| `eval/data_loading.py` | `data/`를 읽는 공통 함수와 건물별 최대 유사도 계산 |
| `eval/reranker.py` | bge-reranker-v2-m3로 질문–설명 관련도를 계산하는 `rerank_scores`와 0~1 정규화 `minmax` |
| `eval/make_splits.py` | 영상 129개를 건물 비율을 유지해 train/val/test로 나눔(seed 0~3) |
| `eval/evaluate_retrieval.py` | 명세 ① 건물 검색 평가 |
| `eval/evaluate_qa.py` | 명세 ② QA 평가(EchoSight·Oracle 조건) |
| `eval/make_report.py` | 결과 표(`results_table.csv`)와 혼동행렬 그림(`confusion_matrices.png`) 생성 |
| `data/` | 평가 입력 파일(아래 표) |
| `results/` | 평가 결과 파일(아래 표) |
| `docs/` | 명세서, 가이드 슬라이드, 참고 논문 PDF(PDF는 저장소에서 제외) |

### `data/` 파일
| 파일 | 내용 |
|---|---|
| `splits.csv` | 영상 129개의 건물(`building`)과 seed0~3별 역할(train/val/test). 영상 ID는 `건물/…/번호.mp4` 형식(예: `김수환/5547.mp4`) |
| `frame_embeddings.npz` | 영상에서 뽑은 프레임 1364장의 DINOv2-base 임베딩(`embeddings`, 1364×1536, 단위벡터), `video_id`, `frame_idx` |
| `video_descriptions.csv` | 영상별 건물, `place_description`, inside·outside 질문과 정답(예/아니오) |

두 파일(`frame_embeddings.npz`, `video_descriptions.csv`)은 원본 영상·`metadata.xlsx`·데모의 DB에서 한 번 내보낸 것입니다. 이 저장소에는 내보내는 코드가 없습니다.

### `results/` 파일
| 파일 | 보여주는 것 |
|---|---|
| `retrieval_results.csv` | 명세 ① seed별 Recall@1/2/4·MRR, 평균, 표준편차, 무작위 기준선 |
| `qa_summary.csv` | 명세 ② seed별 Acc(EchoSight)·Acc(Oracle)과 EchoSight가 고른 설명의 건물 정답률 |
| `qa_calls.jsonl` | 명세 ② 호출 한 건당 기록(영상·질문·정답·예측·고른 설명·토큰). 요약과 혼동행렬의 원자료 |
| `results_table.csv` | 위 두 요약을 슬라이드 42 양식으로 합친 표(Seed 1~4, 평균, 표준편차, 기준선) |
| `confusion_matrices.png` | seed별·조건별(EchoSight·Oracle) 2×2 혼동행렬 그림 8개 |
| `qa_summary_v1_prompt.csv`, `qa_calls_v1_prompt.jsonl` | 프롬프트를 가이드 문구로 고치기 전의 QA 결과(비교용, 공식 결과 아님) |

## 평가 설정 (`docs/명세서.txt` 기준)
- **데이터**: 낮 영상 129개(사진·밤 영상 제외), 8개 건물. 영상 단위로 건물별 비율을 맞춰 train:test = 8:2로 나누고, train을 다시 train:val = 8:2로 나눕니다(영상 수 82/21/26). seed 0·1·2·3 네 번 반복해 평균±표준편차(모집단 표준편차)를 냅니다. 영상 ID가 겹치지 않는지 assert로 확인합니다.
- **모델**: 건물 식별은 DINOv2-base 임베딩, 재정렬기는 `bge-reranker-v2-m3`(추가 학습 없음), 답변은 `claude-haiku-5-5`(명세의 GPT-4 대신). 모든 설정은 기존 값을 그대로 썼고 val로 고른 값이 없습니다. val 영상은 DB에만 들어갑니다.
- 점수는 질의 프레임별로 계산한 뒤 영상 단위로 평균합니다.

## 결과 (공식)

### ① 건물 검색
DB는 train+val 프레임, 질의는 test 프레임. 건물별로 가장 비슷한 프레임의 유사도(max-pooling)로 8개 건물 순위를 매깁니다.

| seed | R@1 | R@2 | R@4 | MRR |
|---|---|---|---|---|
| 0 | 0.746 | 0.863 | 0.992 | 0.846 |
| 1 | 0.789 | 0.896 | 0.978 | 0.872 |
| 2 | 0.782 | 0.919 | 0.989 | 0.874 |
| 3 | 0.819 | 0.893 | 0.962 | 0.883 |
| **평균±표준편차** | **0.784±0.026** | **0.893±0.020** | **0.980±0.012** | **0.869±0.014** |
| 무작위 | 0.125 | 0.250 | 0.500 | 0.340 |

test 영상은 seed마다 26개(질의 프레임 265~284개)입니다.

### ② 예/아니오 QA
test 영상마다 inside·outside 질문 2개(정답은 예/아니오). context는 `place_description`만 씁니다. 프롬프트는 가이드 슬라이드 35~37 문구 그대로입니다(SYSTEM "항상 사용자가 묻는 질문에만 답합니다", 예·아니오 few-shot 각 1개, 각 질문 뒤 "설명 없이 '예' 또는 '아니오'로만 답하세요." + `Short answer is:`). 가이드의 temperature 0은 `claude-haiku-5-5`가 `temperature`를 받지 않아(400 오류) 적용하지 못했습니다.

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

혼동행렬(4개 seed 합산, 행=정답·열=예측. EchoSight는 사진×질문 단위, Oracle은 질문 단위이며 같은 영상이 여러 seed의 test에 들어가면 중복해서 셉니다):

| EchoSight | 예 | 아니오 |
|---|---|---|
| 정답 예 | 206 | 100 |
| 정답 아니오 | 66 | 252 |

| Oracle | 예 | 아니오 |
|---|---|---|
| 정답 예 | 70 | 32 |
| 정답 아니오 | 4 | 102 |

첫 단어가 예/아니오가 아닌 답("기타")은 오답으로 세는데, 이번 실행에는 없었습니다.

## 해석과 한계
- **EchoSight는 Oracle보다 평균 약 9%p 낮습니다**(seed마다 6~11%p, 같은 방향). 다만 seed마다 영상 26개·질문 52개라 seed 간 표준편차가 5~7%p로 크고, 두 조건의 차이를 검정하지는 않았습니다.
- **Oracle도 100%가 아닙니다.** Oracle 오답을 영상·질문 단위로 중복 없이 세면 164문항 중 30개(18%)입니다. 이 30개를 설명과 비교해 직접 읽고 분류하면 25개는 설명에 정답 근거가 없었고(시설명·호수·상호 등), 3개는 근거가 있는데 모델이 틀렸거나 추론을 못 했으며, 2개는 질문·정답이 모호했습니다. 이 분류는 사람이 읽은 판단이라 경계 사례는 달리 볼 수 있습니다. 근거가 없을 때 모델이 "아니오"로 답하는 경향이 혼동행렬에서 보입니다(정답 "예" 중 Oracle의 31%, EchoSight의 33%가 "아니오").
- **프롬프트 수정 전후 비교**: 처음 프롬프트(SYSTEM·지시문 위치가 달랐음) 결과는 `results/qa_summary_v1_prompt.csv`에 남겨 두었고 평균은 EchoSight 0.720·Oracle 0.803이었습니다. 두 실행에서 Oracle의 같은 문항 164개 중 오답이 33개 → 30개로 바뀌었고, 두 실행에 모두 오답인 문항은 27개입니다. 설명에 근거가 없는 문항에서 모델이 찍은 답이 바뀐 경우가 있어, 프롬프트 차이와 호출 무작위성을 구분하지는 못합니다.
- **`place_description`을 어떻게 썼는지에 따라 해석이 달라집니다.** Oracle은 설명에 정답 근거가 있다는 전제의 상한입니다. 질문을 보고 설명에 내용을 넣었다면 Oracle(과, 같은 건물의 다른 영상 설명을 쓰는 EchoSight)이 실제 서비스보다 높게 나올 수 있는데, 작성 방식을 확인하지 못했습니다.
- 질문·정답이 모호한 문항(다솔 IMG_7277, 니콜스 IMG_7123, 처음 실행에서 틀렸던 밤비노 5403)은 팀의 확인이 필요합니다.
- 마리아관은 낮 영상이 6개뿐이라 test에 1개만 들어갑니다. 건물별 점수는 계산하지 않았고, 계산하더라도 마리아관은 영상 1개에 좌우됩니다.
- 사진 3장, 설명 1개, 건물 확률/재정렬기 가중치 0.5는 임의로 정한 값이며 val로 비교하지 않았습니다.
- temperature를 0으로 고정하지 못했고 같은 프롬프트로 반복 실행한 적도 없어서, 호출 무작위성에서 오는 오차는 재지 못했습니다. 위 표는 한 번 실행한 값입니다.
