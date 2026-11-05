"""낮 영상 129개를 영상 단위·건물 기준 stratified로 train/val/test 분할 (seed 0~3).
결과: splits.csv (video_id, character, seed0..seed3 = train|val|test). 팀 공유용.
실행: python make_splits.py
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = [0, 1, 2, 3]
TEST = 0.2  # 전체 train:test = 8:2
VAL = 0.2   # train 안에서 train:val = 8:2 (8:2 확정)

m = pd.read_excel(os.path.join(HERE, "nlp_project_dataset", "metadata.xlsx"), sheet_name=0)
v = m[(m["type"] == "video") & (m["time"] == "낮")][["video_id", "character"]].reset_index(drop=True)
v = v.sort_values("video_id").reset_index(drop=True)  # 정렬 고정 → 같은 seed면 같은 분할
assert len(v) == 129 and v.video_id.is_unique

for s in SEEDS:
    trval, te = train_test_split(v, test_size=TEST, stratify=v.character, random_state=s)
    tr, va = train_test_split(trval, test_size=VAL, stratify=trval.character, random_state=s)
    a, b, c = set(tr.video_id), set(va.video_id), set(te.video_id)
    assert not a & b and not a & c and not b & c and len(a | b | c) == len(v)
    v[f"seed{s}"] = v.video_id.map({**dict.fromkeys(a, "train"), **dict.fromkeys(b, "val"), **dict.fromkeys(c, "test")})
    print(f"seed{s}", pd.crosstab(v.character, v[f"seed{s}"]).T.to_string(), "\n")

v.to_csv(os.path.join(HERE, "splits.csv"), index=False, encoding="utf-8-sig")
