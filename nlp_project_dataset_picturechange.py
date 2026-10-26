"""metadata.xlsx의 영상 -> 균등 분할 프레임, HEIC -> jpg. 결과: frames/ + frames.csv

영상당 장수 N = clamp(ceil(길이/SEC_PER_FRAME), MIN_N, MAX_N), N등분한 구간의 중앙에서 1장씩.
"""
import math
from pathlib import Path

import cv2
import pandas as pd
from PIL import Image
from pillow_heif import register_heif_opener

SEC_PER_FRAME, MIN_N, MAX_N = 1.0, 10, 20

register_heif_opener()
root = Path(__file__).parent / "nlp_project_dataset"
out = Path(__file__).parent / "frames"
out.mkdir(exist_ok=True)

rows = []
for r in pd.read_excel(root / "metadata.xlsx").itertuples():
    src = root / r.video_id
    stem = Path(r.video_id).with_suffix("").as_posix().replace("/", "_")
    base = dict(video_id=r.video_id, character=r.character, time=r.time,
                place_description=r.place_description)
    if r.type == "photo":
        p = out / f"{stem}_00.jpg"
        Image.open(src).convert("RGB").save(p, quality=95)
        rows.append(dict(base, frame_path=p.name, frame_idx=0))
        continue
    cap = cv2.VideoCapture(str(src))
    total, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    n = min(MAX_N, max(MIN_N, math.ceil(total / fps / SEC_PER_FRAME)))
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((i + 0.5) * total / n))
        ok, img = cap.read()
        if not ok:
            print("읽기 실패:", r.video_id, i)
            continue
        p = out / f"{stem}_{i:02d}.jpg"
        p.write_bytes(cv2.imencode(".jpg", img)[1].tobytes())  # 한글 경로 대응
        rows.append(dict(base, frame_path=p.name, frame_idx=i))
    cap.release()

df = pd.DataFrame(rows)
df.to_csv(Path(__file__).parent / "frames.csv", index=False, encoding="utf-8-sig")
print(len(df), "frames,", df.video_id.nunique(), "sources")
