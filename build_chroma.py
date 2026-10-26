import pandas as pd  # torch보다 먼저 import해야 함 (torch→sklearn→pyarrow 순서면 Windows에서 크래시)
import os
import re
import chromadb
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

IMAGE_MODEL = "clip-ViT-B-32"
# 컬렉션 메타데이터 "text_model" = 그 컬렉션을 검색할 때 질의를 임베딩하는 모델 (search.py가 읽음)
TEXT_MODEL = "sentence-transformers/clip-ViT-B-32-multilingual-v1"  # 질의 → 이미지 공간
DESC_MODEL = "BAAI/bge-m3"  # 설명 문장·질의 → 텍스트 전용 공간
HERE = os.path.dirname(os.path.abspath(__file__))
PAD_COLOR = (255, 255, 255)  # 흰색 letterbox


def pad_square(img):
    img = img.convert("RGB")
    side = max(img.size)
    canvas = Image.new("RGB", (side, side), PAD_COLOR)
    canvas.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
    return canvas


def recreate(name, metadata):
    if name in [c.name for c in client.list_collections()]:
        client.delete_collection(name)
    return client.create_collection(
        name,
        embedding_function=None,  # 기본 영어 MiniLM이 끼어들지 않게. 벡터는 항상 직접 넘김
        configuration={"hnsw": {"space": "cosine"}},
        metadata=metadata,
    )


df = pd.read_csv(os.path.join(HERE, "frames.csv"))
device = "cuda" if torch.cuda.is_available() else "cpu"
client = chromadb.HttpClient(host="localhost", port=8000)

# 1) 프레임 이미지: 프레임 1개 = 레코드 1개
model = SentenceTransformer(IMAGE_MODEL, device=device)
images = [pad_square(Image.open(os.path.join(HERE, "frames", p))) for p in df.frame_path]
assert all(im.width == im.height for im in images)
emb = model.encode(images, batch_size=32, normalize_embeddings=True, show_progress_bar=True)

col = recreate("frames_image", {"image_model": IMAGE_MODEL, "text_model": TEXT_MODEL,
                                "padding": f"square_letterbox_rgb{PAD_COLOR}"})
col.add(
    ids=df.frame_path.tolist(),
    embeddings=emb.tolist(),
    documents=df.place_description.tolist(),
    metadatas=df[["video_id", "character", "time", "frame_idx"]].to_dict("records"),
)
print(f"frames_image: {col.count()} records, dim={emb.shape[1]}, device={device}")

# 2) 장소 설명: 영상 1개의 설명을 문장으로 쪼개 문장 1개 = 레코드 1개 (긴 설명에 묻히지 않게 문장 단위 최고점)
videos = df.drop_duplicates("video_id")
rows = [(v.video_id, v.character, v.time, i, s)
        for v in videos.itertuples()
        for i, s in enumerate(re.split(r"(?<=다\.)\s+", v.place_description.strip()))]
sents = [r[4] for r in rows]

text_model = SentenceTransformer(DESC_MODEL, device=device)
lens = [len(text_model.tokenizer(s)["input_ids"]) for s in sents]
over = sum(n > text_model.max_seq_length for n in lens)
temb = text_model.encode(sents, batch_size=64, normalize_embeddings=True)

tcol = recreate("videos_text", {"text_model": DESC_MODEL,"split": "regex (?<=다\\.)\\s+"})
tcol.add(
    ids=[f"{vid}#{i}" for vid, _, _, i, _ in rows],
    embeddings=temb.tolist(),
    documents=sents,
    metadatas=[{"video_id": vid, "character": c, "time": t, "sent_idx": i} for vid, c, t, i, _ in rows],
)
print(f"videos_text: {tcol.count()} sentences from {len(videos)} videos, "
      f"tokens max={max(lens)}, over {text_model.max_seq_length}={over}")
