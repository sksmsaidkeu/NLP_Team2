import pandas as pd  # torch보다 먼저 import해야 함 (build_chroma.py 참고)
import os
import chromadb
from PIL import Image
import dino

NAME = "frames_dino"  # 건물 식별 전용. 다른 컬렉션은 건드리지 않고 이것만 다시 만든다
HERE = os.path.dirname(os.path.abspath(__file__))


def pad_square(img, color=(255, 255, 255)):
    # build_chroma.py와 같은 흰색 letterbox (거기 import하면 DB를 다시 만들어서 복사해 둠)
    img = img.convert("RGB")
    side = max(img.size)
    canvas = Image.new("RGB", (side, side), color)
    canvas.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
    return canvas


df = pd.read_csv(os.path.join(HERE, "frames.csv"))
emb = dino.embed([pad_square(Image.open(os.path.join(HERE, "frames", p))) for p in df.frame_path])

client = chromadb.HttpClient(host="localhost", port=8000)
if NAME in [c.name for c in client.list_collections()]:
    client.delete_collection(NAME)
col = client.create_collection(NAME, embedding_function=None, configuration={"hnsw": {"space": "cosine"}},
                               metadata={"image_model": dino.NAME, "feature": "cls+patchmean", "padding": "square_letterbox_rgb(255, 255, 255)"})
col.add(ids=df.frame_path.tolist(), embeddings=emb.tolist(),
        metadatas=df[["video_id", "character", "time", "frame_idx"]].to_dict("records"))
print(f"{NAME}: {col.count()} records, dim={emb.shape[1]}, device={dino.device}")
