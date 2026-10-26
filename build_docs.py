import pandas  # torch보다 먼저 import해야 함 (Windows 크래시 회피, build_chroma.py와 동일)
import glob
import os
import re
import chromadb
import torch
from sentence_transformers import SentenceTransformer

DESC_MODEL = "BAAI/bge-m3"  # videos_text와 같은 텍스트 공간
HERE = os.path.dirname(os.path.abspath(__file__))
NAME = "building_docs"  # frames_image / videos_text는 건드리지 않고 이 컬렉션만 다시 만든다

# building_docs/*.md: front matter의 building 값 + "## 섹션" 단위 청크 1개 = 레코드 1개
rows = []
for path in sorted(glob.glob(os.path.join(HERE, "building_docs", "*.md"))):
    text = open(path, encoding="utf-8").read()
    building = re.search(r"^building:\s*(.+)$", text, re.M).group(1).strip()
    title = re.search(r"^# (.+)$", text, re.M).group(1).strip()
    body = text.split("\n# ", 1)[1].split("\n", 1)[1]
    for i, sec in enumerate(re.split(r"^## ", body, flags=re.M)[1:]):
        head, content = sec.split("\n", 1)
        # 청크 앞에 건물명·섹션 제목을 붙여 "이 건물의 무슨 내용인지"가 임베딩에 들어가게 함
        rows.append((os.path.basename(path), building, head.strip(), i, f"{title} / {head.strip()}\n{content.strip()}"))

device = "cuda" if torch.cuda.is_available() else "cpu"
client = chromadb.HttpClient(host="localhost", port=8000)
model = SentenceTransformer(DESC_MODEL, device=device)
docs = [r[4] for r in rows]
lens = [len(model.tokenizer(d)["input_ids"]) for d in docs]
emb = model.encode(docs, batch_size=16, normalize_embeddings=True)

if NAME in [c.name for c in client.list_collections()]:
    client.delete_collection(NAME)
col = client.create_collection(
    NAME,
    embedding_function=None,
    configuration={"hnsw": {"space": "cosine"}},
    metadata={"text_model": DESC_MODEL, "split": "markdown ## section"},
)
col.add(
    ids=[f"{f}#{i}" for f, _, _, i, _ in rows],
    embeddings=emb.tolist(),
    documents=docs,
    metadatas=[{"character": b, "section": h, "file": f} for f, b, h, _, _ in rows],  # character = 다른 컬렉션과 같은 건물 키
)
print(f"{NAME}: {col.count()} chunks from {len({r[0] for r in rows})} files, "
      f"tokens max={max(lens)}, over {model.max_seq_length}={sum(n > model.max_seq_length for n in lens)}")
