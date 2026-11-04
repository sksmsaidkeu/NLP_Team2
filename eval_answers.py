import sys, os, io, json, base64
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, numpy as np
import eval_retrieval as E, generate, search
from PIL import Image
STEP = int(sys.argv[1]) if len(sys.argv) > 1 else 1      # 파일럿: 행을 STEP개마다 하나씩
OUT = sys.argv[2] if len(sys.argv) > 2 else r"C:\Users\happy\AppData\Local\Temp\ans.jsonl"
ev = pd.read_excel(os.path.join(E.HERE, "eval_evidence.xlsx"))
m = pd.read_excel(os.path.join(E.HERE, "nlp_project_dataset", "metadata.xlsx"))
mi = m.set_index("video_id")
ev["truth"] = [mi.loc[r.video_id, "ground_truth_" + r.유형] for r in ev.itertuples()]   # 최신 metadata 정답
ev["cat"] = ev.확정_섹션.map(lambda s: s if s in ["없음", "없음(부정형)", "식별만"] else "청크")
gold = [{x.split(" | ")[0].strip() for x in s.split(";")} if c == "청크" else set() for s, c in zip(ev.확정_섹션, ev["cat"])]
DOC_OF = dict(zip(E.C_ID, E.C_DOC))
bld = E.C_BLD
VSYS = generate.SYSTEM.replace("아래 <문서>와 <장소>에 있는 내용으로만 답한다.", "사진과 네가 아는 지식으로 답한다.").replace("문서와 장소 어디에도 근거가 없으면", "모르면").replace("<장소>", "사진") + "근거 문서가 없으므로 '문서에 없음'은 쓰지 말고, 확실하지 않아도 첫 줄을 '예' 또는 '아니오'로 정한다. "
def b64(path):
    im = Image.open(path).convert("RGB"); im.thumbnail((768, 768)); b = io.BytesIO(); im.save(b, "JPEG", quality=85); return base64.b64encode(b.getvalue()).decode()
def vanilla(path, q):
    r = generate.llm.messages.create(model="claude-haiku-5-5", max_tokens=1024, system=VSYS, messages=[{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64(path)}}, {"type": "text", "text": f"질문: {q}"}]}])
    return "".join(b.text for b in r.content if b.type == "text"), r.usage.input_tokens
def gen(q, chunks, places, conf):
    r = generate.llm.messages.create(model="claude-haiku-5-5", max_tokens=2048, system=generate.SYSTEM + (generate.PLACE_SURE if conf else generate.PLACE_UNSURE),
        messages=[{"role": "user", "content": "\n\n".join(f"<문서>\n{c}\n</문서>" for c in chunks) + "".join(f'\n\n<장소 건물="{b}" 유사도="{s:.2f}">\n{d}\n</장소>' for b, s, d in places) + f"\n\n질문: {q}"}])
    return "".join(b.text for b in r.content if b.type == "text"), r.usage.input_tokens
done = {(d["row"], d["cond"]) for d in map(json.loads, open(OUT, encoding="utf-8"))} if os.path.exists(OUT) else set()
with open(OUT, "a", encoding="utf-8") as f:
    for n in range(0, len(ev), STEP):
        r = ev.loc[n]
        fr = E.frames[E.frames.video_id == r.video_id].sort_values("frame_idx").frame_path.tolist(); fid = fr[len(fr) // 2]
        votes, hits = E.identify(fid, r.video_id); share = dict(votes); conf = max(share.values()) >= generate.CONF_HIGH
        q = search.txt_model.encode([r.질문], normalize_embeddings=True)[0]
        cands = [(float(E.C_EMB[i] @ q), bld[i], E.C_SEC[i], E.C_DOC[i], E.C_EMB[i]) for i in range(len(E.C_ID))]
        loc = generate.near_places(hits, search.img_col, max(share, key=share.get), generate.LOC_K)
        conds = {}
        top = generate.rank_chunks(cands, share, "filter", 0, 1)[0]; conds["filter"] = ([top[3]], top[1])
        tc = generate.tune_cands(cands, share, r.질문, loc, True, False, False, search.txt_model)
        top = generate.rank_chunks(tc, share, "echo", 0, 1)[0]; conds["echo+rerank"] = ([top[3]], top[1])
        conds["echo+rerank, 장소 없음"] = (conds["echo+rerank"][0], None)
        if gold[n]: conds["[정답 청크]"] = ([DOC_OF[sorted(gold[n])[0]]], r.건물)
        for cond in ["vanilla"] + list(conds):
            if (n, cond) in done: continue
            if cond == "vanilla":
                t, tok = vanilla(os.path.join(E.HERE, "frames", fid), r.질문)
            else:
                chunks, b = conds[cond]
                places = [] if b is None else generate.near_places(hits, search.img_col, b)
                t, tok = gen(r.질문, chunks, places, conf)
            f.write(json.dumps(dict(row=n, cond=cond, v=generate.verdict(t), truth=r.truth, cat=r["cat"], 유형=r.유형, id_ok=max(share, key=share.get) == r.건물, tok=tok), ensure_ascii=False) + "\n"); f.flush()
print("done")
