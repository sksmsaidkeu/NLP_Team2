"""명세 QA 평가: test 영상마다 inside·outside 예/아니오 질문 2개. context는 place_description만.
조건: oracle(자기 영상 설명) / echo(사진→상위 3건물 후보 영상 설명 → 재정렬기 → 1개, 자기 영상은 후보에서 제외).
영상 단위 평균 Accuracy(seed별) → 평균±표준편차, Confusion Matrix(seed 합산).
실행: python eval_spec_qa.py [seed ...]   (기본 0 1 2 3, Docker chroma·ANTHROPIC_API_KEY 필요)
호출 결과는 eval_spec_qa_results.jsonl에 쌓여 다시 돌리면 이어서 한다(API 비용 절약).
"""
import pandas as pd  # torch보다 먼저 (build_chroma.py 참고)
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import chromadb
from generate import llm, rerank_scores, minmax

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")  # 결과 파일 폴더
OUT = os.path.join(RES, "eval_spec_qa_results.jsonl")
MODEL = "claude-haiku-5-5"
NFRAMES = 3         # 영상당 질의 사진 수 (균등 간격)
TOP_B, ALPHA, TEMP = 3, 0.5, 0.02  # 후보 건물 수, 사진 점수 비중, 건물 softmax 온도 (generate.py·EchoSight와 같은 값)
# 프롬프트는 RAG 평가 가이드 슬라이드 35~37 그대로 (temperature 0은 이 모델이 지원하지 않아 못 맞춤)
SYSTEM = "당신은 항상 사용자가 묻는 질문에만 답합니다. 그 외의 것에는 답하지 마세요."
INSTR = "설명 없이 '예' 또는 '아니오'로만 답하세요."
SHOTS = ("Context: 알프스산맥의 남쪽은 코모 호수와 맞닿아 있다.\n"
         "Question: 이 산은 코모 호수 안에 있거나 그 옆에 있나요?\n" + INSTR + "\nShort answer is: 예\n\n"
         "Context: 에펠탑은 프랑스 파리의 센강 근처에 있다.\n"
         "Question: 이 건물은 이탈리아 로마에 있나요?\n" + INSTR + "\nShort answer is: 아니오\n\n")


def label(text):
    t = text.strip()
    return "아니오" if t.startswith("아니오") else "예" if t.startswith("예") else "기타"


def prompt(ctx, q):
    return f"{SHOTS}Context: {ctx}\nQuestion: {q}\n{INSTR}\nShort answer is:"


def call(job):
    r = llm.messages.create(model=MODEL, max_tokens=2048, system=SYSTEM, messages=[{"role": "user", "content": job["prompt"]}])
    text = "".join(b.text for b in r.content if b.type == "text")
    return {**{k: v for k, v in job.items() if k != "prompt"}, "pred": label(text), "raw": text.strip()[:40],
            "tok_in": r.usage.input_tokens, "tok_out": r.usage.output_tokens}


if __name__ == "__main__":
    assert label("예") == "예" and label(" 아니오.") == "아니오" and label("모르겠다") == "기타" and label("") == "기타"
    seeds = [int(s) for s in sys.argv[1:]] or [0, 1, 2, 3]
    sp = pd.read_csv(os.path.join(HERE, "splits.csv"))
    m = pd.read_excel(os.path.join(HERE, "nlp_project_dataset", "metadata.xlsx"), sheet_name=0).set_index("video_id")
    desc = m.place_description.to_dict()
    qa = {v: [("inside", m.at[v, "query_inside"], m.at[v, "ground_truth_inside"].strip()),
              ("outside", m.at[v, "query_outside"], m.at[v, "ground_truth_outside"].strip())] for v in sp.video_id}
    assert all(gt in ("예", "아니오") for v in qa.values() for _, _, gt in v)

    g = chromadb.HttpClient(host="localhost", port=8000).get_collection("frames_dino").get(include=["embeddings", "metadatas"])
    fr = pd.DataFrame(g["metadatas"])
    fr["i"] = range(len(fr))
    fr = fr[fr.video_id.isin(set(sp.video_id))].sort_values(["video_id", "frame_idx"]).reset_index(drop=True)
    X = np.asarray(g["embeddings"], dtype=np.float32)[fr.i.to_numpy()]
    bld = sorted(sp.character.unique())
    vb = dict(zip(sp.video_id, sp.character))
    fb = fr.video_id.map(vb).to_numpy()
    vids_of = {b: sorted(sp.video_id[sp.character == b]) for b in bld}

    done = {}
    if os.path.exists(OUT):
        done = {(r["key"]): r for r in map(json.loads, open(OUT, encoding="utf-8"))}

    test_vids = sorted({v for s in seeds for v in sp.video_id[sp[f"seed{s}"] == "test"]})
    jobs, rr_cache = [], {}
    for v in test_vids:
        for qt, q, gt in qa[v]:
            jobs.append({"key": f"oracle|{v}|-|{qt}", "cond": "oracle", "video": v, "frame": "-", "qtype": qt, "gt": gt,
                         "prompt": prompt(desc[v], q)})
        idx = np.where(fr.video_id.to_numpy() == v)[0]
        for j in np.linspace(0, len(idx) - 1, NFRAMES).round().astype(int):
            qi = idx[j]
            keep = fr.video_id.to_numpy() != v  # 자기 영상 제외 (assert로 확인)
            sim = X[keep] @ X[qi]
            mx = {b: sim[fb[keep] == b].max() for b in bld}  # 건물별 max-pooling
            top = sorted(mx, key=mx.get, reverse=True)[:TOP_B]
            e = {b: math.exp((mx[b] - mx[top[0]]) / TEMP) for b in bld}
            prob = {b: x / sum(e.values()) for b, x in e.items()}
            cand = [c for b in top for c in vids_of[b] if c != v]
            for qt, q, gt in qa[v]:
                if f"echo|{v}|{fr.frame_idx.iloc[qi]}|{qt}" in done:
                    continue
                if (q, tuple(cand)) not in rr_cache:
                    rr_cache[(q, tuple(cand))] = minmax(rerank_scores(q, [desc[c] for c in cand]))
                score = [ALPHA * prob[vb[c]] + (1 - ALPHA) * s for c, s in zip(cand, rr_cache[(q, tuple(cand))])]
                pick = cand[int(np.argmax(score))]
                assert pick != v
                jobs.append({"key": f"echo|{v}|{fr.frame_idx.iloc[qi]}|{qt}", "cond": "echo", "video": v,
                             "frame": int(fr.frame_idx.iloc[qi]), "qtype": qt, "gt": gt, "pick": pick,
                             "pick_building_ok": vb[pick] == vb[v], "prompt": prompt(desc[pick], q)})
    todo = [j for j in jobs if j["key"] not in done]
    print(f"호출 {len(todo)}건 (이미 완료 {len(jobs) - len(todo)}건)", flush=True)
    with ThreadPoolExecutor(8) as ex, open(OUT, "a", encoding="utf-8") as f:
        for r in ex.map(call, todo):
            r["key"] = r.pop("key")
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            done[r["key"]] = r
    used = [r for r in done.values()]
    print(f"누적 토큰 입력 {sum(r['tok_in'] for r in used):,} / 출력 {sum(r['tok_out'] for r in used):,}")

    d = pd.DataFrame(done.values())
    d["ok"] = d.pred == d["gt"]
    rows, cms = [], {c: pd.DataFrame(0, index=["예", "아니오"], columns=["예", "아니오", "기타"]) for c in ("echo", "oracle")}
    for s in seeds:
        t = d[d.video.isin(set(sp.video_id[sp[f"seed{s}"] == "test"]))]
        r = {"seed": s}
        for c in ("echo", "oracle"):
            tc = t[t.cond == c]
            r[f"Acc({c})"] = tc.groupby("video").ok.mean().mean()  # 질문·사진 평균 → 영상 평균
            for gt in ("예", "아니오"):
                for p in ("예", "아니오", "기타"):
                    cms[c].loc[gt, p] += ((tc["gt"] == gt) & (tc.pred == p)).sum()
        e = t[t.cond == "echo"]
        r["픽 건물 정답률"] = e.groupby("video").pick_building_ok.mean().mean()
        rows.append(r)
    res = pd.DataFrame(rows).set_index("seed")
    res.loc["mean"], res.loc["std"] = res.mean(), res.std(ddof=0)
    print(res.round(3).to_string())
    for c in cms:
        print(f"\nConfusion Matrix ({c}, 행=정답·열=예측, seed 합산)\n{cms[c].to_string()}")
    res.round(4).to_csv(os.path.join(RES, "eval_spec_qa_summary.csv"), encoding="utf-8-sig")
