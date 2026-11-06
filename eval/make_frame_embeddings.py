"""프레임 이미지의 임베딩을 계산해 data/frame_embeddings_<이름>.npz로 저장한다 (평가에 쓰는 영상 129개의 프레임만).
입력: 프레임 이미지 폴더(기본 data/frames)와 목록 csv(기본 data/frames.csv: video_id, frame_path, frame_idx 열), data/splits.csv
실행: python eval/make_frame_embeddings.py --model dinov2 [--frames-dir 폴더] [--frames-csv 파일] [--out 파일]
프레임 이미지는 이 저장소에 없다. 같은 프레임을 쓰려면: git checkout c65f330 -- data/frames data/frames.csv
"""
import argparse
import os

import pandas as pd
import numpy as np
from PIL import Image

from data_loading import DATA_DIR, embeddings_path, load_splits
from image_embedding import ENCODERS, pad_square

BATCH_SIZE = 32


def main():
    parser = argparse.ArgumentParser(description="프레임 임베딩 계산")
    parser.add_argument("--model", choices=sorted(ENCODERS), default="dinov2")
    parser.add_argument("--frames-dir", default=os.path.join(DATA_DIR, "frames"))
    parser.add_argument("--frames-csv", default=os.path.join(DATA_DIR, "frames.csv"))
    parser.add_argument("--out", default=None, help="기본값: data/frame_embeddings_<model>.npz")
    args = parser.parse_args()

    splits = load_splits()
    frames = pd.read_csv(args.frames_csv)
    frames = frames[frames.video_id.isin(set(splits.video_id))].sort_values(["video_id", "frame_idx"]).reset_index(drop=True)
    assert set(frames.video_id) == set(splits.video_id), "frames.csv에 없는 평가 영상이 있다"

    encoder = ENCODERS[args.model]
    chunks = []
    for start in range(0, len(frames), BATCH_SIZE):
        paths = frames.frame_path[start:start + BATCH_SIZE]
        images = [pad_square(Image.open(os.path.join(args.frames_dir, path))) for path in paths]
        chunks.append(encoder(images))
        print(f"\r{min(start + BATCH_SIZE, len(frames))}/{len(frames)}", end="", flush=True)
    embeddings = np.concatenate(chunks).astype(np.float32)

    out = args.out or embeddings_path(args.model)
    np.savez_compressed(out, embeddings=embeddings, video_id=frames.video_id.to_numpy(str),
                        frame_idx=frames.frame_idx.to_numpy(int))
    print(f"\n저장: {out}  {embeddings.shape}")


if __name__ == "__main__":
    main()
