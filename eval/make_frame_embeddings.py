"""영상에서 프레임을 뽑아 data/frames.csv를 만들고, 프레임 임베딩을 data/frame_embeddings_<이름>.npz로 저장한다 (평가에 쓰는 영상 129개만).
실행: python eval/make_frame_embeddings.py [--name evaclip] [--videos-dir data]
이미 뽑은 프레임 이미지(data/frames)는 다시 뽑지 않고 그대로 쓴다.
"""
import argparse
import math
import os

import pandas as pd
import numpy as np
import cv2

from data_loading import DATA_DIR, embeddings_path, load_splits
from image_embedding import EVACLIP

VIDEO_EXTENSIONS = (".mp4", ".mov")
SECONDS_PER_FRAME = 1.0
MIN_FRAMES = 10
MAX_FRAMES = 20


def find_video_files(videos_dir):
    """videos_dir 아래의 모든 영상을 {소문자 파일 이름: 경로}로 모은다. 파일 이름은 영상마다 달라야 한다."""
    video_files = {}
    for folder, _, file_names in os.walk(videos_dir):
        for file_name in file_names:
            if file_name.lower().endswith(VIDEO_EXTENSIONS):
                assert file_name.lower() not in video_files, f"같은 이름의 영상이 둘 이상이다: {file_name}"
                video_files[file_name.lower()] = os.path.join(folder, file_name)
    return video_files


def frame_file_name(video_id, frame_idx):
    video_name = os.path.splitext(os.path.basename(video_id))[0]
    return f"{video_name}_{frame_idx:02d}.jpg"


def save_image(image, path):
    # cv2.imwrite는 Windows에서 한글 경로를 못 쓰므로 인코딩한 뒤 직접 쓴다
    encoded = cv2.imencode(".jpg", image)[1]
    with open(path, "wb") as file:
        file.write(encoded.tobytes())


def count_frames_to_extract(video_frame_count, frames_per_second):
    video_seconds = video_frame_count / frames_per_second
    return min(MAX_FRAMES, max(MIN_FRAMES, math.ceil(video_seconds / SECONDS_PER_FRAME)))


def extract_frames(video_path, video_id, frames_dir):
    """영상을 같은 길이의 구간 N개로 나누고 각 구간의 가운데 프레임을 frames_dir에 저장한다. 저장한 frame_idx 목록을 돌려준다."""
    video = cv2.VideoCapture(video_path)
    video_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    count = count_frames_to_extract(video_frame_count, frames_per_second)

    saved_frame_indices = []
    for frame_idx in range(count):
        video.set(cv2.CAP_PROP_POS_FRAMES, int((frame_idx + 0.5) * video_frame_count / count))
        has_frame, image = video.read()
        if not has_frame:
            print(f"프레임을 읽지 못했다: {video_id} {frame_idx}번째")
            continue
        save_image(image, os.path.join(frames_dir, frame_file_name(video_id, frame_idx)))
        saved_frame_indices.append(frame_idx)
    video.release()
    return saved_frame_indices


def make_frames_table(splits, videos_dir, frames_dir):
    video_files = find_video_files(videos_dir)
    rows = []
    for number, video_id in enumerate(splits.video_id, start=1):
        video_path = video_files[os.path.basename(video_id).lower()]
        frame_indices = extract_frames(video_path, video_id, frames_dir)
        for frame_idx in frame_indices:
            rows.append({"video_id": video_id, "frame_path": frame_file_name(video_id, frame_idx), "frame_idx": frame_idx})
        print(f"\r프레임 추출 {number}/{len(splits)}", end="", flush=True)
    print()
    return pd.DataFrame(rows)


def load_or_make_frames_table(splits, args):
    if os.path.exists(args.frames_csv):
        print(f"기존 프레임 목록 사용: {args.frames_csv}")
        return pd.read_csv(args.frames_csv)
    os.makedirs(args.frames_dir, exist_ok=True)
    frames = make_frames_table(splits, args.videos_dir, args.frames_dir)
    frames.to_csv(args.frames_csv, index=False)
    print(f"저장: {args.frames_csv}  프레임 {len(frames)}장")
    return frames


def main():
    parser = argparse.ArgumentParser(description="영상에서 프레임을 뽑고 프레임 임베딩 계산")
    parser.add_argument("--name", default="evaclip", help="임베딩 이름: data/frame_embeddings_<이름>.npz")
    parser.add_argument("--videos-dir", default=DATA_DIR, help="영상이 들어 있는 폴더(하위 폴더까지 찾음)")
    parser.add_argument("--frames-dir", default=os.path.join(DATA_DIR, "frames"))
    parser.add_argument("--frames-csv", default=os.path.join(DATA_DIR, "frames.csv"))
    args = parser.parse_args()

    splits = load_splits()
    frames = load_or_make_frames_table(splits, args)
    frames = frames.sort_values(["video_id", "frame_idx"]).reset_index(drop=True)
    assert set(frames.video_id) == set(splits.video_id), "frames.csv에 없는 평가 영상이 있다"

    image_paths = [os.path.join(args.frames_dir, path) for path in frames.frame_path]
    embeddings = EVACLIP().transform(image_paths).astype(np.float32)

    out_path = embeddings_path(args.name)
    np.savez_compressed(out_path, embeddings=embeddings, video_id=frames.video_id.to_numpy(str),
                        frame_idx=frames.frame_idx.to_numpy(int))
    print(f"저장: {out_path}  {embeddings.shape}")


if __name__ == "__main__":
    main()
