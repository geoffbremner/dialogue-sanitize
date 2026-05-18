import os

import numpy as np
import soundfile as sf

from models import Chunk
from utils import (
    find_nearest_zero_crossing,
    rms_db,
    to_mono,
)

TARGET_CHUNK_SEC = 30.0
SEARCH_WINDOW_SEC = 3.0

FRAME_SIZE = 1024
HOP_SIZE = 256

SILENCE_THRESHOLD_DBFS = -40.0


def find_split_point(
    mono_audio: np.ndarray,
    sample_rate: int,
    target_sec: float,
) -> int | None:
    target_idx = int(target_sec * sample_rate)

    search_start = int(
        max(
            0,
            target_idx - (SEARCH_WINDOW_SEC * sample_rate),
        )
    )

    search_end = int(
        min(
            len(mono_audio),
            target_idx + (SEARCH_WINDOW_SEC * sample_rate),
        )
    )

    best_db = None
    best_idx = None

    for idx in range(
        search_start,
        search_end - FRAME_SIZE,
        HOP_SIZE,
    ):
        frame = mono_audio[idx: idx + FRAME_SIZE]

        db = rms_db(frame)

        if best_db is None or db < best_db:
            best_db = db
            best_idx = idx

    if best_db is None:
        return None

    if best_db > SILENCE_THRESHOLD_DBFS:
        return None

    return find_nearest_zero_crossing(
        mono_audio,
        best_idx,
    )


def split_audio(
    input_path: str,
    output_dir: str,
    target_chunk_sec: float = TARGET_CHUNK_SEC,
) -> list[Chunk]:
    os.makedirs(output_dir, exist_ok=True)

    audio, sample_rate = sf.read(input_path)

    mono_audio = to_mono(audio)

    total_duration_sec = len(audio) / sample_rate

    chunks: list[Chunk] = []

    chunk_start_sec = 0.0
    chunk_id = 0

    while chunk_start_sec < total_duration_sec:
        target_split_sec = (
            chunk_start_sec + target_chunk_sec
        )

        if target_split_sec >= total_duration_sec:
            split_sec = total_duration_sec
        else:
            split_idx = find_split_point(
                mono_audio,
                sample_rate,
                target_split_sec,
            )

            if split_idx is None:
                target_split_sec += target_chunk_sec

                if target_split_sec >= total_duration_sec:
                    split_sec = total_duration_sec
                else:
                    split_idx = find_split_point(
                        mono_audio,
                        sample_rate,
                        target_split_sec,
                    )

                    if split_idx is None:
                        split_sec = total_duration_sec
                    else:
                        split_sec = split_idx / sample_rate
            else:
                split_sec = split_idx / sample_rate

        start_idx = int(chunk_start_sec * sample_rate)
        end_idx = int(split_sec * sample_rate)

        chunk_audio = audio[start_idx:end_idx]

        chunk_path = os.path.join(
            output_dir,
            f"chunk_{chunk_id:03d}.wav",
        )

        sf.write(
            chunk_path,
            chunk_audio,
            sample_rate,
        )

        chunks.append(
            Chunk(
                id=chunk_id,
                file_path=chunk_path,
                absolute_start_sec=chunk_start_sec,
                duration_sec=split_sec - chunk_start_sec,
            )
        )

        chunk_id += 1
        chunk_start_sec = split_sec

    return chunks