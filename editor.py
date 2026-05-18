import numpy as np
import soundfile as sf

from models import ExclusionZone
from utils import (
    find_nearest_zero_crossing,
    to_mono,
)

FADE_MS = 5
MERGE_THRESHOLD_SEC = 0.05


def merge_exclusions(
    exclusions: list[ExclusionZone],
) -> list[ExclusionZone]:
    if not exclusions:
        return []

    exclusions = sorted(
        exclusions,
        key=lambda x: x.absolute_start,
    )

    merged = [exclusions[0]]

    for current in exclusions[1:]:
        previous = merged[-1]

        if (
            current.absolute_start
            <= previous.absolute_end
            + MERGE_THRESHOLD_SEC
        ):
            previous.absolute_end = max(
                previous.absolute_end,
                current.absolute_end,
            )
        else:
            merged.append(current)

    return merged


def apply_crossfade(
    left: np.ndarray,
    right: np.ndarray,
    fade_samples: int,
) -> np.ndarray:
    fade_samples = min(
        fade_samples,
        len(left),
        len(right),
    )

    if fade_samples <= 0:
        return np.concatenate([left, right])

    t = np.linspace(0, 1, fade_samples)

    fade_out = np.cos(t * np.pi / 2)
    fade_in = np.sin(t * np.pi / 2)

    left_keep = left[:-fade_samples]
    right_keep = right[fade_samples:]

    left_fade = left[-fade_samples:]
    right_fade = right[:fade_samples]

    # Mono
    if left.ndim == 1:
        crossfaded = (
            left_fade * fade_out
            + right_fade * fade_in
        )

    # Stereo / multi-channel
    else:
        crossfaded = (
            left_fade * fade_out[:, None]
            + right_fade * fade_in[:, None]
        )

    return np.concatenate([
        left_keep,
        crossfaded,
        right_keep,
    ])


def rebuild_audio(
    input_path: str,
    exclusions: list[ExclusionZone],
    output_path: str,
) -> None:
    audio, sample_rate = sf.read(input_path)

    mono_audio = to_mono(audio)

    exclusions = merge_exclusions(exclusions)

    if not exclusions:
        sf.write(output_path, audio, sample_rate)
        return

    fade_samples = int(
        sample_rate * (FADE_MS / 1000.0)
    )

    keep_regions = []

    current_start_sec = 0.0

    for exclusion in exclusions:
        keep_regions.append(
            (
                current_start_sec,
                exclusion.absolute_start,
            )
        )

        current_start_sec = exclusion.absolute_end

    total_duration_sec = len(audio) / sample_rate

    keep_regions.append(
        (
            current_start_sec,
            total_duration_sec,
        )
    )

    output = None

    for idx, (start_sec, end_sec) in enumerate(keep_regions):
        start_idx = int(start_sec * sample_rate)
        end_idx = int(end_sec * sample_rate)

        start_idx = find_nearest_zero_crossing(
            mono_audio,
            start_idx,
        )

        end_idx = find_nearest_zero_crossing(
            mono_audio,
            end_idx,
        )

        segment = audio[start_idx:end_idx]

        if len(segment) == 0:
            continue

        if output is None:
            output = segment
            continue

        output = apply_crossfade(
            output,
            segment,
            fade_samples,
        )

    if output is None:
        output = np.zeros((1,), dtype=np.float32)

    sf.write(
        output_path,
        output,
        sample_rate,
    )