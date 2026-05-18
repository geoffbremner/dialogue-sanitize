import numpy as np


def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio

    return np.mean(audio, axis=1)


def rms_db(samples: np.ndarray) -> float:
    rms = np.sqrt(np.mean(np.square(samples)))

    if rms <= 1e-12:
        return -120.0

    return 20 * np.log10(rms)


def find_nearest_zero_crossing(
    samples: np.ndarray,
    target_idx: int,
    search_radius: int = 2048,
) -> int:
    start = max(0, target_idx - search_radius)
    end = min(len(samples) - 1, target_idx + search_radius)

    region = samples[start:end]

    zero_crossings = np.where(np.diff(np.sign(region)))[0]

    if len(zero_crossings) == 0:
        return target_idx

    absolute_crossings = zero_crossings + start

    nearest = absolute_crossings[
        np.argmin(np.abs(absolute_crossings - target_idx))
    ]

    return int(nearest)


def normalize_word(word: str) -> str:
    return word.lower().strip(".,!? ")