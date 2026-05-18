from faster_whisper import WhisperModel
from tqdm import tqdm

from models import Chunk, ExclusionZone
from utils import normalize_word

TARGET_FILLERS = {
    "um",
    "uh",
    "uhm",
}

INITIAL_PROMPT = "Uh, um, okay. Uhm, so yeah."

TIMESTAMP_PAD_SEC = 0.5


model = WhisperModel(
    "base.en",
    device="cpu",
    compute_type="int8",
)


def detect_fillers(
    chunks: list[Chunk],
) -> list[ExclusionZone]:
    exclusions: list[ExclusionZone] = []

    for chunk in tqdm(chunks):
        segments, _ = model.transcribe(
            chunk.file_path,
            word_timestamps=True,
            initial_prompt=INITIAL_PROMPT,
            vad_filter=True,
        )

        for segment in segments:
            print(segment)
            if not segment.words:
                continue

            for word in segment.words:
                cleaned = normalize_word(word.word)

                if cleaned not in TARGET_FILLERS:
                    continue

                start = (
                    chunk.absolute_start_sec
                    + word.start
                    - TIMESTAMP_PAD_SEC
                )

                end = (
                    chunk.absolute_start_sec
                    + word.end
                    + TIMESTAMP_PAD_SEC
                )

                exclusions.append(
                    ExclusionZone(
                        word=cleaned,
                        absolute_start=max(0.0, start),
                        absolute_end=end,
                    )
                )

    return exclusions