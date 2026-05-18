from dataclasses import dataclass


@dataclass
class Chunk:
    id: int
    file_path: str
    absolute_start_sec: float
    duration_sec: float


@dataclass
class ExclusionZone:
    word: str
    absolute_start: float
    absolute_end: float