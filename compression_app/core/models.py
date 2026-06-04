from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal

DataType = Literal["Image", "Text", "Audio", "Binary"]
ImageClass = Literal["photo-like", "grayscale-like", "binary/icon-like"]
BranchName = Literal["image", "text", "audio"]
Status = Literal["selected-smallest", "candidate"]


@dataclass
class CandidateResult:
    method: str
    output_name: str
    output_bytes: bytes
    file_size_bytes: int
    extension: str
    compression_ratio_x: float
    property_display: str
    integrity_ok: bool
    quality_metrics: Dict[str, float | bool]
    status: Status
    selected: bool = False


@dataclass
class FinalSelection:
    branch: BranchName
    original_name: str
    original_size_bytes: int
    original_extension: str
    original_property: str
    candidates: List[CandidateResult]
