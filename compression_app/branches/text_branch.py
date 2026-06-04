from __future__ import annotations

import bz2
import gzip
import io
import lzma
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
import zstandard as zstd

from compression_app.branches.base import CompressionBranch
from compression_app.core.helpers import human_size, text_property_display_from_bytes
from compression_app.core.models import CandidateResult, DataType, FinalSelection

TEXT_LADDER = [
    "zstd_1",
    "zstd_3",
    "zstd_10",
    "gzip_1",
    "gzip_9",
    "bz2_9",
    "xz_0",
    "xz_9",
    "zip_deflate",
]


def encode_text_candidate_bytes(original_name: str, original_bytes: bytes, method: str) -> Tuple[bytes, str]:
    if method == "zstd_1":
        return zstd.ZstdCompressor(level=1).compress(original_bytes), ".zst"
    if method == "zstd_3":
        return zstd.ZstdCompressor(level=3).compress(original_bytes), ".zst"
    if method == "zstd_10":
        return zstd.ZstdCompressor(level=10).compress(original_bytes), ".zst"
    if method == "gzip_1":
        return gzip.compress(original_bytes, compresslevel=1), ".gz"
    if method == "gzip_9":
        return gzip.compress(original_bytes, compresslevel=9), ".gz"
    if method == "bz2_9":
        return bz2.compress(original_bytes, compresslevel=9), ".bz2"
    if method == "xz_0":
        return lzma.compress(original_bytes, preset=0), ".xz"
    if method == "xz_9":
        return lzma.compress(original_bytes, preset=9), ".xz"
    if method == "zip_deflate":
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            zf.writestr(original_name, original_bytes)
        return buffer.getvalue(), ".zip"
    raise ValueError(f"Unsupported text method: {method}")


def decode_text_candidate_bytes(original_name: str, compressed_bytes: bytes, method: str) -> bytes:
    if method.startswith("zstd_"):
        return zstd.ZstdDecompressor().decompress(compressed_bytes)
    if method.startswith("gzip_"):
        return gzip.decompress(compressed_bytes)
    if method == "bz2_9":
        return bz2.decompress(compressed_bytes)
    if method.startswith("xz_"):
        return lzma.decompress(compressed_bytes)
    if method == "zip_deflate":
        with zipfile.ZipFile(io.BytesIO(compressed_bytes), mode="r") as zf:
            names = zf.namelist()
            target = original_name if original_name in names else names[0]
            return zf.read(target)
    raise ValueError(f"Unsupported text method: {method}")


def compute_text_quality_metrics(original_bytes: bytes, restored_bytes: bytes) -> Dict[str, float | bool]:
    lossless_ok = restored_bytes == original_bytes
    if lossless_ok:
        return {"lossless_ok": True, "char_match_ratio": 1.0}

    max_len = max(len(original_bytes), len(restored_bytes), 1)
    match = sum(1 for a, b in zip(original_bytes, restored_bytes) if a == b)
    return {"lossless_ok": False, "char_match_ratio": match / max_len}


def render_quality_metric_text(metrics: Dict[str, float | bool]) -> str:
    ratio = metrics.get("char_match_ratio")
    return f"lossless_ok={bool(metrics['lossless_ok'])}, char_match={float(ratio):.4f}"


def choose_smallest_candidate(candidates: List[CandidateResult]) -> CandidateResult:
    best = min(candidates, key=lambda item: (item.file_size_bytes, item.method))
    best.selected = True
    best.status = "selected-smallest"
    return best


def build_results_dataframe(selection: FinalSelection) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for idx, candidate in enumerate(selection.candidates):
        rows.append(
            {
                "selected": candidate.selected,
                "status": candidate.status,
                "method": candidate.method,
                "file_size_bytes": candidate.file_size_bytes,
                "file_size": human_size(candidate.file_size_bytes),
                "property": candidate.property_display,
                "extension": candidate.extension,
                "compression_ratio_display": f"{candidate.compression_ratio_x:.3f}x",
                "integrity_ok": candidate.integrity_ok,
                "quality_metric": render_quality_metric_text(candidate.quality_metrics),
                "_output_name": candidate.output_name,
                "_row_id": idx,
            }
        )
    return pd.DataFrame(rows).sort_values(by=["file_size_bytes", "method"], ascending=[True, True]).reset_index(drop=True)


class TextBranch(CompressionBranch):
    name = "text"

    def can_handle(self, data_type: DataType) -> bool:
        return data_type == "Text"

    def process(self, uploaded_file) -> Tuple[FinalSelection, pd.DataFrame, Dict[str, bytes]]:
        original_bytes = uploaded_file.getvalue()
        original_name = uploaded_file.name

        candidates: List[CandidateResult] = []
        files_for_download: Dict[str, bytes] = {}

        for method in TEXT_LADDER:
            encoded_bytes, extension = encode_text_candidate_bytes(original_name, original_bytes, method)
            restored_bytes = decode_text_candidate_bytes(original_name, encoded_bytes, method)

            candidate = CandidateResult(
                method=method,
                output_name=f"{Path(original_name).name}{extension}",
                output_bytes=encoded_bytes,
                file_size_bytes=len(encoded_bytes),
                extension=extension,
                compression_ratio_x=len(original_bytes) / len(encoded_bytes) if len(encoded_bytes) > 0 else 0.0,
                property_display=text_property_display_from_bytes(restored_bytes),
                integrity_ok=restored_bytes == original_bytes,
                quality_metrics=compute_text_quality_metrics(original_bytes, restored_bytes),
                status="candidate",
            )
            candidates.append(candidate)
            files_for_download[candidate.output_name] = candidate.output_bytes

        valid_candidates = [candidate for candidate in candidates if candidate.integrity_ok]
        choose_smallest_candidate(valid_candidates if valid_candidates else candidates)

        selection = FinalSelection(
            branch="text",
            original_name=original_name,
            original_size_bytes=len(original_bytes),
            original_extension=Path(original_name).suffix.lower(),
            original_property=text_property_display_from_bytes(original_bytes),
            candidates=candidates,
        )
        return selection, build_results_dataframe(selection), files_for_download

    def render_original_preview(self, filename: str, data: bytes) -> None:
        st.subheader("Original File Preview")
        with st.container(border=True):
            text = data.decode("utf-8", errors="replace")
            st.code(text[:4000] or "[Empty text]")

    def render_compressed_preview(self, selected_name: str, files_for_download: Dict[str, bytes], original_name: str) -> None:
        st.subheader("Compressed File Preview")
        with st.container(border=True):
            suffix_map = {
                "zstd_1": ".zst",
                "zstd_3": ".zst",
                "zstd_10": ".zst",
                "gzip_1": ".gz",
                "gzip_9": ".gz",
                "bz2_9": ".bz2",
                "xz_0": ".xz",
                "xz_9": ".xz",
                "zip_deflate": ".zip",
            }

            restored = None
            for method, suffix in suffix_map.items():
                if selected_name.endswith(suffix):
                    try:
                        restored = decode_text_candidate_bytes(original_name, files_for_download[selected_name], method)
                        break
                    except Exception:
                        restored = None

            if restored is None:
                st.info("Preview is not available for the selected compressed file.")
                return

            text = restored.decode("utf-8", errors="replace")
            st.code(text[:4000] or "[Empty text]")


_TEXT_BRANCH = TextBranch()


def process(uploaded_file) -> Tuple[FinalSelection, pd.DataFrame, Dict[str, bytes]]:
    return _TEXT_BRANCH.process(uploaded_file)


def render_original_preview(filename: str, data: bytes) -> None:
    _TEXT_BRANCH.render_original_preview(filename, data)


def render_compressed_preview(selected_name: str, files_for_download: Dict[str, bytes], original_name: str) -> None:
    _TEXT_BRANCH.render_compressed_preview(selected_name, files_for_download, original_name)


def preview_text(selected_name: str, data: bytes, original_name: str) -> str:
    suffix_map = {
        "zstd_1": ".zst",
        "zstd_3": ".zst",
        "zstd_10": ".zst",
        "gzip_1": ".gz",
        "gzip_9": ".gz",
        "bz2_9": ".bz2",
        "xz_0": ".xz",
        "xz_9": ".xz",
        "zip_deflate": ".zip",
    }

    restored = None
    for method, suffix in suffix_map.items():
        if selected_name.endswith(suffix):
            try:
                restored = decode_text_candidate_bytes(original_name, data, method)
                break
            except Exception:
                restored = None

    if restored is None:
        raise ValueError("Preview is not available for the selected compressed file.")

    return restored.decode("utf-8", errors="replace")
