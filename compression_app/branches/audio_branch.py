# ~/Uyen_Project/streamlit/compression_app/branches/audio_branch.py

from __future__ import annotations

import io
import mimetypes
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from pydub import AudioSegment

from compression_app.core.models import CandidateResult, FinalSelection


@dataclass(frozen=True)
class AudioMethodSpec:
    method: str
    export_format: str
    extension: str
    codec: Optional[str] = None
    bitrate: Optional[str] = None
    parameters: Tuple[str, ...] = ()
    read_formats: Tuple[str, ...] = ()


def _human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{int(value)} B" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def _safe_ratio(original_size: int, candidate_size: int) -> float:
    if candidate_size <= 0:
        return 0.0
    return original_size / candidate_size


def _ensure_ffmpeg_runtime() -> Tuple[str, str]:
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")

    if not ffmpeg_path or not ffprobe_path:
        raise RuntimeError(
            "ffmpeg/ffprobe is not available in PATH. Install ffmpeg before using audio compression."
        )

    AudioSegment.converter = ffmpeg_path
    AudioSegment.ffmpeg = ffmpeg_path
    AudioSegment.ffprobe = ffprobe_path
    return ffmpeg_path, ffprobe_path


def _get_available_ffmpeg_encoders(ffmpeg_path: str) -> set[str]:
    try:
        result = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to inspect ffmpeg encoders: {exc}") from exc

    encoders: set[str] = set()
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("A"):
            encoders.add(parts[1])
    return encoders


def _resolve_method_specs(ffmpeg_path: str) -> Dict[str, AudioMethodSpec]:
    encoders = _get_available_ffmpeg_encoders(ffmpeg_path)
    specs: Dict[str, AudioMethodSpec] = {}

    if "libmp3lame" in encoders:
        specs["mp3_64k"] = AudioMethodSpec(
            method="mp3_64k",
            export_format="mp3",
            extension=".mp3",
            codec="libmp3lame",
            bitrate="64k",
            read_formats=("mp3",),
        )
        specs["mp3_128k"] = AudioMethodSpec(
            method="mp3_128k",
            export_format="mp3",
            extension=".mp3",
            codec="libmp3lame",
            bitrate="128k",
            read_formats=("mp3",),
        )

    if "libopus" in encoders:
        specs["ogg_24k"] = AudioMethodSpec(
            method="ogg_24k",
            export_format="ogg",
            extension=".ogg",
            codec="libopus",
            bitrate="24k",
            read_formats=("ogg", "opus"),
        )
        specs["ogg_48k"] = AudioMethodSpec(
            method="ogg_48k",
            export_format="ogg",
            extension=".ogg",
            codec="libopus",
            bitrate="48k",
            read_formats=("ogg", "opus"),
        )
        specs["opus_24k"] = AudioMethodSpec(
            method="opus_24k",
            export_format="opus",
            extension=".opus",
            codec="libopus",
            bitrate="24k",
            read_formats=("opus", "ogg"),
        )
        specs["opus_48k"] = AudioMethodSpec(
            method="opus_48k",
            export_format="opus",
            extension=".opus",
            codec="libopus",
            bitrate="48k",
            read_formats=("opus", "ogg"),
        )
    elif "libvorbis" in encoders:
        specs["ogg_64k"] = AudioMethodSpec(
            method="ogg_64k",
            export_format="ogg",
            extension=".ogg",
            codec="libvorbis",
            bitrate="64k",
            read_formats=("ogg",),
        )
        specs["ogg_128k"] = AudioMethodSpec(
            method="ogg_128k",
            export_format="ogg",
            extension=".ogg",
            codec="libvorbis",
            bitrate="128k",
            read_formats=("ogg",),
        )

    if "flac" in encoders:
        specs["flac_lossless"] = AudioMethodSpec(
            method="flac_lossless",
            export_format="flac",
            extension=".flac",
            codec="flac",
            read_formats=("flac",),
        )

    if "pcm_s16le" in encoders:
        specs["wav_pcm16"] = AudioMethodSpec(
            method="wav_pcm16",
            export_format="wav",
            extension=".wav",
            codec="pcm_s16le",
            read_formats=("wav",),
        )

    return specs


def _select_audio_ladder(specs: Dict[str, AudioMethodSpec]) -> List[str]:
    preferred_order = [
        "opus_24k",
        "opus_48k",
        "ogg_24k",
        "ogg_48k",
        "ogg_64k",
        "ogg_128k",
        "mp3_64k",
        "mp3_128k",
        "flac_lossless",
        "wav_pcm16",
    ]
    return [method for method in preferred_order if method in specs]


def _guess_input_format(filename: str) -> Optional[str]:
    suffix = Path(filename).suffix.lower().lstrip(".")
    if suffix in {"mp3", "ogg", "opus", "flac", "wav"}:
        return suffix
    return None


def _load_audio_segment(data: bytes, filename: str) -> AudioSegment:
    file_obj = io.BytesIO(data)
    return AudioSegment.from_file(file_obj, format=_guess_input_format(filename))


def _audio_property_display(segment: AudioSegment) -> str:
    duration_sec = len(segment) / 1000.0
    channels = segment.channels
    frame_rate = segment.frame_rate
    sample_width_bits = segment.sample_width * 8
    return f"{duration_sec:.2f}s, {channels}ch, {frame_rate}Hz, {sample_width_bits}-bit"


def _build_export_kwargs(spec: AudioMethodSpec) -> Dict[str, object]:
    kwargs: Dict[str, object] = {"format": spec.export_format}
    if spec.codec:
        kwargs["codec"] = spec.codec
    if spec.bitrate:
        kwargs["bitrate"] = spec.bitrate
    if spec.parameters:
        kwargs["parameters"] = list(spec.parameters)
    return kwargs


def _export_segment(
    segment: AudioSegment,
    method: str,
    specs: Dict[str, AudioMethodSpec],
) -> Tuple[bytes, str]:
    spec = specs[method]
    buffer = io.BytesIO()
    segment.export(buffer, **_build_export_kwargs(spec))
    return buffer.getvalue(), spec.extension


def _decode_exported_audio(
    data: bytes,
    method: str,
    specs: Dict[str, AudioMethodSpec],
) -> AudioSegment:
    spec = specs[method]
    last_error: Optional[Exception] = None

    for fmt in spec.read_formats:
        try:
            return AudioSegment.from_file(io.BytesIO(data), format=fmt)
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        raise last_error

    return AudioSegment.from_file(io.BytesIO(data))


def _compute_audio_quality_metrics(
    original_segment: AudioSegment,
    candidate_segment: AudioSegment,
) -> Dict[str, float | bool]:
    duration_diff_ms = abs(len(original_segment) - len(candidate_segment))
    original_duration_ms = max(len(original_segment), 1)

    return {
        "duration_diff_ms": float(duration_diff_ms),
        "duration_match": duration_diff_ms <= 50,
        "channels_match": original_segment.channels == candidate_segment.channels,
        "frame_rate_match": original_segment.frame_rate == candidate_segment.frame_rate,
        "duration_ratio": max(len(candidate_segment), 1) / original_duration_ms,
    }


def _integrity_ok(original_segment: AudioSegment, candidate_segment: AudioSegment) -> bool:
    duration_diff_ms = abs(len(original_segment) - len(candidate_segment))
    return duration_diff_ms <= 50 and original_segment.channels == candidate_segment.channels


def _render_quality_metric_text(metrics: Dict[str, float | bool]) -> str:
    return (
        f"duration_diff_ms={float(metrics['duration_diff_ms']):.0f}, "
        f"channels_match={bool(metrics['channels_match'])}, "
        f"rate_match={bool(metrics['frame_rate_match'])}"
    )


def _choose_smallest_candidate(candidates: List[CandidateResult]) -> CandidateResult:
    best = min(candidates, key=lambda item: (item.file_size_bytes, item.method))
    best.selected = True
    best.status = "selected-smallest"
    return best


def _build_results_dataframe(selection: FinalSelection) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for idx, candidate in enumerate(selection.candidates):
        rows.append(
            {
                "selected": candidate.selected,
                "status": candidate.status,
                "method": candidate.method,
                "file_size_bytes": candidate.file_size_bytes,
                "file_size": _human_size(candidate.file_size_bytes),
                "property": candidate.property_display,
                "extension": candidate.extension,
                "compression_ratio_display": f"{candidate.compression_ratio_x:.3f}x",
                "integrity_ok": candidate.integrity_ok,
                "quality_metric": _render_quality_metric_text(candidate.quality_metrics),
                "_output_name": candidate.output_name,
                "_row_id": idx,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(by=["file_size_bytes", "method"], ascending=[True, True])
        .reset_index(drop=True)
    )


def _evaluate_candidate(
    original_name: str,
    original_bytes: bytes,
    original_segment: AudioSegment,
    method: str,
    specs: Dict[str, AudioMethodSpec],
) -> CandidateResult:
    encoded_bytes, extension = _export_segment(original_segment, method, specs)
    decoded_segment = _decode_exported_audio(encoded_bytes, method, specs)

    file_size_bytes = len(encoded_bytes)
    compression_ratio_x = _safe_ratio(len(original_bytes), file_size_bytes)
    property_display = _audio_property_display(decoded_segment)
    integrity_ok = _integrity_ok(original_segment, decoded_segment)
    quality_metrics = _compute_audio_quality_metrics(original_segment, decoded_segment)
    output_name = f"{Path(original_name).stem}_{method}{extension}"

    return CandidateResult(
        method=method,
        output_name=output_name,
        output_bytes=encoded_bytes,
        file_size_bytes=file_size_bytes,
        extension=extension,
        compression_ratio_x=compression_ratio_x,
        property_display=property_display,
        integrity_ok=integrity_ok,
        quality_metrics=quality_metrics,
        status="candidate",
    )


def preview_mime(filename: str) -> str:
    mime = mimetypes.guess_type(filename)[0]
    if mime:
        return mime

    return {
        ".mp3": "audio/mpeg",
        ".ogg": "audio/ogg",
        ".opus": "audio/ogg",
        ".flac": "audio/flac",
        ".wav": "audio/wav",
    }.get(Path(filename).suffix.lower(), "application/octet-stream")


def process(uploaded_file):
    ffmpeg_path, _ = _ensure_ffmpeg_runtime()
    specs = _resolve_method_specs(ffmpeg_path)
    ladder = _select_audio_ladder(specs)

    if not ladder:
        raise RuntimeError(
            "No supported audio encoder is available in this ffmpeg build. "
            "Need at least one of: libmp3lame, libopus, libvorbis, flac, pcm_s16le."
        )

    original_bytes = uploaded_file.getvalue()
    original_name = uploaded_file.name
    original_extension = Path(original_name).suffix.lower() or "N/A"

    original_segment = _load_audio_segment(original_bytes, original_name)
    original_property = _audio_property_display(original_segment)

    candidates: List[CandidateResult] = []
    files_for_download: Dict[str, bytes] = {}

    for method in ladder:
        candidate = _evaluate_candidate(
            original_name=original_name,
            original_bytes=original_bytes,
            original_segment=original_segment,
            method=method,
            specs=specs,
        )
        candidates.append(candidate)
        files_for_download[candidate.output_name] = candidate.output_bytes

    valid_candidates = [candidate for candidate in candidates if candidate.integrity_ok]
    _choose_smallest_candidate(valid_candidates if valid_candidates else candidates)

    selection = FinalSelection(
        branch="audio",
        original_name=original_name,
        original_size_bytes=len(original_bytes),
        original_extension=original_extension,
        original_property=original_property,
        candidates=candidates,
    )

    result_df = _build_results_dataframe(selection)
    return selection, result_df, files_for_download