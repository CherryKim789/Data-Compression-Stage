from __future__ import annotations

import io
import mimetypes
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from PIL import Image, ImageOps
from pydub import AudioSegment

from compression_app.core.models import DataType


def human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{int(value)} B" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def normalize_image(image: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(image).copy()


def load_image_from_bytes(data: bytes) -> Image.Image:
    from io import BytesIO

    with Image.open(BytesIO(data)) as img:
        return normalize_image(img)


def guess_data_type(filename: str, data: bytes) -> DataType:
    mime, _ = mimetypes.guess_type(filename)
    if mime:
        main_type = mime.split("/")[0]
        if main_type == "image":
            return "Image"
        if main_type == "text":
            return "Text"
        if main_type == "audio":
            return "Audio"

    try:
        load_image_from_bytes(data)
        return "Image"
    except Exception:
        pass

    try:
        data.decode("utf-8")
        return "Text"
    except Exception:
        pass

    try:
        AudioSegment.from_file(io.BytesIO(data))
        return "Audio"
    except Exception:
        return "Binary"


def text_property_display_from_bytes(data: bytes) -> str:
    text = data.decode("utf-8", errors="replace")
    lines = text.count("\n") + (1 if text else 0)
    chars = len(text)
    return f"{lines} lines, {chars} chars"


def image_property_display_from_bytes(data: bytes) -> str:
    image = load_image_from_bytes(data)
    return f"{image.size[0]}×{image.size[1]}"


def audio_property_display_from_bytes(data: bytes) -> str:
    segment = AudioSegment.from_file(io.BytesIO(data))
    duration_sec = len(segment) / 1000.0
    return f"{duration_sec:.2f} sec, {segment.frame_rate} Hz, {segment.channels} ch"


def extract_data_size_property(filename: str, data: bytes, data_type: DataType) -> str:
    if data_type == "Image":
        try:
            return image_property_display_from_bytes(data)
        except Exception:
            return "Unknown"
    if data_type == "Text":
        try:
            return text_property_display_from_bytes(data)
        except Exception:
            return "Unknown"
    if data_type == "Audio":
        try:
            return audio_property_display_from_bytes(data)
        except Exception:
            return "Unknown"
    return "N/A"


def initialize_table_selection(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["selected"] = False
    out.loc[out.index[0], "selected"] = True
    return out


def get_selected_output_name(df: pd.DataFrame) -> str:
    selected_rows = df[df["selected"]]
    if selected_rows.empty:
        return str(df.iloc[0]["_output_name"])
    return str(selected_rows.iloc[0]["_output_name"])


def normalize_single_selection(
    edited_df: pd.DataFrame,
    previous_df: Optional[pd.DataFrame],
) -> Tuple[pd.DataFrame, bool]:
    df = edited_df.copy()
    if "selected" not in df.columns or df.empty:
        return df, False

    selected_now = df.index[df["selected"]].tolist()

    if not selected_now:
        df["selected"] = False
        df.loc[df.index[0], "selected"] = True
        return df, True

    if len(selected_now) == 1:
        return df, False

    chosen_index = selected_now[-1]
    if previous_df is not None and len(previous_df) == len(df) and "selected" in previous_df.columns:
        changed_to_true = [
            idx for idx in selected_now
            if idx < len(previous_df) and not bool(previous_df.loc[idx, "selected"])
        ]
        if changed_to_true:
            chosen_index = changed_to_true[-1]

    df["selected"] = False
    df.loc[chosen_index, "selected"] = True
    return df, True


def file_extension(filename: str) -> str:
    return Path(filename).suffix.lower() or "N/A"
