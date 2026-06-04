# ~/Uyen_Project/streamlit/app.py

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import streamlit as st

from compression_app.branches import audio_branch, image_branch, text_branch
from compression_app.core.helpers import (
    get_selected_output_name,
    guess_data_type,
    initialize_table_selection,
    normalize_single_selection,
)
from compression_app.core.style import inject_styles
from compression_app.core.models import DataType


BRANCHES = {
    "Image": image_branch,
    "Text": text_branch,
    "Audio": audio_branch,
}


def _human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    value = float(num_bytes)

    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{int(value)} B" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1024

    return f"{num_bytes} B"


def _safe_extension(filename: str) -> str:
    return Path(filename).suffix.lower() or "N/A"


def _extract_property_value(filename: str, data: bytes, data_type: DataType) -> str:
    if data_type == "Image":
        try:
            from PIL import Image, ImageOps
            import io

            with Image.open(io.BytesIO(data)) as img:
                normalized = ImageOps.exif_transpose(img).copy()
                return f"{normalized.size[0]}×{normalized.size[1]}"
        except Exception:
            return "Unknown"

    if data_type == "Text":
        try:
            text = data.decode("utf-8", errors="replace")
            lines = text.count("\n") + (1 if text else 0)
            chars = len(text)
            return f"{lines} lines, {chars} chars"
        except Exception:
            return "Unknown"

    if data_type == "Audio":
        try:
            from pydub import AudioSegment
            import io

            audio = AudioSegment.from_file(io.BytesIO(data), format=Path(filename).suffix.lower().lstrip(".") or None)
            duration_sec = len(audio) / 1000.0
            sample_width_bits = audio.sample_width * 8
            return f"{duration_sec:.2f}s, {audio.channels}ch, {audio.frame_rate}Hz, {sample_width_bits}-bit"
        except Exception:
            return "Unknown"

    return "N/A"


def render_original_preview(filename: str, data: bytes, data_type: DataType) -> None:
    st.subheader("Original File Preview")
    with st.container(border=True):
        if data_type == "Image":
            try:
                from PIL import Image, ImageOps
                import io

                with Image.open(io.BytesIO(data)) as img:
                    image = ImageOps.exif_transpose(img).copy()
                preview_width = max(160, image.size[0] // 2)
                st.image(image, width=preview_width)
            except Exception as exc:
                st.info(f"Preview is not available for this file: {filename} ({exc})")
            return

        if data_type == "Text":
            text = data.decode("utf-8", errors="replace")
            st.code(text[:4000] or "[Empty text]")
            return

        if data_type == "Audio":
            mime = mimetypes.guess_type(filename)[0] or "audio/wav"
            st.audio(data, format=mime)
            return

        st.info(f"Preview is not available for this file type: {filename}")


def render_data_properties(filename: str, data: bytes, data_type: DataType) -> None:
    st.subheader("Data Properties")
    property_value = _extract_property_value(filename, data, data_type)
    property_label = "Dimension" if data_type == "Image" else "Lines / Size" if data_type == "Text" else "Duration / Audio Info" if data_type == "Audio" else "Property"

    st.markdown(
        f"""
        <div class="data-card">
            <div class="data-props-grid">
                <div>
                    <div class="prop-label">Data Type</div>
                    <div class="prop-value">{data_type}</div>
                </div>
                <div>
                    <div class="prop-label">File Size</div>
                    <div class="prop-value">{_human_size(len(data))}</div>
                </div>
                <div>
                    <div class="prop-label">{property_label}</div>
                    <div class="prop-value">{property_value}</div>
                </div>
                <div>
                    <div class="prop-label">File Extension</div>
                    <div class="prop-value">{_safe_extension(filename)}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_compressed_preview(
    selected_name: str,
    files_for_download: Dict[str, bytes],
    branch_name: str,
    original_name: str,
) -> None:
    st.subheader("Compressed File Preview")
    with st.container(border=True):
        data = files_for_download[selected_name]

        if branch_name == "image":
            try:
                from PIL import Image, ImageOps
                import io

                with Image.open(io.BytesIO(data)) as img:
                    image = ImageOps.exif_transpose(img).copy()
                preview_width = max(160, image.size[0] // 2)
                st.image(image, width=preview_width)
            except Exception as exc:
                st.info(f"Image preview failed: {exc}")
            return

        if branch_name == "text":
            try:
                restored_text = text_branch.preview_text(selected_name, data, original_name)
                st.code(restored_text[:4000] or "[Empty text]")
            except Exception as exc:
                st.info(f"Text preview failed: {exc}")
            return

        if branch_name == "audio":
            mime = mimetypes.guess_type(selected_name)[0]
            if not mime:
                suffix = Path(selected_name).suffix.lower()
                mime = {
                    ".mp3": "audio/mpeg",
                    ".ogg": "audio/ogg",
                    ".opus": "audio/ogg",
                    ".flac": "audio/flac",
                    ".wav": "audio/wav",
                }.get(suffix, "application/octet-stream")
            st.audio(data, format=mime)
            return

        st.info("Preview is not available for the selected compressed file.")


def render_empty_state() -> None:
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Original File Preview")
        with st.container(border=True):
            st.write("Preview will appear here.")

    with right:
        st.subheader("Data Properties")
        st.markdown(
            """
            <div class="data-card">
                <div class="data-props-grid">
                    <div><div class="prop-label">Data Type</div><div class="prop-value">N/A</div></div>
                    <div><div class="prop-label">File Size</div><div class="prop-value">N/A</div></div>
                    <div><div class="prop-label">Property</div><div class="prop-value">N/A</div></div>
                    <div><div class="prop-label">File Extension</div><div class="prop-value">N/A</div></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.button("RUN", width="stretch")

    st.subheader("Compression Results Table")
    with st.container(border=True):
        st.write("Results will appear here after running compression.")

    st.subheader("Compressed File Preview")
    with st.container(border=True):
        st.write("Compressed preview will appear here.")


def _reset_result_state(file_signature: str) -> None:
    previous_signature = st.session_state.get("uploaded_signature")
    if previous_signature == file_signature:
        return

    st.session_state["uploaded_signature"] = file_signature
    for key in [
        "compression_result_df",
        "compression_files",
        "selected_output_name",
        "compression_branch",
        "compression_original_name",
    ]:
        st.session_state.pop(key, None)


def _run_selected_branch(uploaded_file, data_type: DataType) -> None:
    branch = BRANCHES.get(data_type)
    if branch is None:
        st.error("This compression stage currently supports Image, Text, and Audio only.")
        return

    with st.spinner(f"Running {data_type.lower()} compression ladder..."):
        selection, result_df, files_for_download = branch.process(uploaded_file)

    result_df = initialize_table_selection(result_df)
    st.session_state["compression_result_df"] = result_df
    st.session_state["compression_files"] = files_for_download
    st.session_state["selected_output_name"] = get_selected_output_name(result_df)
    st.session_state["compression_branch"] = selection.branch
    st.session_state["compression_original_name"] = selection.original_name


def _render_results_editor(result_df: pd.DataFrame) -> pd.DataFrame:
    editable_df = result_df[
        [
            "selected",
            "status",
            "method",
            "file_size",
            "property",
            "extension",
            "compression_ratio_display",
            "integrity_ok",
            "quality_metric",
            "_output_name",
            "_row_id",
        ]
    ].copy()

    return st.data_editor(
        editable_df,
        width="stretch",
        hide_index=True,
        disabled=[
            "status",
            "method",
            "file_size",
            "property",
            "extension",
            "compression_ratio_display",
            "integrity_ok",
            "quality_metric",
            "_output_name",
            "_row_id",
        ],
        column_config={
            "selected": st.column_config.CheckboxColumn("selected"),
            "status": st.column_config.TextColumn("status"),
            "method": st.column_config.TextColumn("method"),
            "file_size": st.column_config.TextColumn("file_size"),
            "property": st.column_config.TextColumn("property"),
            "extension": st.column_config.TextColumn("extension"),
            "compression_ratio_display": st.column_config.TextColumn("compression_ratio"),
            "integrity_ok": st.column_config.CheckboxColumn("integrity_ok"),
            "quality_metric": st.column_config.TextColumn("quality_metric"),
            "_output_name": None,
            "_row_id": None,
        },
        key="compression_table_editor",
    )


def main() -> None:
    st.set_page_config(page_title="Data Compression Stage", layout="wide")
    inject_styles()

    st.title("Data Compression Stage")

    uploaded_file = st.file_uploader(
        "Upload File",
        accept_multiple_files=False,
        type=[
            "png", "jpg", "jpeg", "webp", "bmp", "tiff",
            "txt", "md", "csv", "json", "xml", "yaml", "yml", "log",
            "wav", "mp3", "ogg", "opus", "flac",
        ],
    )

    if uploaded_file is None:
        render_empty_state()
        return

    uploaded_bytes = uploaded_file.getvalue()
    data_type = guess_data_type(uploaded_file.name, uploaded_bytes)
    file_signature = f"{uploaded_file.name}_{len(uploaded_bytes)}"
    _reset_result_state(file_signature)

    left, right = st.columns([1, 1])
    with left:
        render_original_preview(uploaded_file.name, uploaded_bytes, data_type)
    with right:
        render_data_properties(uploaded_file.name, uploaded_bytes, data_type)

    run_clicked = st.button("RUN", width="stretch")
    if run_clicked:
        _run_selected_branch(uploaded_file, data_type)

    result_df: Optional[pd.DataFrame] = st.session_state.get("compression_result_df")
    files_for_download: Optional[Dict[str, bytes]] = st.session_state.get("compression_files")
    branch_name: Optional[str] = st.session_state.get("compression_branch")
    original_name = st.session_state.get("compression_original_name", uploaded_file.name)

    st.subheader("Compression Results Table")

    if result_df is None or files_for_download is None or branch_name is None:
        with st.container(border=True):
            st.write("Results will appear here after running compression.")
        st.subheader("Compressed File Preview")
        with st.container(border=True):
            st.write("Compressed preview will appear here.")
        return

    edited_df = _render_results_editor(result_df)
    normalized_df, changed = normalize_single_selection(edited_df, result_df)
    st.session_state["compression_result_df"] = normalized_df
    st.session_state["selected_output_name"] = get_selected_output_name(normalized_df)

    if changed:
        st.rerun()

    selected_output_name = st.session_state["selected_output_name"]
    selected_bytes = files_for_download[selected_output_name]
    selected_mime = mimetypes.guess_type(selected_output_name)[0]
    if not selected_mime:
        selected_mime = {
            ".mp3": "audio/mpeg",
            ".ogg": "audio/ogg",
            ".opus": "audio/ogg",
            ".flac": "audio/flac",
            ".wav": "audio/wav",
        }.get(Path(selected_output_name).suffix.lower(), "application/octet-stream")

    st.markdown("**Download Selected Candidate:**")
    st.download_button(
        label=selected_output_name,
        data=selected_bytes,
        file_name=selected_output_name,
        mime=selected_mime,
        width="stretch",
    )

    render_compressed_preview(
        selected_name=selected_output_name,
        files_for_download=files_for_download,
        branch_name=branch_name,
        original_name=original_name,
    )


if __name__ == "__main__":
    main()