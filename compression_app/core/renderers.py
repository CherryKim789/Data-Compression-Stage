from __future__ import annotations

import streamlit as st

from compression_app.core.helpers import extract_data_size_property, file_extension, human_size
from compression_app.core.models import DataType


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        div.stButton > button,
        div.stDownloadButton > button {
            background-color: #b7e4c7;
            color: #1b4332;
            font-weight: 700;
            border: 1px solid #95d5b2;
        }

        div.stButton > button:hover,
        div.stDownloadButton > button:hover {
            background-color: #95d5b2;
            color: #081c15;
            border: 1px solid #74c69d;
        }

        .data-card {
            border: 1px solid #d9d9d9;
            border-radius: 10px;
            padding: 14px 16px;
            min-height: 150px;
            background: #ffffff;
        }

        .data-props-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 14px 24px;
        }

        .prop-label {
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 4px;
        }

        .prop-value {
            font-size: 0.875rem;
            line-height: 1.4;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_data_properties(filename: str, data: bytes, data_type: DataType) -> None:
    st.subheader("Data Properties")
    file_size = len(data)
    property_value = extract_data_size_property(filename, data, data_type)
    extension = file_extension(filename)

    if data_type == "Image":
        property_label = "Dimension"
    elif data_type == "Text":
        property_label = "Lines / Size"
    elif data_type == "Audio":
        property_label = "Duration / Audio"
    else:
        property_label = "Property"

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
                    <div class="prop-value">{human_size(file_size)}</div>
                </div>
                <div>
                    <div class="prop-label">{property_label}</div>
                    <div class="prop-value">{property_value}</div>
                </div>
                <div>
                    <div class="prop-label">File Extension</div>
                    <div class="prop-value">{extension}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_layout() -> None:
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
