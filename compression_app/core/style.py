# ~/Uyen_Project/streamlit/compression_app/core/styles.py

from __future__ import annotations

import streamlit as st


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