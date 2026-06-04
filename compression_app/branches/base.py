from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import pandas as pd

from compression_app.core.models import DataType, FinalSelection


class CompressionBranch(ABC):
    name: str

    @abstractmethod
    def can_handle(self, data_type: DataType) -> bool:
        raise NotImplementedError

    @abstractmethod
    def process(self, uploaded_file) -> Tuple[FinalSelection, pd.DataFrame, Dict[str, bytes]]:
        raise NotImplementedError

    @abstractmethod
    def render_original_preview(self, filename: str, data: bytes) -> None:
        raise NotImplementedError

    @abstractmethod
    def render_compressed_preview(
        self,
        selected_name: str,
        files_for_download: Dict[str, bytes],
        original_name: str,
    ) -> None:
        raise NotImplementedError
