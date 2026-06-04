from __future__ import annotations

from compression_app.branches.audio_branch import AudioBranch
from compression_app.branches.base import CompressionBranch
from compression_app.branches.image_branch import ImageBranch
from compression_app.branches.text_branch import TextBranch
from compression_app.core.models import DataType

BRANCHES: list[CompressionBranch] = [
    ImageBranch(),
    TextBranch(),
    AudioBranch(),
]


def get_branch_for_data_type(data_type: DataType) -> CompressionBranch | None:
    for branch in BRANCHES:
        if branch.can_handle(data_type):
            return branch
    return None
