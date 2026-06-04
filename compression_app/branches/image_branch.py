from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from skimage.metrics import structural_similarity as skimage_ssim

from compression_app.branches.base import CompressionBranch
from compression_app.core.helpers import human_size, load_image_from_bytes
from compression_app.core.models import CandidateResult, DataType, FinalSelection, ImageClass

PHOTO_LADDER = [
    "webp_q1",
    "webp_q5",
    "webp_q10",
    "webp_q50",
    "jpeg_q1",
    "jpeg_q5",
    "jpeg_q10",
    "jpeg_q50",
    "webp_lossless",
    "png_lossless",
]

GRAYSCALE_LADDER = [
    "webp_q1",
    "jpeg_q1",
    "webp_q5",
    "jpeg_q5",
    "webp_q10",
    "jpeg_q10",
    "webp_q50",
    "jpeg_q50",
    "webp_lossless",
    "png_lossless",
]

BINARY_LADDER = [
    "png_lossless",
    "webp_lossless",
    "webp_q1",
    "jpeg_q1",
    "webp_q5",
    "jpeg_q5",
    "webp_q10",
    "jpeg_q10",
    "webp_q50",
    "jpeg_q50",
]


def pil_to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    if image.mode in ("RGBA", "LA"):
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        composed = Image.alpha_composite(background, image.convert("RGBA"))
        return composed.convert("RGB")
    if image.mode == "P":
        return image.convert("RGBA").convert("RGB")
    return image.convert("RGB")


def image_to_rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(pil_to_rgb(image), dtype=np.uint8)


def rgb_to_gray_float(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0].astype(np.float64)
    g = rgb[..., 1].astype(np.float64)
    b = rgb[..., 2].astype(np.float64)
    return 0.299 * r + 0.587 * g + 0.114 * b


def otsu_threshold(gray: np.ndarray) -> int:
    hist, _ = np.histogram(gray.astype(np.uint8), bins=256, range=(0, 256))
    total = gray.size
    sum_total = np.dot(np.arange(256), hist)
    sum_b = 0.0
    weight_b = 0.0
    max_var = -1.0
    threshold = 127

    for t in range(256):
        weight_b += hist[t]
        if weight_b == 0:
            continue
        weight_f = total - weight_b
        if weight_f == 0:
            break
        sum_b += t * hist[t]
        mean_b = sum_b / weight_b
        mean_f = (sum_total - sum_b) / weight_f
        var_between = weight_b * weight_f * (mean_b - mean_f) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t

    return int(threshold)


def binary_mask(gray: np.ndarray) -> np.ndarray:
    return gray >= otsu_threshold(gray)


def edge_map(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    if m.shape[0] < 3 or m.shape[1] < 3:
        return np.zeros_like(m, dtype=bool)

    center = m[1:-1, 1:-1]
    neighbors_equal = (
        (center == m[:-2, 1:-1])
        & (center == m[2:, 1:-1])
        & (center == m[1:-1, :-2])
        & (center == m[1:-1, 2:])
        & (center == m[:-2, :-2])
        & (center == m[:-2, 2:])
        & (center == m[2:, :-2])
        & (center == m[2:, 2:])
    )
    edges = np.zeros_like(m, dtype=bool)
    edges[1:-1, 1:-1] = ~neighbors_equal
    return edges


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def edge_agreement(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    edge_a = edge_map(mask_a)
    edge_b = edge_map(mask_b)
    union = np.logical_or(edge_a, edge_b).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(edge_a, edge_b).sum()
    return float(intersection / union)


def classify_image(rgb: np.ndarray) -> ImageClass:
    gray = rgb_to_gray_float(rgb)
    gray_u8 = gray.astype(np.uint8)

    channel_std_mean = float(
        np.mean(
            [
                np.std(rgb[..., 0].astype(np.float64) - rgb[..., 1].astype(np.float64)),
                np.std(rgb[..., 0].astype(np.float64) - rgb[..., 2].astype(np.float64)),
                np.std(rgb[..., 1].astype(np.float64) - rgb[..., 2].astype(np.float64)),
            ]
        )
    )

    unique_levels = int(np.unique(gray_u8).size)
    hist, _ = np.histogram(gray_u8, bins=256, range=(0, 256))
    dominant_bins = np.sort(hist)[-2:]
    dominant_ratio = float(dominant_bins.sum() / gray_u8.size)

    if unique_levels <= 8 and dominant_ratio >= 0.90:
        return "binary/icon-like"
    if channel_std_mean < 3.0:
        return "grayscale-like"
    return "photo-like"


def select_image_ladder(image_class: ImageClass) -> List[str]:
    if image_class == "photo-like":
        return PHOTO_LADDER
    if image_class == "grayscale-like":
        return GRAYSCALE_LADDER
    return BINARY_LADDER


def method_to_image_extension(method: str) -> str:
    if method.startswith("webp_"):
        return ".webp"
    if method.startswith("jpeg_"):
        return ".jpg"
    if method == "png_lossless":
        return ".png"
    raise ValueError(f"Unsupported image method: {method}")


def encode_image_candidate_bytes(image: Image.Image, method: str, image_class: ImageClass) -> Tuple[bytes, str]:
    rgb = pil_to_rgb(image)
    gray = image.convert("L")
    ext = method_to_image_extension(method)
    buffer = io.BytesIO()

    if method == "png_lossless":
        if image_class == "binary/icon-like":
            threshold = otsu_threshold(np.asarray(gray))
            bw = gray.point(lambda p: 255 if p >= threshold else 0, mode="1")
            bw.save(buffer, format="PNG", optimize=True)
        elif image_class == "grayscale-like":
            gray.save(buffer, format="PNG", optimize=True)
        else:
            rgb.save(buffer, format="PNG", optimize=True)
        return buffer.getvalue(), ext

    if method == "webp_lossless":
        if image_class in ("grayscale-like", "binary/icon-like"):
            gray.save(buffer, format="WEBP", lossless=True, method=6)
        else:
            rgb.save(buffer, format="WEBP", lossless=True, method=6)
        return buffer.getvalue(), ext

    if method.startswith("webp_q"):
        quality = int(method.split("q", 1)[1])
        if image_class in ("grayscale-like", "binary/icon-like"):
            gray.save(buffer, format="WEBP", quality=quality, method=6)
        else:
            rgb.save(buffer, format="WEBP", quality=quality, method=6)
        return buffer.getvalue(), ext

    if method.startswith("jpeg_q"):
        quality = int(method.split("q", 1)[1])
        if image_class in ("grayscale-like", "binary/icon-like"):
            gray.save(buffer, format="JPEG", quality=quality, optimize=True, progressive=False, subsampling=0)
        else:
            rgb.save(buffer, format="JPEG", quality=quality, optimize=True, progressive=False, subsampling=2)
        return buffer.getvalue(), ext

    raise ValueError(f"Unsupported image method: {method}")


def _safe_win_size(height: int, width: int) -> int:
    smallest = min(height, width)
    if smallest < 3:
        return 3
    if smallest % 2 == 1:
        return smallest if smallest <= 11 else 11
    adjusted = smallest - 1
    return adjusted if adjusted <= 11 else 11


def compute_ssim_local(original: np.ndarray, candidate: np.ndarray, grayscale: bool) -> float:
    if original.shape != candidate.shape:
        return 0.0

    h, w = original.shape[:2]
    if min(h, w) < 3:
        return 1.0 if np.array_equal(original, candidate) else 0.0

    win_size = _safe_win_size(h, w)
    kwargs: Dict[str, Any] = {
        "data_range": 255,
        "gaussian_weights": True,
        "sigma": 1.5,
        "use_sample_covariance": False,
        "win_size": win_size,
    }

    if grayscale:
        score = skimage_ssim(original.astype(np.float64), candidate.astype(np.float64), **kwargs)
    else:
        score = skimage_ssim(
            original.astype(np.float64),
            candidate.astype(np.float64),
            channel_axis=2,
            **kwargs,
        )
    return float(score)


def compute_image_quality_metrics(original_rgb: np.ndarray, candidate_rgb: np.ndarray, image_class: ImageClass) -> Dict[str, float]:
    if image_class == "binary/icon-like":
        original_gray = rgb_to_gray_float(original_rgb)
        candidate_gray = rgb_to_gray_float(candidate_rgb)
        original_mask = binary_mask(original_gray)
        candidate_mask = binary_mask(candidate_gray)
        return {
            "mask_iou": mask_iou(original_mask, candidate_mask),
            "edge_agreement": edge_agreement(original_mask, candidate_mask),
        }

    if image_class == "grayscale-like":
        original_gray = rgb_to_gray_float(original_rgb).astype(np.uint8)
        candidate_gray = rgb_to_gray_float(candidate_rgb).astype(np.uint8)
        return {"ssim_local": compute_ssim_local(original_gray, candidate_gray, grayscale=True)}

    return {"ssim_local": compute_ssim_local(original_rgb, candidate_rgb, grayscale=False)}


def render_quality_metric_text(metrics: Dict[str, float]) -> str:
    if "ssim_local" in metrics:
        return f"ssim_local={float(metrics['ssim_local']):.4f}"
    return (
        f"mask_iou={float(metrics.get('mask_iou', 0.0)):.4f}, "
        f"edge_agreement={float(metrics.get('edge_agreement', 0.0)):.4f}"
    )


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


class ImageBranch(CompressionBranch):
    name = "image"

    def can_handle(self, data_type: DataType) -> bool:
        return data_type == "Image"

    def process(self, uploaded_file) -> Tuple[FinalSelection, pd.DataFrame, Dict[str, bytes]]:
        original_bytes = uploaded_file.getvalue()
        original_name = uploaded_file.name
        original_image = load_image_from_bytes(original_bytes)
        original_rgb = image_to_rgb_array(original_image)
        image_class = classify_image(original_rgb)
        ladder = select_image_ladder(image_class)

        candidates: List[CandidateResult] = []
        files_for_download: Dict[str, bytes] = {}

        for method in ladder:
            encoded_bytes, extension = encode_image_candidate_bytes(original_image, method, image_class)
            candidate_image = load_image_from_bytes(encoded_bytes)
            candidate_rgb = image_to_rgb_array(candidate_image)

            candidate = CandidateResult(
                method=method,
                output_name=f"{Path(original_name).stem}_{method}{extension}",
                output_bytes=encoded_bytes,
                file_size_bytes=len(encoded_bytes),
                extension=extension,
                compression_ratio_x=len(original_bytes) / len(encoded_bytes) if len(encoded_bytes) > 0 else 0.0,
                property_display=f"{candidate_image.size[0]}×{candidate_image.size[1]}",
                integrity_ok=candidate_image.size == original_image.size,
                quality_metrics=compute_image_quality_metrics(original_rgb, candidate_rgb, image_class),
                status="candidate",
            )
            candidates.append(candidate)
            files_for_download[candidate.output_name] = candidate.output_bytes

        choose_smallest_candidate(candidates)

        selection = FinalSelection(
            branch="image",
            original_name=original_name,
            original_size_bytes=len(original_bytes),
            original_extension=Path(original_name).suffix.lower(),
            original_property=f"{original_image.size[0]}×{original_image.size[1]}",
            candidates=candidates,
        )
        return selection, build_results_dataframe(selection), files_for_download

    def render_original_preview(self, filename: str, data: bytes) -> None:
        st.subheader("Original File Preview")
        with st.container(border=True):
            try:
                image = load_image_from_bytes(data)
                preview_width = max(160, image.size[0] // 2)
                st.image(image, width=preview_width)
            except Exception:
                st.info(f"Preview is not available for this file type: {filename}")

    def render_compressed_preview(self, selected_name: str, files_for_download: Dict[str, bytes], original_name: str) -> None:
        st.subheader("Compressed File Preview")
        with st.container(border=True):
            image = load_image_from_bytes(files_for_download[selected_name])
            preview_width = max(160, image.size[0] // 2)
            st.image(image, width=preview_width)


_IMAGE_BRANCH = ImageBranch()


def process(uploaded_file) -> Tuple[FinalSelection, pd.DataFrame, Dict[str, bytes]]:
    return _IMAGE_BRANCH.process(uploaded_file)


def render_original_preview(filename: str, data: bytes) -> None:
    _IMAGE_BRANCH.render_original_preview(filename, data)


def render_compressed_preview(selected_name: str, files_for_download: Dict[str, bytes], original_name: str) -> None:
    _IMAGE_BRANCH.render_compressed_preview(selected_name, files_for_download, original_name)
