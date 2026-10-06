# pyright: basic
"""Numerical rendering regression checks shared by the CI backends."""

from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
from typing import Iterable, Mapping

from PIL import Image
import jax
import numpy as np

from examples.render_baselines import render_cube, render_head_frames

REFERENCE_DIR = Path(__file__).parent / "references"
ARTIFACT_DIR = Path(
    os.environ.get("JAXRENDERER_ARTIFACT_DIR", ".artifacts/render-regression")
)


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Clip normalized RGB values and round to the nearest 8-bit channel."""
    return np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def read_reference(name: str) -> np.ndarray:
    with Image.open(REFERENCE_DIR / name) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def foreground(
    image: np.ndarray, threshold: int, white_background: bool = False
) -> np.ndarray:
    if white_background:
        return np.max(np.abs(image.astype(np.int16) - 255), axis=2) > threshold
    return np.max(image, axis=2) > threshold


def compare_images(
    name: str,
    actual: np.ndarray,
    expected: np.ndarray,
    threshold: int,
    white_background: bool = False,
) -> dict[str, float]:
    assert actual.shape == expected.shape, (
        f"{name}: expected shape {expected.shape}, got {actual.shape}"
    )
    expected_mask = foreground(expected, threshold, white_background)
    actual_mask = foreground(actual, threshold, white_background)
    union = expected_mask | actual_mask
    intersection = expected_mask & actual_mask
    iou = float(intersection.sum() / max(union.sum(), 1))
    pixel_max_difference = np.max(
        np.abs(actual.astype(np.int16) - expected.astype(np.int16)), axis=2
    )
    union_channel_mae = float(
        np.abs(actual.astype(np.int16) - expected.astype(np.int16))[union].mean()
    )
    union_p95 = float(np.percentile(pixel_max_difference[union], 95))
    metrics = {
        "foreground_iou": iou,
        "union_channel_mae": union_channel_mae,
        "union_pixel_max_abs_p95": union_p95,
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    diff = np.minimum(pixel_max_difference * 4, 255).astype(np.uint8)
    Image.fromarray(np.repeat(diff[:, :, None], 3, axis=2)).save(
        ARTIFACT_DIR / f"{name}-diff-x4.png"
    )
    return metrics


def assert_image_metrics(name: str, metrics: dict[str, float]) -> None:
    assert metrics["foreground_iou"] >= 0.90, (
        f"{name}: foreground IoU {metrics['foreground_iou']:.5f} < 0.90"
    )
    assert metrics["union_channel_mae"] <= 5.0, (
        f"{name}: union foreground channel MAE {metrics['union_channel_mae']:.3f} > 5"
    )
    assert metrics["union_pixel_max_abs_p95"] <= 20.0, (
        f"{name}: foreground p95 {metrics['union_pixel_max_abs_p95']:.1f} > 20"
    )


def save_image(name: str, image: np.ndarray) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(ARTIFACT_DIR / name)


def write_report(section: str, metrics: Mapping[str, object]) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = ARTIFACT_DIR / "numeric-report.json"
    report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.exists()
        else {}
    )
    report["environment"] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "jax": jax.__version__,
        "jaxlib": version("jaxlib"),
        "numpy": np.__version__,
    }
    report[section] = dict(metrics)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({section: metrics}, sort_keys=True))


def test_cube_matches_numeric_reference() -> None:
    actual = to_uint8(render_cube(width=640, height=480))
    metrics = compare_images(
        "cube", actual, read_reference("cube.png"), threshold=20, white_background=True
    )
    save_image("cube.png", actual)
    write_report("cube", metrics)
    assert_image_metrics("cube", metrics)


def test_head_animation_matches_references_and_smoothly_moves() -> None:
    frames: Iterable[np.ndarray] = render_head_frames(
        width=320, height=240, frame_count=30
    )
    rendered = [to_uint8(frame) for frame in frames]
    assert len(rendered) == 30, f"expected 30 frames, got {len(rendered)}"

    foreground_ratios = [float(foreground(frame, 8).mean()) for frame in rendered]
    adjacent_mae = [
        float(np.abs(rendered[index + 1].astype(np.int16) - rendered[index]).mean())
        for index in range(len(rendered) - 1)
    ]
    frame_0_metrics = compare_images(
        "head-frame-00", rendered[0], read_reference("head-frame-00.png"), 8
    )
    frame_15_metrics = compare_images(
        "head-frame-15", rendered[15], read_reference("head-frame-15.png"), 8
    )
    metrics: dict[str, object] = {
        "foreground_ratio_min": min(foreground_ratios),
        "foreground_ratio_max": max(foreground_ratios),
        "adjacent_frame_mae_min": min(adjacent_mae),
        "adjacent_frame_mae_max": max(adjacent_mae),
        "frame_0": frame_0_metrics,
        "frame_15": frame_15_metrics,
    }
    save_image("head-frame-00.png", rendered[0])
    save_image("head-frame-15.png", rendered[15])
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rendered[0], mode="RGB").save(
        ARTIFACT_DIR / "head-animation.png",
        save_all=True,
        append_images=[Image.fromarray(frame, mode="RGB") for frame in rendered[1:]],
        duration=100,
        loop=0,
        disposal=2,
        format="PNG",
    )
    write_report("head_animation", metrics)
    assert min(foreground_ratios) >= 0.15, (
        f"head foreground ratio minimum {min(foreground_ratios):.4f} < 0.15"
    )
    assert max(foreground_ratios) <= 0.27, (
        f"head foreground ratio maximum {max(foreground_ratios):.4f} > 0.27"
    )
    assert min(adjacent_mae) >= 0.5, (
        f"head adjacent-frame MAE minimum {min(adjacent_mae):.4f} < 0.5"
    )
    assert max(adjacent_mae) <= 5.0, (
        f"head adjacent-frame MAE maximum {max(adjacent_mae):.4f} > 5"
    )
    assert_image_metrics("head-frame-00", frame_0_metrics)
    assert_image_metrics("head-frame-15", frame_15_metrics)
