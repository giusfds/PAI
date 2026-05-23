from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import main as app  # noqa: E402


def test_load_image_normalizes_8_and_16_bit(tmp_path: Path) -> None:
    eight_path = tmp_path / "eight.png"
    sixteen_path = tmp_path / "sixteen.tif"
    Image.fromarray(np.array([[0, 255]], dtype=np.uint8)).save(eight_path)
    Image.fromarray(np.array([[0, 4095]], dtype=np.uint16)).save(sixteen_path)

    eight = app.MammogramImage(eight_path)
    sixteen = app.MammogramImage(sixteen_path)
    app.ImageProcessor.load_image(eight)
    app.ImageProcessor.load_image(sixteen)

    assert eight.original is not None
    assert sixteen.original is not None
    assert np.isclose(float(eight.original.max()), 1.0)
    assert np.isclose(float(sixteen.original.max()), 1.0)
    assert eight.bit_depth == 8
    assert sixteen.bit_depth in {16, 32}


def test_natural_number_and_split_rule(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    class_dir = dataset / "D + right + CC"
    class_dir.mkdir(parents=True)
    for number in [1, 4, 8]:
        Image.fromarray(np.ones((2, 2), dtype=np.uint8)).save(class_dir / f"d_right_cc ({number}).png")

    records = app.DatasetManager.discover_dataset(dataset)
    by_number = {record.number: record for record in records}

    assert app.DatasetManager._natural_number_from_name(Path("x (123).png")) == 123
    assert by_number[1].split == "train"
    assert by_number[4].split == "test"
    assert by_number[8].split == "test"


def test_discover_dataset_is_recursive(tmp_path: Path) -> None:
    image_path = tmp_path / "RCC" / "nested" / "G + right + CC" / "g_right_cc (12).png"
    image_path.parent.mkdir(parents=True)
    Image.fromarray(np.ones((2, 2), dtype=np.uint8)).save(image_path)

    records = app.DatasetManager.discover_dataset(tmp_path / "RCC")

    assert len(records) == 1
    assert records[0].class_name == "G"
    assert records[0].class_index == app.CLASS_TO_INDEX["G"]
    assert records[0].split == "test"


def test_largest_component_fallback_without_cv2() -> None:
    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[0, 0] = 1
    mask[2:5, 2:5] = 1

    cleaned = app.ImageProcessor._largest_component_fallback(mask)

    assert int(cleaned.sum()) == 9
    assert cleaned[0, 0] == 0
    assert cleaned[3, 3] == 1


def test_segment_breast_fallback_without_cv2(monkeypatch) -> None:
    monkeypatch.setattr(app, "cv2", None)
    image_obj = app.MammogramImage(Path("synthetic.png"))
    image_obj.preprocessed = np.zeros((10, 10), dtype=np.float32)
    image_obj.preprocessed[2:8, 2:8] = 0.8

    app.ImageProcessor.segment_breast(image_obj)

    assert image_obj.mask is not None
    assert image_obj.segmented is not None
    assert image_obj.segmented_ready is True
    assert float(image_obj.segmented.max()) > 0.0
