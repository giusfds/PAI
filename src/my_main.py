from dataclasses import dataclass
from PIL import Image, ImageOps, ImageTk
from enum import Enum, StrEnum
from pathlib import Path
import numpy as np
import re

# ============================================================================
# CONSTANTES
# ============================================================================

IMAGE_EXTENSIONS: set[str] = {".png", ".tif", ".tiff"}

TEST_SPLIT_MOD = 4

# ============================================================================
# CLASSE DE IMAGEM BASE
# ============================================================================

class SetType(Enum):
    TRAIN = "train"
    TEST = "test"

class SampleSide(Enum):
    LEFT = "L"
    RIGHT = "R"

class SampleView(Enum):
    CC = "CC"
    MLO = "MLO"

class SampleBiradsClass(StrEnum):
    D = "BIRADS I"
    E = "BIRADS II"
    F = "BIRADS III"
    G = "BIRADS IV"

@dataclass
class Sample:
    path: Path
    filename: str
    birads_class: SampleBiradsClass
    number: int
    set_type: SetType
    side: SampleSide
    view: SampleView

class DatasetManager:
    def __init__(self):
        self._dataset_dir: Path | None = None
        self.samples: list[Sample] = []
        self.train_samples: list[Sample] = []
        self.test_samples: list[Sample] = []

    def load_dataset(self, dataset_dir_path: str | Path) -> None:
        self._dataset_dir = Path(dataset_dir_path) if isinstance(dataset_dir_path, str) else dataset_dir_path

        self._validate_directory(self._dataset_dir)
        self._clear_old_data()

        for file in self._dataset_dir.rglob("*"):
            if not file.is_file():
                continue

            if file.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            sample = self._create_sample(file)
            print("Sample criado:", sample)
            self.samples.append(sample)

            if sample.set_type == SetType.TRAIN:
                self.train_samples.append(sample)
            else:
                self.test_samples.append(sample)

    def _validate_directory(self, path: Path) -> None:
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(f"Diretório {path} não encontrado ou não é um diretório válido.")
    
    def _clear_old_data(self) -> None:
        self.samples.clear()
        self.train_samples.clear()
        self.test_samples.clear()

    def _create_sample(self, file: Path) -> Sample:
        birads_class = self._get_birads_class(file.name)

        number = self._get_image_number(file.name)

        side = self._get_side(file.name)

        view = self._get_view(file.name)

        set_type = SetType.TEST if number % TEST_SPLIT_MOD == 0 else SetType.TRAIN

        return Sample(
            path=file,
            filename=file.name,
            birads_class=birads_class,
            number=number,
            set_type=set_type,
            side=side,
            view=view
        )

    def _get_birads_class(self, filename: str) -> SampleBiradsClass:
        return SampleBiradsClass[filename[0].upper()]

    def _get_image_number(self, filename: str) -> int:
        match = re.search(r"\((\d+)\)", filename)

        if not match:
            return 1

        return int(match.group(1))

    def _get_side(self, filename: str) -> SampleSide:
        low = filename.lower()
        if "left" in low:
            return SampleSide.LEFT
        if "right" in low:
            return SampleSide.RIGHT
        return SampleSide.RIGHT

    def _get_view(self, filename: str) -> SampleView:
        up = filename.lower()
        if "cc" in up:
            return SampleView.CC
        if "mlo" in up:
            return SampleView.MLO
        return SampleView.CC
# DatasetManager

class ImageManager:
    @staticmethod
    def load_image(sample: Sample) -> np.ndarray:
        image = Image.open(sample.path)
        return np.array(image)


def main():
    dataset_manager = DatasetManager()
    image_manager = ImageManager()
    dataset_manager.load_dataset("dataset")

    for sample in dataset_manager.train_samples[:5]:
        image = image_manager.load_image(sample)
        print(sample.filename)
        print(image.shape)
        print(image.dtype)
        print(image.min(), image.max())

if __name__ == "__main__":
    main()
